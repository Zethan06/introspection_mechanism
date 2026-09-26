"""Paired attention-distribution effects of removing Q/K score terms."""

import torch


def attention_kl_metrics(
    scores: torch.Tensor, removed: torch.Tensor, positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure KL(original || ablated), in nats, per trial and head.

    Scores/removed have shape [batch, head, visible_context]. Positions selects
    a fixed proper subset of that context. Subset KL conditions each distribution
    on membership; coarse KL instead retains those positions plus one outside
    bin. All softmax/log operations use float64, without probability clipping.
    """
    if scores.ndim != 3 or scores.shape != removed.shape:
        raise ValueError('scores and removed must have equal [batch, head, context] shapes')
    if not torch.isfinite(scores).all() or not torch.isfinite(removed).all():
        raise ValueError('expected finite scores on visible positions')
    positions = torch.as_tensor(positions, dtype=torch.long, device=scores.device)
    context = scores.shape[-1]
    if (positions.ndim != 1 or not 0 < positions.numel() < context
            or positions.unique().numel() != positions.numel()
            or int(positions.min()) < 0 or int(positions.max()) >= context):
        raise ValueError('positions must be a nonempty unique proper subset of the context')
    original = scores.double()
    ablated = original - removed.double()
    lp, lq = original.log_softmax(-1), ablated.log_softmax(-1)
    p, q = lp.exp(), lq.exp()

    def kl(a, b):
        value = (a.exp() * (a - b)).sum(-1)
        if (value < -1e-10).any():
            raise RuntimeError('negative KL beyond floating-point tolerance')
        return value.clamp_min(0)

    sp, sq = lp[..., positions], lq[..., positions]
    log_mp, log_mq = sp.logsumexp(-1), sq.logsumexp(-1)
    cp, cq = sp - log_mp.unsqueeze(-1), sq - log_mq.unsqueeze(-1)
    outside = torch.ones(context, dtype=torch.bool, device=scores.device)
    outside[positions] = False
    coarse_p = torch.cat((sp, lp[..., outside].logsumexp(-1, keepdim=True)), -1)
    coarse_q = torch.cat((sq, lq[..., outside].logsumexp(-1, keepdim=True)), -1)
    mp, mq = log_mp.exp(), log_mq.exp()
    return {
        'full_kl': kl(lp, lq),
        'full_tv': 0.5 * (p - q).abs().sum(-1),
        'subset_conditional_kl': kl(cp, cq),
        'subset_conditional_tv': 0.5 * (cp.exp() - cq.exp()).abs().sum(-1),
        'subset_coarse_kl': kl(coarse_p, coarse_q),
        'subset_native_mass': mp,
        'subset_ablated_mass': mq,
        'subset_mass_delta': mq - mp,
        'subset_mass_abs_delta': (mq - mp).abs(),
    }
