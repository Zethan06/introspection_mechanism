"""Unprojected OV modes using an exact reduced output-space factorization."""
from __future__ import annotations

import torch


def output_qr(weights: torch.Tensor) -> torch.Tensor:
    """Return R for W_O=QR; weights use repository [head, value, model] order.

    Q is an isometry, so R deltaV.T has the same singular values and right
    singular vectors as W_O deltaV.T, without materializing model-width outputs.
    """
    if weights.ndim != 3 or weights.shape[-1] < weights.shape[-2]:
        raise ValueError('weights must be [head, value_width, model_width>=value_width]')
    if not torch.isfinite(weights).all():
        raise ValueError('nonfinite output weights')
    return torch.linalg.qr(weights.transpose(-1, -2), mode='reduced').R


@torch.inference_mode()
def output_svd_metrics(
    a0: torch.Tensor, ai: torch.Tensor, v0: torch.Tensor, vi: torch.Tensor,
    output_r: torch.Tensor, *, heads: list[int], n_heads: int,
    positions: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Full M=W_O deltaV.T SVD statistics [batch, selected head], no report axis.

    Compute SVD of (R deltaV.T).T. Its left vectors are positional modes.
    Subset attention preserves native weights without renormalization.
    Zero-denominator statistics are zero with explicit validity indicators.
    """
    if v0.ndim != 4 or v0.shape != vi.shape or a0.shape != ai.shape:
        raise ValueError('incompatible clean/injected inputs')
    batch, length, kv, width = v0.shape
    if (length < 2 or n_heads % kv or len(set(heads)) != len(heads)
            or any(h < 0 or h >= n_heads for h in heads)
            or ai.shape != (batch, len(heads), length)
            or output_r.shape != (len(heads), width, width)):
        raise ValueError('incompatible GQA/attention/output dimensions')
    device, dtype = output_r.device, output_r.dtype
    if not all(torch.isfinite(x).all() for x in (a0, ai, v0, vi, output_r)):
        raise ValueError('nonfinite SVD inputs')
    a0, ai, v0, vi = [x.to(device=device, dtype=dtype) for x in (a0, ai, v0, vi)]
    index = torch.tensor(heads, device=device) // (n_heads // kv)
    dv = (vi-v0).index_select(2, index).transpose(1, 2)
    if positions is not None:
        if len(positions) < 2 or len(set(positions)) != len(positions) or min(positions)<0 or max(positions)>=length:
            raise ValueError('invalid source subset')
        dv = dv[:, :, positions]
        a0, ai = a0[:, :, positions], ai[:, :, positions]
    # x is [batch, head, source, reduced_output_width].
    x = dv @ output_r.transpose(-1, -2)
    # CPU LAPACK or CUDA's accurate Jacobi driver (not approximate gesvda).
    kwargs = {'driver': 'gesvdj'} if x.is_cuda else {}
    u, s, vh = torch.linalg.svd(x, full_matrices=False, **kwargs)
    reading = torch.einsum('bhnk,bhn->bhk', u, ai)
    clean_reading = torch.einsum('bhnk,bhn->bhk', u, a0)
    coeff = s * reading
    clean_coeff = s * clean_reading
    interaction_coeff = coeff-clean_coeff
    direct = torch.einsum('bhnd,bhn->bhd', x, ai)
    rebuilt = torch.einsum('bhk,bhkd->bhd', coeff, vh)
    error = (rebuilt-direct).norm(dim=-1)
    response = direct.norm(dim=-1)
    scale = x.norm(dim=(-1,-2))*ai.norm(dim=-1)
    closure_scaled = error / scale.clamp_min(torch.finfo(dtype).tiny)
    tolerance = 1e-9 if dtype == torch.float64 else 2e-5
    if (closure_scaled > tolerance).any():
        raise RuntimeError(f'OV SVD reconstruction failed: {closure_scaled.max().item()}')
    matrix_energy, response_energy = s.square(), coeff.square()
    matrix_total, response_total = matrix_energy.sum(-1), response_energy.sum(-1)
    attention_norm = ai.norm(dim=-1)
    positive_matrix = matrix_total > 0
    positive_response = response_total > 0
    valid_mode = s > (torch.finfo(dtype).eps * max(x.shape[-2:]) * s[..., :1])
    tiny = torch.finfo(dtype).tiny
    def fraction(num, den):
        return torch.where(den>0, num/den.clamp_min(tiny), torch.zeros_like(num))
    metrics = dict(attention_norm=attention_norm, attention_mass=ai.sum(-1),
        clean_attention_norm=a0.norm(dim=-1), delta_attention_norm=(ai-a0).norm(dim=-1),
        frobenius=matrix_total.sqrt(), sigma1=s[...,0], sigma2=s[...,1],
        sigma1_over_sigma2=fraction(s[...,0],s[...,1]), sigma_ratio_valid=(s[...,1]>0).to(dtype),
        matrix_nonzero=positive_matrix.to(dtype), response_nonzero=positive_response.to(dtype),
        attention_nonzero=(attention_norm>0).to(dtype),
        reading1_abs=torch.where(valid_mode[...,0],reading[...,0].abs(),0.),
        reading1_cos_abs=torch.where(valid_mode[...,0],fraction(reading[...,0].abs(),attention_norm),0.),
        response_norm=response, first_mode_norm=coeff[...,0].abs(),
        clean_response_norm=clean_coeff.norm(dim=-1), interaction_response_norm=interaction_coeff.norm(dim=-1),
        modal_norm_sum=coeff.abs().sum(-1), closure_absolute=error, closure_scaled=closure_scaled)
    for k in (1,3,5,10):
        metrics[f'energy_top{k}']=fraction(matrix_energy[...,:k].sum(-1),matrix_total)
        metrics[f'response_energy_top{k}']=fraction(response_energy[...,:k].sum(-1),response_total)
        metrics[f'response_norm_top{k}']=coeff[...,:k].norm(dim=-1)
        metrics[f'clean_response_norm_top{k}']=clean_coeff[...,:k].norm(dim=-1)
        metrics[f'interaction_response_norm_top{k}']=interaction_coeff[...,:k].norm(dim=-1)
    for k in range(min(10,s.shape[-1])):
        metrics[f'mode{k+1}_sigma']=s[...,k]
        metrics[f'mode{k+1}_valid']=valid_mode[...,k].to(dtype)
        metrics[f'mode{k+1}_reading_abs']=torch.where(valid_mode[...,k],reading[...,k].abs(),0.)
        metrics[f'mode{k+1}_reading_cos_abs']=torch.where(valid_mode[...,k],fraction(reading[...,k].abs(),attention_norm),0.)
        metrics[f'mode{k+1}_response_norm']=coeff[...,k].abs()
        metrics[f'mode{k+1}_response_energy_fraction']=fraction(response_energy[...,k],response_total)
        metrics[f'mode{k+1}_clean_response_norm']=clean_coeff[...,k].abs()
        metrics[f'mode{k+1}_interaction_response_norm']=interaction_coeff[...,k].abs()
    names = list(metrics)
    values = torch.stack([metrics[name] for name in names]).cpu()
    return dict(zip(names, values.unbind(0)))


@torch.inference_mode()
def topk_content_writes(
    ai: torch.Tensor, v0: torch.Tensor, vi: torch.Tensor,
    output_q: torch.Tensor, output_r: torch.Tensor, *,
    heads: list[int], n_heads: int, top_k: int, top_ks: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct full-context top-k content writes in actual output space.

    Singular modes are ranked by matrix singular value, not response magnitude.
    Q/R are the reduced QR factors of W_O for each selected query head.
    Returns [batch, selected head, model width] top/rest/full vectors and
    [batch, selected head] reconstruction error and retained response energy.
    """
    if v0.ndim != 4 or v0.shape != vi.shape:
        raise ValueError('incompatible value shapes')
    batch, length, kv, width = v0.shape
    if (n_heads % kv or ai.shape != (batch,len(heads),length)
            or output_r.shape != (len(heads),width,width)
            or output_q.ndim != 3 or output_q.shape[0] != len(heads)
            or output_q.shape[-1] != width or len(set(heads)) != len(heads)
            or any(h<0 or h>=n_heads for h in heads)
            or not 1 <= top_k <= min(length,width)):
        raise ValueError('invalid top-k/GQA/factor dimensions')
    ranks = sorted(set([top_k] + (top_ks or [])))
    if any(not 1 <= k <= min(length, width) for k in ranks):
        raise ValueError("invalid additional top-k rank")
    device,dtype=output_r.device,output_r.dtype
    ai,v0,vi,output_q=[x.to(device=device,dtype=dtype) for x in (ai,v0,vi,output_q)]
    if not all(torch.isfinite(x).all() for x in (ai,v0,vi,output_q,output_r)):
        raise ValueError('nonfinite modal intervention input')
    index=torch.tensor(heads,device=device)//(n_heads//kv)
    dv=(vi-v0).index_select(2,index).transpose(1,2)
    x=dv @ output_r.transpose(-1,-2)
    kwargs={'driver':'gesvdj'} if x.is_cuda else {}
    u,s,vh=torch.linalg.svd(x,full_matrices=False,**kwargs)
    coefficients=s*torch.einsum('bhnk,bhn->bhk',u,ai)
    top=torch.einsum('bhk,bhkd->bhd',coefficients[...,:top_k],vh[...,:top_k,:])
    direct=torch.einsum('bhn,bhnd->bhd',ai,x)
    rebuilt=torch.einsum('bhk,bhkd->bhd',coefficients,vh)
    scale=x.norm(dim=(-1,-2))*ai.norm(dim=-1)
    error=(rebuilt-direct).norm(dim=-1)/scale.clamp_min(torch.finfo(dtype).tiny)
    tolerance=1e-9 if dtype==torch.float64 else 2e-5
    if (error>tolerance).any():
        raise RuntimeError(f'modal write reconstruction failed: {error.max().item()}')
    top_output=torch.einsum('bhd,hmd->bhm',top,output_q)
    full_output=torch.einsum('bhd,hmd->bhm',direct,output_q)
    energy=coefficients.square().sum(-1)
    fraction=torch.where(energy>0,coefficients[...,:top_k].square().sum(-1)/energy.clamp_min(torch.finfo(dtype).tiny),0.)
    result = dict(top=top_output.cpu(), rest=(full_output-top_output).cpu(), full=full_output.cpu(),
                  closure_scaled=error.cpu(), response_energy_fraction=fraction.cpu())
    for k in ranks if top_ks is not None else []:
        reduced = torch.einsum("bhk,bhkd->bhd", coefficients[..., :k], vh[..., :k, :])
        result[f"top{k}"] = torch.einsum("bhd,hmd->bhm", reduced, output_q).cpu()
    return result
