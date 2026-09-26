"""Random-head controls for the STE Top-k cardinality sweep.

The Top-k curve alone cannot separate "these particular heads carry the
position decision" from "patching any k heads at this depth disturbs the
answer".  The control here draws k heads uniformly from the same candidate
pool the STE search ranged over, so the only difference from the Top-k arm is
which heads were chosen.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DrawStatistics:
    """Mean and 95% interval of one rate across independent random draws."""

    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    minimum: float
    maximum: float


# Two-sided 0.975 Student-t quantiles for 1..29 degrees of freedom. The sweep
# uses a handful of draws, where the normal quantile is visibly too narrow, and
# this keeps the module free of a SciPy import at plot time.
_T_QUANTILES = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
)


def _t_quantile(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees of freedom must be positive")
    if degrees_of_freedom <= len(_T_QUANTILES):
        return _T_QUANTILES[degrees_of_freedom - 1]
    return 1.96


def candidate_components(
    *, layers: Sequence[int], n_heads: int
) -> list[tuple[int, int]]:
    """Every (layer, head) the STE search could have selected."""

    normalized_layers = [int(layer) for layer in layers]
    if not normalized_layers:
        raise ValueError("candidate pool needs at least one layer")
    if normalized_layers != sorted(set(normalized_layers)):
        raise ValueError("candidate layers must be sorted and unique")
    if int(n_heads) <= 0:
        raise ValueError("candidate pool needs a positive head count")
    return [
        (layer, head)
        for layer in normalized_layers
        for head in range(int(n_heads))
    ]


def sample_random_components(
    *,
    layers: Sequence[int],
    n_heads: int,
    top_k: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Draw ``top_k`` distinct heads uniformly from the candidate pool.

    Sampling is uniform over the whole pool rather than matched to the Top-k
    layer histogram, so the control does not inherit the depth preference of
    the STE solution. The returned order is sorted, which makes the draw a
    function of ``seed`` alone and not of the enumeration order.
    """

    pool = candidate_components(layers=layers, n_heads=n_heads)
    if not 0 < int(top_k) <= len(pool):
        raise ValueError(
            f"top_k must be in 1..{len(pool)} for this candidate pool, "
            f"got {top_k}"
        )
    generator = random.Random(int(seed))
    return sorted(generator.sample(pool, int(top_k)))


def components_to_mask(
    components: Sequence[tuple[int, int]],
    *,
    layers: Sequence[int],
    n_heads: int,
) -> torch.Tensor:
    """Build the [len(layers), n_heads] binary mask for ``components``."""

    normalized_layers = [int(layer) for layer in layers]
    offsets = {layer: index for index, layer in enumerate(normalized_layers)}
    mask = torch.zeros(len(normalized_layers), int(n_heads), dtype=torch.bool)
    seen: set[tuple[int, int]] = set()
    for raw_layer, raw_head in components:
        layer, head = int(raw_layer), int(raw_head)
        if layer not in offsets:
            raise ValueError(f"component layer {layer} is outside the mask layers")
        if not 0 <= head < int(n_heads):
            raise ValueError(f"component head {head} is outside the mask heads")
        if (layer, head) in seen:
            raise ValueError(f"duplicate component: {(layer, head)}")
        seen.add((layer, head))
        mask[offsets[layer], head] = True
    return mask


def draw_seed(*, base_seed: int, top_k: int, draw: int) -> int:
    """Derive one reproducible per-draw seed.

    The seed depends on the cardinality and the draw index but not on the
    gate direction: a random head set is a property of the heads, so both
    directions are evaluated on the same draws and their curves stay paired.
    """

    if int(draw) < 0:
        raise ValueError("draw index must be non-negative")
    return (int(base_seed) * 1_000_003 + int(top_k) * 1_009 + int(draw)) % (2**31 - 1)


def summarize_draws(values: Sequence[float]) -> DrawStatistics:
    """Mean and Student-t 95% interval of the mean across draws."""

    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize zero draws")
    count = len(samples)
    mean = sum(samples) / count
    if count == 1:
        return DrawStatistics(
            n=1,
            mean=mean,
            std=0.0,
            ci_low=mean,
            ci_high=mean,
            minimum=mean,
            maximum=mean,
        )
    variance = sum((value - mean) ** 2 for value in samples) / (count - 1)
    std = math.sqrt(variance)
    half_width = _t_quantile(count - 1) * std / math.sqrt(count)
    return DrawStatistics(
        n=count,
        mean=mean,
        std=std,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
        minimum=min(samples),
        maximum=max(samples),
    )
