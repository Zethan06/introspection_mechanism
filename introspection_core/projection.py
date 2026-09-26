"""CPU-only PCA used by the latent and head-output galleries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ProjectionResult:
    coords: torch.Tensor
    method: str
    n_components: int
    variance_ratio: list[float] | None = None
    extra: dict = field(default_factory=dict)


def pca(x: torch.Tensor, *, n_components: int = 3, niter: int = 4, seed: int = 0) -> ProjectionResult:
    x = x.detach().to("cpu", dtype=torch.float32)
    centered = x - x.mean(dim=0, keepdim=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _u, s, v = torch.pca_lowrank(centered, q=n_components, center=False, niter=niter)
    coords = centered @ v[:, :n_components]
    denom = float((centered.square().sum() / max(centered.shape[0] - 1, 1)).item())
    if denom <= 0:
        variance_ratio = [float("nan")] * n_components
    else:
        eigvals = (s[:n_components].square() / max(centered.shape[0] - 1, 1)).tolist()
        variance_ratio = [eigval / denom for eigval in eigvals]
    return ProjectionResult(
        coords=coords,
        method="pca",
        n_components=n_components,
        variance_ratio=variance_ratio,
    )


_METHODS = {"pca": pca}


def project(x: torch.Tensor, *, method: str = "pca", n_components: int = 3, seed: int = 0, **kwargs) -> ProjectionResult:
    if method not in _METHODS:
        raise ValueError(f"Unsupported projection method: {method!r} (expected one of {sorted(_METHODS)})")
    x = x.detach().to("cpu", dtype=torch.float32)
    return pca(x, n_components=n_components, seed=seed, **kwargs)
