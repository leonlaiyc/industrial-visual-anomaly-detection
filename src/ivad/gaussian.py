"""Per-position Gaussian model of normal patch features and Mahalanobis scoring.

This module is implemented in the project (no Anomalib). For every spatial
position p of the feature grid we estimate, from the fit images only,

    mu_p    = mean of the d-dim patch features at p
    Sigma_p = sample covariance at p                          (unbiased, N-1)
    S_p     = Sigma_p + (lambda * mean(diag Sigma_p) + eps) * I  (regularised)

and score a new patch feature x at p by its Mahalanobis distance

    D_p(x) = sqrt( (x - mu_p)^T S_p^{-1} (x - mu_p) ).

Numerics:
  * Moments are accumulated in float64 with a *shifted* (pseudo-mean) sum, so
    the covariance does not suffer the cancellation of the naive E[xx]-E[x]^2.
  * S_p^{-1} is never formed explicitly. We take the Cholesky factor
    S_p = L L^T and store the whitening matrix W = L^{-1}; then
    D_p(x)^2 = || W (x - mu_p) ||^2, which is a batched mat-vec at score time.
  * The ridge term makes S_p positive definite even when Sigma_p is singular
    (e.g. N <= d, or constant channels); a failed Cholesky raises loudly.

No Gaussianity is assumed to *hold*: the distance is used as an anomaly score
whose operating threshold is calibrated empirically on held-out normals.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class CovarianceAccumulator:
    """Streaming per-position mean/covariance in float64 (shifted-data method)."""

    def __init__(self) -> None:
        self.n = 0
        self.shift: torch.Tensor | None = None  # (P, d)
        self.s1: torch.Tensor | None = None  # (P, d)      sum of (x - shift)
        self.s2: torch.Tensor | None = None  # (P, d, d)   sum of (x - shift)(x - shift)^T
        self.grid: tuple[int, int] | None = None

    def update(self, feats: torch.Tensor) -> None:
        """feats: (B, d, h, w) patch features of normal images."""
        if feats.ndim != 4:
            raise ValueError(f"expected (B, d, h, w), got {tuple(feats.shape)}")
        b, d, h, w = feats.shape
        x = feats.permute(2, 3, 0, 1).reshape(h * w, b, d).double()  # (P, B, d)
        if self.shift is None:
            self.grid = (h, w)
            self.shift = x.mean(dim=1)
            self.s1 = torch.zeros_like(self.shift)
            self.s2 = torch.zeros(h * w, d, d, dtype=torch.float64, device=x.device)
        elif (h, w) != self.grid or d != self.shift.shape[1]:
            raise ValueError("feature shape changed between batches")
        xc = x - self.shift[:, None, :]
        self.s1 += xc.sum(dim=1)
        self.s2 += xc.transpose(1, 2) @ xc
        self.n += b

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean (P, d), unbiased covariance (P, d, d)), both float64."""
        if self.n < 2:
            raise ValueError("need at least 2 samples to estimate a covariance")
        m = self.s1 / self.n
        cov = (self.s2 - self.n * m[:, :, None] * m[:, None, :]) / (self.n - 1)
        cov = 0.5 * (cov + cov.transpose(1, 2))  # remove round-off asymmetry
        return self.shift + m, cov


def regularize(cov: torch.Tensor, rel: float, eps: float) -> torch.Tensor:
    """Ridge: add (rel * mean diagonal + eps) * I per position. cov: (..., d, d)."""
    if rel < 0 or eps < 0 or (rel == 0 and eps == 0):
        raise ValueError("regularisation must be non-negative and not both zero")
    d = cov.shape[-1]
    scale = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (...,)
    ridge = (rel * scale + eps)[..., None, None] * torch.eye(d, dtype=cov.dtype, device=cov.device)
    return cov + ridge


def whitening_matrices(cov_reg: torch.Tensor) -> torch.Tensor:
    """W = L^{-1} with L the lower Cholesky factor of cov_reg. Shape (..., d, d)."""
    L, info = torch.linalg.cholesky_ex(cov_reg)
    if (info != 0).any():
        bad = int((info != 0).sum())
        raise torch.linalg.LinAlgError(f"{bad} covariance matrices are not positive definite")
    eye = torch.eye(cov_reg.shape[-1], dtype=cov_reg.dtype, device=cov_reg.device)
    return torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)


def mahalanobis(x: torch.Tensor, mean: torch.Tensor, whitener: torch.Tensor) -> torch.Tensor:
    """Distances for x: (B, P, d), mean: (P, d), whitener: (P, d, d) -> (B, P)."""
    if x.shape[1:] != mean.shape or whitener.shape != mean.shape + mean.shape[-1:]:
        raise ValueError(
            f"shape mismatch x={tuple(x.shape)} mean={tuple(mean.shape)} W={tuple(whitener.shape)}"
        )
    diff = (x - mean[None]).permute(1, 2, 0)  # (P, d, B)
    # One bmm with positions as the batch dim; avoids materialising B copies of W.
    z = torch.bmm(whitener, diff)  # (P, d, B)
    return z.pow(2).sum(1).clamp_min(0).sqrt().T  # (B, P)


@dataclass
class GaussianPatchModel:
    mean: torch.Tensor  # (P, d) float32
    whitener: torch.Tensor  # (P, d, d) float32
    grid: tuple[int, int]
    n_fit: int

    @classmethod
    def from_accumulator(cls, acc: CovarianceAccumulator, rel: float, eps: float) -> "GaussianPatchModel":
        mean, cov = acc.finalize()
        W = whitening_matrices(regularize(cov, rel, eps))
        return cls(mean.float(), W.float(), acc.grid, acc.n)

    def patch_distances(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, d, h, w) features -> (B, h, w) Mahalanobis distances."""
        b, d, h, w = feats.shape
        if (h, w) != tuple(self.grid) or d != self.mean.shape[1]:
            raise ValueError(f"features {tuple(feats.shape)} do not match model grid {self.grid}, d={self.mean.shape[1]}")
        x = feats.permute(0, 2, 3, 1).reshape(b, h * w, d).to(self.mean.dtype)
        return mahalanobis(x, self.mean, self.whitener).reshape(b, h, w)

    def to(self, device: torch.device) -> "GaussianPatchModel":
        return GaussianPatchModel(self.mean.to(device), self.whitener.to(device), self.grid, self.n_fit)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "whitener": self.whitener.cpu(),
                "grid": list(self.grid), "n_fit": self.n_fit}

    @classmethod
    def from_state_dict(cls, s: dict) -> "GaussianPatchModel":
        return cls(s["mean"], s["whitener"], tuple(s["grid"]), int(s["n_fit"]))


def anomaly_map(dist: torch.Tensor, out_hw: tuple[int, int], sigma: float) -> torch.Tensor:
    """Upsample patch distances (B, h, w) to (B, H, W) and Gaussian-smooth."""
    m = F.interpolate(dist[:, None], size=tuple(out_hw), mode="bilinear", align_corners=False)
    if sigma > 0:
        k = 2 * int(4 * sigma + 0.5) + 1
        m = gaussian_blur(m, kernel_size=[k, k], sigma=[sigma, sigma])
    return m[:, 0]


def image_score(amap: torch.Tensor) -> torch.Tensor:
    """Image-level score = maximum of the smoothed anomaly map. (B, H, W) -> (B,)."""
    return amap.flatten(1).amax(dim=1)
