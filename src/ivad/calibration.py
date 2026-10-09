"""Empirical threshold calibration on held-out normal images only.

For a target false-positive rate alpha and n calibration normal scores, the
threshold is the k-th smallest calibration score with

    k = ceil((n + 1) * (1 - alpha)),   flag "REVIEW" iff score > threshold.

This is the split-conformal order statistic. If calibration and future normal
scores are exchangeable (same distribution, no drift), then over random draws
of the calibration set:

    E[FPR] = (n + 1 - k) / (n + 1) <= alpha
    FPR | threshold  ~  Beta(n + 1 - k, k)      (continuous scores)

The Beta law quantifies how far the *actual* FPR of a threshold estimated from
only n normals can sit from its target. It says nothing about drift: if line
conditions change, the guarantee does not apply.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


def conformal_rank(n: int, alpha: float) -> int:
    if n < 1:
        raise ValueError("need at least one calibration score")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)  # tolerance for float round-off
    if k > n:
        raise ValueError(f"alpha={alpha} is too small for n={n} calibration scores (need n >= {k})")
    return k


def conformal_threshold(scores: Sequence[float], alpha: float) -> tuple[float, int]:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("calibration scores must be a finite 1-D array")
    k = conformal_rank(len(s), alpha)
    return float(s[k - 1]), k


def fpr_distribution(n: int, k: int, level: float = 0.95) -> dict[str, float]:
    """Mean and central interval of Beta(n+1-k, k): FPR induced by calibration sampling."""
    a, b = n + 1 - k, k
    lo, hi = stats.beta.ppf([(1 - level) / 2, (1 + level) / 2], a, b)
    return {"expected": a / (n + 1), "lower": float(lo), "upper": float(hi), "level": level}


def calibrate(calib_scores: Sequence[float], candidates: dict[str, float], level: float = 0.95) -> dict[str, Any]:
    """candidates: name -> target FPR alpha. Returns per-candidate thresholds."""
    n = len(calib_scores)
    out = {}
    for name, alpha in candidates.items():
        t, k = conformal_threshold(calib_scores, alpha)
        out[name] = {
            "target_fpr": alpha,
            "threshold": t,
            "rank_k": k,
            "n_calib": n,
            "calib_fpr_observed": float(np.mean(np.asarray(calib_scores) > t)),
            "fpr_due_to_calibration_sampling": fpr_distribution(n, k, level),
        }
    return out


def flag(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Review recommendation: True = REVIEW."""
    return np.asarray(scores) > threshold
