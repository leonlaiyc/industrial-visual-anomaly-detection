"""Evaluation metrics with uncertainty.

* Recall and FPR: exact Clopper-Pearson binomial intervals (conservative;
  appropriate for the small counts here, and well defined at 0 or n).
* AUROC: Mann-Whitney statistic (ties count 1/2) with a *stratified*
  percentile bootstrap — anomalies and normals are resampled separately so
  every replicate keeps the test-set class sizes fixed.
* Paired robustness deltas: bootstrap over ORIGINAL images, so the perturbed
  copy of an image is always resampled together with its unperturbed copy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _labels_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays of equal length")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN/Inf")
    return y, s


def auroc(labels, scores) -> float:
    """P(score_anomaly > score_normal) + 0.5 P(tie). labels: 1 = anomaly."""
    y, s = _labels_scores(labels, scores)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC needs both classes")
    r = stats.rankdata(s)  # average ranks handle ties
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) for thresholds at every distinct score (flag = score > t)."""
    y, s = _labels_scores(labels, scores)
    ts = np.concatenate([[np.inf], np.unique(s)[::-1], [-np.inf]])
    tpr = np.array([(s[y] > t).mean() for t in ts])
    fpr = np.array([(s[~y] > t).mean() for t in ts])
    return fpr, tpr


def clopper_pearson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 0 or not 0 <= x <= n:
        raise ValueError(f"invalid binomial counts x={x}, n={n}")
    a = 1 - level
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def confusion(labels, flags) -> dict[str, int]:
    y = np.asarray(labels).astype(bool)
    f = np.asarray(flags).astype(bool)
    if y.shape != f.shape:
        raise ValueError("labels and flags differ in shape")
    return {"TP": int((y & f).sum()), "FN": int((y & ~f).sum()),
            "FP": int((~y & f).sum()), "TN": int((~y & ~f).sum())}


def rate_with_ci(x: int, n: int, level: float = 0.95) -> dict[str, Any]:
    lo, hi = clopper_pearson(x, n, level)
    return {"value": x / n, "x": x, "n": n, "ci_low": lo, "ci_high": hi,
            "ci_method": "Clopper-Pearson exact", "level": level}


def operating_point(labels, scores, threshold: float, level: float = 0.95) -> dict[str, Any]:
    y, s = _labels_scores(labels, scores)
    cm = confusion(y, s > threshold)
    out: dict[str, Any] = {"threshold": threshold, "confusion": cm}
    if y.any():
        out["recall"] = rate_with_ci(cm["TP"], cm["TP"] + cm["FN"], level)
    if (~y).any():
        out["fpr"] = rate_with_ci(cm["FP"], cm["FP"] + cm["TN"], level)
    return out


def bootstrap_auroc(labels, scores, n_boot: int, seed: int, level: float = 0.95) -> dict[str, Any]:
    y, s = _labels_scores(labels, scores)
    pos, neg = s[y], s[~y]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    lab = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    for b in range(n_boot):
        sp = pos[rng.integers(0, len(pos), len(pos))]
        sn = neg[rng.integers(0, len(neg), len(neg))]
        vals[b] = auroc(lab, np.r_[sp, sn])
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return {"value": auroc(y, s), "ci_low": float(lo), "ci_high": float(hi), "level": level,
            "ci_method": f"stratified percentile bootstrap, B={n_boot}, seed={seed}"}


def paired_bootstrap_delta(
    labels, flags_base, flags_pert, n_boot: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    """CI for Δrecall and ΔFPR (perturbed − base), resampling original images.

    Resampling is stratified by class so each replicate keeps the class sizes.
    """
    y = np.asarray(labels).astype(bool)
    fb = np.asarray(flags_base).astype(bool)
    fp = np.asarray(flags_pert).astype(bool)
    ia, inn = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for name, idx in (("recall", ia), ("fpr", inn)):
        if len(idx) == 0:
            continue
        d = fp[idx].astype(float) - fb[idx].astype(float)  # per-image paired change
        boots = np.array([d[rng.integers(0, len(idx), len(idx))].mean() for _ in range(n_boot)])
        lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
        out[f"delta_{name}"] = {
            "value": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_flip_to_flag": int(((~fb[idx]) & fp[idx]).sum()),
            "n_flip_to_clear": int((fb[idx] & ~fp[idx]).sum()),
            "n_images": int(len(idx)),
        }
    return out
