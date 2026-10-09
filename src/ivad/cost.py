"""Normalised expected decision cost.

Per inspected board, with defect prevalence pi, cost of a missed defect C_FN
and cost of an unnecessary review C_FP (costs of correct decisions taken as 0):

    E[cost] = pi * (1 - Recall) * C_FN + (1 - pi) * FPR * C_FP

Dividing by (1 - pi) * C_FP gives the dimensionless

    C(k, t) = k * (1 - Recall(t)) + FPR(t),   k = (pi / (1 - pi)) * (C_FN / C_FP)

Reference policies: review none (Recall=0, FPR=0) -> C = k;
review all (Recall=1, FPR=1) -> C = 1.

k must come from the USER's prevalence and costs. The benchmark's 50/50 test
set says nothing about real prevalence, and Recall/FPR here are benchmark
estimates with uncertainty.
"""
from __future__ import annotations

import numpy as np


def k_value(prevalence: float, cost_fn: float, cost_fp: float) -> float:
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError("costs must be positive")
    return prevalence / (1 - prevalence) * cost_fn / cost_fp


def normalized_cost(k, recall: float, fpr: float):
    if not (0 <= recall <= 1 and 0 <= fpr <= 1):
        raise ValueError("recall and fpr must be in [0, 1]")
    return np.asarray(k, dtype=float) * (1 - recall) + fpr


def breakeven_k(recall: float, fpr: float) -> dict[str, float]:
    """k range where the threshold beats both reference policies.

    beats review-none:  k(1-R) + F < k      <=>  k > F / R
    beats review-all:   k(1-R) + F < 1      <=>  k < (1 - F) / (1 - R)
    """
    lo = fpr / recall if recall > 0 else float("inf")
    hi = (1 - fpr) / (1 - recall) if recall < 1 else float("inf")
    return {"k_low": lo, "k_high": hi, "useful": lo < hi}
