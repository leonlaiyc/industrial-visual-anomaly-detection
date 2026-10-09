"""Paired v1-vs-v2 statistics, the pre-registered outcome rule and acceptance status."""
import numpy as np
import pytest
from scipy import stats

from ivad.acceptance import acceptance_status, load_criteria
from ivad.compare import classify, mcnemar_exact, paired_auroc_delta, paired_rate_delta


def test_mcnemar_exact_matches_binomial():
    base = np.array([1] * 60 + [0] * 40, bool)
    new = base.copy()
    new[60:72] = True  # 12 newly flagged
    new[:2] = False  # 2 lost
    r = mcnemar_exact(base, new)
    assert (r["flagged_by_new_only"], r["flagged_by_base_only"], r["n_discordant"]) == (12, 2, 14)
    assert r["p_value"] == pytest.approx(stats.binomtest(2, 14, 0.5).pvalue)
    assert r["p_value"] < 0.05
    same = mcnemar_exact(base, base)
    assert same["p_value"] == 1.0 and same["n_discordant"] == 0
    with pytest.raises(ValueError):
        mcnemar_exact(base, new[:-1])


def test_paired_rate_delta_uses_pairing():
    rng = np.random.default_rng(0)
    base = rng.random(100) < 0.7
    new = base.copy()
    new[np.flatnonzero(~base)[:10]] = True  # +10 points, no losses
    r = paired_rate_delta(base, new, 2000, 1)
    assert r["delta"] == pytest.approx(0.10)
    assert r["ci_low"] <= 0.10 <= r["ci_high"]
    assert r["ci_low"] > 0  # paired: every discordant pair favours new
    # unpaired intuition would give a CI spanning 0 for 70% vs 80% at n=100; pairing is what makes it tight
    assert r["ci_high"] - r["ci_low"] < 0.15
    a = paired_rate_delta(base, new, 500, 3)
    b = paired_rate_delta(base, new, 500, 3)
    assert a == b  # deterministic given the seed


def test_paired_auroc_delta():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    s = rng.normal(size=100) + y
    same = paired_auroc_delta(y, s, s.copy(), 300, 0)
    assert same["delta"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = s + 3 * y
    r = paired_auroc_delta(y, s, better, 300, 0)
    assert r["delta"] > 0 and r["ci_low"] > 0 and r["new"] == pytest.approx(1.0, abs=0.01)


def _d(base, new, p=1.0):
    return {"base": base, "new": new, "delta": new - base, "p_value": p}


def test_classify_outcomes():
    ok = _d(0.05, 0.06)
    assert classify(_d(0.70, 0.85, 0.001), ok, ok, 0.05, 0.05)["outcome"] == "IMPROVEMENT SUPPORTED"
    assert classify(_d(0.70, 0.74, 0.20), ok, ok, 0.05, 0.05)["outcome"] == "NO MEANINGFUL EFFECT DEMONSTRATED"
    assert classify(_d(0.70, 0.66, 0.30), ok, ok, 0.05, 0.05)["outcome"] == "NO MEANINGFUL EFFECT DEMONSTRATED"
    assert classify(_d(0.70, 0.55, 0.001), ok, ok, 0.05, 0.05)["outcome"] == "UNFAVOURABLE TRADE-OFF"
    worse_fpr = _d(0.05, 0.11)  # +6 points > 5-point margin
    r = classify(_d(0.70, 0.90, 0.001), ok, worse_fpr, 0.05, 0.05)
    assert r["outcome"] == "UNFAVOURABLE TRADE-OFF" and r["fpr_noninferior"] == {"official_test": True, "holdout": False}
    edge = _d(0.05, 0.10)  # exactly at the margin is non-inferior
    assert classify(_d(0.70, 0.85, 0.001), edge, edge, 0.05, 0.05)["outcome"] == "IMPROVEMENT SUPPORTED"


def _metrics(auroc, recall, fpr, hold, lat):
    return {"official_test": {"auroc": {"value": auroc},
                              "operating_points": {"P95": {"recall": {"value": recall}, "fpr": {"value": fpr}}}},
            "supplementary_holdout": {"operating_points": {"P95": {"fpr": {"value": hold}}}},
            "latency": {"median_ms": lat}}


def _rb(drop, rise):
    return {"perturbations": {"x": {"paired": {"delta_recall": {"value": -drop}, "delta_fpr": {"value": rise}}}}}


def test_acceptance_status_reproduces_v1_rules():
    acc = load_criteria()
    v1 = acceptance_status(_metrics(0.938, 0.72, 0.07, 6 / 154, 139), _rb(0.03, 0.043), "P95", acc)
    assert v1["verdict"] == "NO-GO" and v1["no_go_triggers"] == ["recall_below_floor"]
    assert v1["auroc"] and not v1["recall"] and v1["fpr_test"] and v1["fpr_holdout"] and v1["robustness"]
    go = acceptance_status(_metrics(0.97, 0.92, 0.06, 0.05, 150), _rb(0.02, 0.03), "P95", acc)
    assert go["verdict"] == "GO"
    mid = acceptance_status(_metrics(0.95, 0.85, 0.06, 0.05, 150), _rb(0.02, 0.03), "P95", acc)
    assert mid["verdict"] == "INCONCLUSIVE"
    no_rb = acceptance_status(_metrics(0.97, 0.92, 0.06, 0.05, 150), None, "P95", acc)
    assert no_rb["robustness"] is None and no_rb["verdict"] == "INCONCLUSIVE"
    collapse = acceptance_status(_metrics(0.97, 0.92, 0.06, 0.05, 150), _rb(0.25, 0.0), "P95", acc)
    assert collapse["verdict"] == "NO-GO" and "robustness_collapse" in collapse["no_go_triggers"]
