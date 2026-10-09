"""Threshold calibration, metric calculations and cost model."""
import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from ivad import calibration, cost, metrics


# ----------------------------------------------------------------- calibration
@pytest.mark.parametrize("alpha,k", [(0.10, 136), (0.05, 144), (0.01, 150)])
def test_conformal_rank_for_n150(alpha, k):
    assert calibration.conformal_rank(150, alpha) == k


def test_conformal_rank_rejects_too_small_alpha():
    with pytest.raises(ValueError):
        calibration.conformal_rank(50, 0.01)  # needs n >= 100
    with pytest.raises(ValueError):
        calibration.conformal_rank(10, 1.5)


def test_threshold_is_kth_order_statistic_and_flags_strictly_above():
    scores = np.arange(1, 151, dtype=float)[::-1]  # unsorted input
    t, k = calibration.conformal_threshold(scores, 0.05)
    assert t == 144.0 and k == 144
    assert calibration.flag(scores, t).sum() == 150 - 144  # 145..150
    assert not calibration.flag(np.array([t]), t)[0]


def test_threshold_rejects_nonfinite():
    with pytest.raises(ValueError):
        calibration.conformal_threshold([1.0, np.nan, 2.0], 0.5)


def test_fpr_distribution_matches_simulation():
    """Over random calibration sets, the realised FPR follows Beta(n+1-k, k)."""
    n, alpha = 150, 0.05
    rng = np.random.default_rng(0)
    fprs = []
    for _ in range(4000):
        t, _ = calibration.conformal_threshold(rng.uniform(size=n), alpha)
        fprs.append(1 - t)  # true FPR of threshold t for Uniform(0,1) normals
    fprs = np.array(fprs)
    d = calibration.fpr_distribution(n, calibration.conformal_rank(n, alpha))
    assert fprs.mean() == pytest.approx(d["expected"], abs=0.002)
    assert d["expected"] <= alpha
    inside = ((fprs >= d["lower"]) & (fprs <= d["upper"])).mean()
    assert inside == pytest.approx(0.95, abs=0.015)


def test_calibrate_output_fields():
    out = calibration.calibrate(np.linspace(0, 1, 150), {"P95": 0.05})
    p = out["P95"]
    assert p["rank_k"] == 144 and p["n_calib"] == 150 and p["target_fpr"] == 0.05
    assert p["calib_fpr_observed"] == pytest.approx(6 / 150)


# --------------------------------------------------------------------- metrics
def test_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
    assert metrics.auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auroc_edge_cases():
    assert metrics.auroc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert metrics.auroc([0, 1], [1.0, 1.0]) == 0.5
    with pytest.raises(ValueError):
        metrics.auroc([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        metrics.auroc([0, 1], [0.2, np.inf])


@pytest.mark.parametrize("x,n", [(0, 100), (3, 100), (90, 100), (100, 100), (7, 154)])
def test_clopper_pearson_matches_scipy(x, n):
    ref = stats.binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact")
    lo, hi = metrics.clopper_pearson(x, n)
    assert lo == pytest.approx(ref.low, abs=1e-10) and hi == pytest.approx(ref.high, abs=1e-10)


def test_confusion_and_operating_point():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    s = np.array([5, 4, 1, 3, 0.5, 0.2, 0.1])
    op = metrics.operating_point(y, s, threshold=2.0)
    assert op["confusion"] == {"TP": 2, "FN": 1, "FP": 1, "TN": 3}
    assert op["recall"]["value"] == pytest.approx(2 / 3)
    assert op["fpr"]["value"] == pytest.approx(1 / 4)
    assert op["recall"]["ci_method"] == "Clopper-Pearson exact"


def test_bootstrap_auroc_is_seeded_and_brackets_point():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(100), np.zeros(100)]
    s = rng.normal(size=200) + y
    a = metrics.bootstrap_auroc(y, s, 500, seed=3)
    b = metrics.bootstrap_auroc(y, s, 500, seed=3)
    assert a == b
    assert a["ci_low"] < a["value"] < a["ci_high"]


def test_roc_curve_endpoints():
    fpr, tpr = metrics.roc_curve([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.3])
    assert (fpr[0], tpr[0]) == (0, 0) and (fpr[-1], tpr[-1]) == (1, 1)


def test_paired_bootstrap_delta():
    y = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    base = np.r_[np.ones(45, bool), np.zeros(5, bool), np.zeros(45, bool), np.ones(5, bool)]
    same = metrics.paired_bootstrap_delta(y, base, base, 200, seed=0)
    assert same["delta_recall"]["value"] == 0 and same["delta_recall"]["ci_high"] == 0
    pert = base.copy()
    pert[:5] = False  # 5 anomalies newly missed
    d = metrics.paired_bootstrap_delta(y, base, pert, 500, seed=0)["delta_recall"]
    assert d["value"] == pytest.approx(-0.1) and d["n_flip_to_clear"] == 5 and d["ci_high"] < 0


# ------------------------------------------------------------------------ cost
def test_cost_reference_policies_and_breakeven():
    assert cost.normalized_cost(3.0, recall=0.0, fpr=0.0) == 3.0  # review none = k
    assert cost.normalized_cost(3.0, recall=1.0, fpr=1.0) == 1.0  # review all = 1
    b = cost.breakeven_k(recall=0.9, fpr=0.05)
    assert b["k_low"] == pytest.approx(0.05 / 0.9) and b["k_high"] == pytest.approx(9.5)
    k = np.sqrt(b["k_low"] * b["k_high"])
    assert cost.normalized_cost(k, 0.9, 0.05) < min(k, 1.0)


def test_k_value():
    assert cost.k_value(0.01, 100, 1) == pytest.approx(0.01 / 0.99 * 100)
    with pytest.raises(ValueError):
        cost.k_value(0, 1, 1)


# ------------------------------------------------------------- report checks
def test_pdf_traceability_flags_untraced_numbers():
    from ivad.consistency import pdf_tokens, traceable_tokens

    allowed = traceable_tokens([{"auroc": 0.9378, "recall": {"x": 72, "n": 100, "value": 0.72}}])
    toks = pdf_tokens("AUROC 0.938, recall 72% (72/100) on PCB1 at P95, but 77% elsewhere")
    assert "0.938" in allowed and "72" in allowed and "100" in allowed
    assert [t for t in toks if t not in allowed] == ["77"]  # P95 / PCB1 are not numeric tokens
