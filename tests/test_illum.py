"""PCB3 illumination experiment: correction, N1 perturbation, paired analysis, configs.

Synthetic data only; no PCB3 image is needed (docs/illumination_experiment_protocol.md §6 step 2).
"""
import copy

import numpy as np
import pytest
from PIL import Image

from ivad import photometric
from ivad.acceptance import acceptance_status
from ivad.config import REPO_ROOT, config_hash, load_config
from ivad.illum import classify_illum, did_bootstrap, subgroup_table, thirds
from ivad.robustness import apply_perturbation, illumination_ramp

CFG = REPO_ROOT / "config"
G1 = {"kind": "photometric", "brightness": 0.90, "contrast": 0.90}


def _luma(img):
    return np.asarray(img.convert("L"), np.float64)


# --------------------------------------------------------------------------- #
# Correction
# --------------------------------------------------------------------------- #
def test_standardise_hits_reference_stats(board):
    out = photometric.standardise(board(1), 120.0, 40.0)
    y = _luma(out)
    assert abs(y.mean() - 120.0) < 1.0 and abs(y.std() - 40.0) < 1.0


def test_standardise_undoes_global_darkening(board):
    """G1 is a global affine change with one gain: the correction inverts it up to rounding."""
    img = board(2)
    a = np.asarray(photometric.standardise(img, 100.0, 45.0), np.float64)
    b = np.asarray(photometric.standardise(apply_perturbation(img, G1), 100.0, 45.0), np.float64)
    assert np.abs(a - b).mean() < 1.5
    raw = np.abs(np.asarray(img, np.float64) - np.asarray(apply_perturbation(img, G1), np.float64)).mean()
    assert raw > 5 * np.abs(a - b).mean()


def test_standardise_does_not_undo_nonuniform_ramp(board):
    """A global correction cannot remove a left-to-right gradient (protocol §2, N1)."""
    img = board(3)
    a = _luma(photometric.standardise(img, 100.0, 45.0))
    b = _luma(photometric.standardise(illumination_ramp(img, 1.0, 0.8), 100.0, 45.0))
    w = a.shape[1]
    left, right = (b - a)[:, : w // 5].mean(), (b - a)[:, -w // 5:].mean()
    assert left - right > 8  # residual gradient remains


def test_standardise_uses_one_gain_for_all_channels():
    rng = np.random.default_rng(0)
    a = rng.integers(40, 200, (60, 80, 3)).astype(np.uint8)
    img = Image.fromarray(a)
    out = np.asarray(photometric.standardise(img, 128.0, 30.0), np.float64)
    mu, sd = photometric.luma_stats(img)
    gain = 30.0 / sd
    expect = np.clip(np.rint((a - mu) * gain + 128.0), 0, 255)
    assert np.array_equal(out, expect)


def test_local_defect_barely_moves_global_stats(board):
    clean, defect = board(4), board(4, anomaly=True)
    (m0, s0), (m1, s1) = photometric.luma_stats(clean), photometric.luma_stats(defect)
    assert abs(m1 - m0) / m0 < 0.01 and abs(s1 - s0) / s0 < 0.02


def test_standardise_rejects_flat_image_and_bad_reference():
    flat = Image.new("RGB", (20, 20), (90, 90, 90))
    with pytest.raises(ValueError):
        photometric.standardise(flat, 100.0, 40.0)
    with pytest.raises(ValueError):
        photometric.standardise(flat, 100.0, 0.0)


def test_fit_reference_is_median_of_per_image_stats(board):
    imgs = [board(s) for s in range(5)]
    ref = photometric.fit_reference(imgs)
    stats = np.array([photometric.luma_stats(im) for im in imgs])
    assert ref["ref_mean"] == pytest.approx(np.median(stats[:, 0]))
    assert ref["ref_std"] == pytest.approx(np.median(stats[:, 1]))
    assert ref["n_images"] == 5


def test_from_config_requires_frozen_reference():
    cfg = {"preprocess": {"photometric": {"method": "global_luma_affine", "ref_mean": None, "ref_std": None}}}
    with pytest.raises(ValueError):
        photometric.from_config(cfg)
    cfg["preprocess"]["photometric"].update(ref_mean=100.0, ref_std=40.0)
    assert callable(photometric.from_config(cfg))
    assert photometric.from_config({"preprocess": {}}) is None
    with pytest.raises(ValueError):
        photometric.from_config({"preprocess": {"photometric": {"method": "grey_world"}}})


def test_preprocessor_applies_correction_before_registration(board):
    from ivad.pipeline import Preprocessor

    cfg = {"preprocess": {"register": False, "crop_wh": None, "size_hw": [64, 96],
                          "photometric": {"method": "global_luma_affine", "ref_mean": 110.0, "ref_std": 35.0}}}
    img = board(5)
    pre = Preprocessor(cfg, None)
    assert np.array_equal(np.asarray(pre.view(img)), np.asarray(photometric.standardise(img, 110.0, 35.0)))
    cfg0 = copy.deepcopy(cfg)
    del cfg0["preprocess"]["photometric"]
    assert np.array_equal(np.asarray(Preprocessor(cfg0, None).view(img)), np.asarray(img))


# --------------------------------------------------------------------------- #
# N1 perturbation
# --------------------------------------------------------------------------- #
def test_illumination_ramp_gains():
    a = np.full((10, 101, 3), 200, np.uint8)
    out = np.asarray(illumination_ramp(Image.fromarray(a), 1.0, 0.8), np.float64)
    assert np.all(out[:, 0] == 200) and np.all(out[:, -1] == 160) and np.all(out[:, 50] == 180)
    assert np.array_equal(out, np.asarray(apply_perturbation(
        Image.fromarray(a), {"kind": "illum_ramp", "gain_start": 1.0, "gain_end": 0.8}), np.float64))


# --------------------------------------------------------------------------- #
# Paired analysis
# --------------------------------------------------------------------------- #
def test_did_bootstrap_values():
    n = 200
    b0 = np.zeros(n, bool); b1 = np.zeros(n, bool); b1[:50] = True     # baseline +25 pts
    c0 = np.zeros(n, bool); c1 = np.zeros(n, bool); c1[:10] = True     # candidate +5 pts
    r = did_bootstrap(b0, b1, c0, c1, 2000, 1)
    assert r["delta_base"] == pytest.approx(0.25) and r["delta_cand"] == pytest.approx(0.05)
    assert r["did"] == pytest.approx(-0.20)
    assert r["ci_high"] < 0 and r["ci_low"] < r["did"] < r["ci_high"]
    same = did_bootstrap(b0, b1, b0, b1, 500, 1)
    assert same["did"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0


def _rate(base, new, p=1.0):
    return {"base": base, "new": new, "delta": new - base, "p_value": p}


def test_classify_illum_outcomes():
    good_did = {"ci_low": -0.25, "ci_high": -0.10, "delta_base": 0.25}
    flat_did = {"ci_low": -0.05, "ci_high": 0.03, "delta_base": 0.25}
    bad_did = {"ci_low": 0.02, "ci_high": 0.10, "delta_base": 0.25}
    ok = (_rate(0.80, 0.78), _rate(0.05, 0.06), _rate(0.04, 0.05))
    assert classify_illum(good_did, *ok)["outcome"] == "ROBUSTNESS IMPROVEMENT SUPPORTED"
    assert classify_illum(flat_did, *ok)["outcome"] == "INCONCLUSIVE"
    assert classify_illum(bad_did, *ok)["outcome"] == "UNFAVOURABLE TRADE-OFF"
    # exactly -5 pts is acceptable; below it is not
    assert classify_illum(good_did, _rate(0.80, 0.75), *ok[1:])["clean_recall_gate_pass"]
    assert classify_illum(good_did, _rate(0.80, 0.74), *ok[1:])["outcome"] == "UNFAVOURABLE TRADE-OFF"
    # a significant loss fails even inside the margin
    assert classify_illum(good_did, _rate(0.80, 0.77, p=0.03), *ok[1:])["outcome"] == "UNFAVOURABLE TRADE-OFF"
    # clean FPR non-inferiority, official test and holdout separately
    assert classify_illum(good_did, ok[0], _rate(0.05, 0.11), ok[2])["outcome"] == "UNFAVOURABLE TRADE-OFF"
    assert classify_illum(good_did, ok[0], ok[1], _rate(0.04, 0.10))["outcome"] == "UNFAVOURABLE TRADE-OFF"
    weak = classify_illum({"ci_low": -0.04, "ci_high": -0.01, "delta_base": 0.03}, *ok)
    assert weak["baseline_rise_weakly_informative"]


def test_thirds_and_subgroups():
    v = np.arange(100, dtype=float)[::-1]
    lab = thirds(v, [f"img{i:03d}" for i in range(100)])
    assert sorted(np.bincount(lab).tolist()) == [33, 33, 34]
    assert lab[np.argmin(v)] == 0 and lab[np.argmax(v)] == 2
    fb = np.ones(100, bool); fc = fb.copy(); fc[lab == 0] = False
    t = subgroup_table(lab, v, fb, fc, ("small", "medium", "large"))
    assert t["small"]["detected_base"] - t["small"]["detected_cand"] == t["small"]["n"]
    assert t["large"]["base_only"] == 0


def test_acceptance_ignores_non_p5_conditions():
    m = {"official_test": {"auroc": {"value": 0.95}, "operating_points": {"P95": {
        "recall": {"value": 0.92}, "fpr": {"value": 0.05}}}},
         "supplementary_holdout": {"operating_points": {"P95": {"fpr": {"value": 0.05}}}},
         "latency": {"median_ms": 200}}
    acc = {"performance": {"auroc_min": .9, "recall_min": .9, "fpr_max": .1, "latency_ms_max": 1000,
                           "robustness_recall_drop_max": .1, "robustness_fpr_rise_max": .1},
           "no_go": {"recall_floor": .8, "fpr_ceiling": .2, "robustness_recall_collapse": .2}}

    def pert(rise, **spec):
        return {"spec": spec, "paired": {"delta_recall": {"value": 0.0}, "delta_fpr": {"value": rise}}}
    rb = {"perturbations": {"g1": pert(0.02), "n1": pert(0.30, acceptance=False)}}
    st = acceptance_status(m, rb, "P95", acc)
    assert st["robustness"] is True and st["robustness_max_fpr_rise"] == pytest.approx(0.02)
    rb["perturbations"]["n1"]["spec"]["acceptance"] = True
    assert acceptance_status(m, rb, "P95", acc)["robustness"] is False


# --------------------------------------------------------------------------- #
# Configs: one change only
# --------------------------------------------------------------------------- #
def test_pcb3_configs_keep_pcb2_v2_settings():
    v2, base, cand = (load_config(CFG / f) for f in ("pcb2_v2.yaml", "pcb3_baseline.yaml", "pcb3_illum.yaml"))
    for c in (base, cand):
        for k in ("seed", "model", "calibration", "evaluation", "cost", "runtime", "scoring"):
            assert c[k] == v2[k], k
        extra = {"photometric", "board_mask"}  # the correction (candidate) and Addendum B1 (both)
        assert {k: v for k, v in c["preprocess"].items() if k not in extra} == v2["preprocess"]
        assert c["preprocess"]["board_mask"] == {"exclude_hue": [84, 106]}
        p5 = [{k: v for k, v in p.items() if k != "protocol_id"} for p in c["robustness"]["perturbations"]
              if p.get("acceptance", True)]
        assert p5 == v2["robustness"]["perturbations"]  # original P5 definitions unchanged
        assert (c["dataset"]["object"], c["dataset"]["n_holdout"], c["dataset"]["n_test_normal"]) == ("pcb3", 155, 101)
    assert "photometric" not in base["preprocess"]
    assert cand["preprocess"]["photometric"]["method"] == "global_luma_affine"


def test_pcb3_variants_differ_only_in_the_correction():
    base, cand = load_config(CFG / "pcb3_baseline.yaml"), load_config(CFG / "pcb3_illum.yaml")
    strip = []
    for c in (base, cand):
        c = copy.deepcopy(c)
        for k in ("_config_path", "run_name"):
            c.pop(k)
        c["preprocess"].pop("photometric", None)
        for k in ("artifacts", "reports", "figures"):
            c["paths"].pop(k)
        strip.append(c)
    assert strip[0] == strip[1]
    assert config_hash(base) != config_hash(cand)
    ids = {p.get("protocol_id") for p in cand["robustness"]["perturbations"]}
    assert {"G1", "N1", "B1"} <= ids


def test_exclude_hue_removes_saturated_background(board):
    """Addendum B1: on a saturated green background the original rule marks the whole frame."""
    from ivad.align import foreground_mask

    img = np.asarray(board(6), np.float64)
    grey = (np.abs(img - 95).max(axis=2) < 15)  # the synthetic grey cloth
    img[grey] = (35, 125, 60)  # saturated green cloth
    green = Image.fromarray(img.astype(np.uint8))
    hue = np.asarray(Image.new("RGB", (1, 1), (35, 125, 60)).convert("HSV"))[0, 0, 0]
    assert foreground_mask(green).mean() > 0.7  # cloth counted as board
    m = foreground_mask(green, exclude_hue=[hue - 11, hue + 11])
    assert np.array_equal(m, foreground_mask(board(6)))  # same board mask as on the grey cloth
    assert np.array_equal(foreground_mask(board(6)), foreground_mask(board(6), exclude_hue=None))


def test_mask_kwargs_from_config():
    from ivad.pipeline import mask_kwargs

    assert mask_kwargs({"preprocess": {}}) == {}
    assert mask_kwargs({"preprocess": {"board_mask": {"exclude_hue": [84, 106]}}}) == {"exclude_hue": [84, 106]}
    with pytest.raises(ValueError):
        mask_kwargs({"preprocess": {"board_mask": {"hue": [1, 2]}}})
