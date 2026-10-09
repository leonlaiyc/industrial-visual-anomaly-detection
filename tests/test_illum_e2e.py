"""End-to-end test of the PCB3 paired illumination analysis (`ivad illum-compare`) on synthetic artifacts.

Builds a complete synthetic world in a temporary directory: two configs derived from the real PCB3
configs, a split manifest, tiny images with ground-truth masks, and every score artifact the analysis
reads (frozen operating points, metrics, clean scores per role, robustness scores per condition).
No PCB3 image, score or test output is used. Then runs the real `illum_compare` and checks the
outcome classification, gates, N1 reporting, subgroup analysis and the written reports, plus the
refusals on malformed inputs.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from ivad import illum
from ivad.config import REPO_ROOT, config_hash, load_config, sha256_text
from ivad.pipeline import write_scores

N_FIT, N_CALIB, N_HOLD, N_TN, N_TA = 3, 10, 25, 20, 30
OBJ = "pcb3"
THR = 1.0
HI, LO = 1.5, 0.5


def _img(rel, root, colour=100):
    p = root / "data" / "visa" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((48, 64, 3), colour, np.uint8)
    a[:, 32:] += 40  # some luminance variation, so the candidate correction is defined
    Image.fromarray(a).save(p)


def _anomaly(i, root):
    """Anomaly i: square defect, side growing with i (size terciles), contrast cycling (contrast terciles)."""
    rel = f"{OBJ}/Data/Images/Anomaly/{i:03d}.png"
    mrel = f"{OBJ}/Data/Masks/Anomaly/{i:03d}.png"
    side = 2 + i // 2
    delta = 8 + (i * 7) % 30 * 4
    img = Image.new("RGB", (64, 48), (100, 100, 100))
    mask = Image.new("L", (64, 48), 0)
    x0, y0 = 20, 10
    ImageDraw.Draw(img).rectangle([x0, y0, x0 + side - 1, y0 + side - 1], fill=(100 + delta,) * 3)
    ImageDraw.Draw(mask).rectangle([x0, y0, x0 + side - 1, y0 + side - 1], fill=255)
    for r, im in ((rel, img), (mrel, mask)):
        p = root / "data" / "visa" / r
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p)
    return rel, mrel


def _rows(root):
    """Split manifest rows (test anomalies, test normals, then normal train roles), and image_anno rows."""
    rows, anno = [], []
    for i in range(N_TA):
        rel, mrel = _anomaly(i, root)
        rows.append({"image": rel, "official_split": "test", "label": "anomaly", "role": "test", "sha256": f"a{i}"})
        # VisA image_anno.csv: `label` is the defect TYPE for anomalies, "normal" otherwise
        anno.append({"image": rel, "label": ("bent", "melt", "scratch", "missing")[i % 4], "mask": mrel})
    k = 0
    for role, n, split in (("test", N_TN, "test"), ("fit", N_FIT, "train"), ("calib", N_CALIB, "train"),
                           ("holdout", N_HOLD, "train")):
        for _ in range(n):
            rel = f"{OBJ}/Data/Images/Normal/{k:04d}.png"
            _img(rel, root, colour=90 + k % 20)
            rows.append({"image": rel, "official_split": split, "label": "normal", "role": role, "sha256": f"n{k}"})
            anno.append({"image": rel, "label": "normal", "mask": ""})
            k += 1
    return rows, anno


def _write_csv(path, rows, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def default_scores():
    """Per-variant score design (both thresholds 1.0). Returns {variant: {condition: (anomaly, test-normal,
    holdout-normal) arrays}} for the 'supported' scenario."""
    an = np.full(N_TA, 2.0)
    an[-3:] = LO  # three misses (largest defects) for both variants
    tn, hn = np.full(N_TN, LO), np.full(N_HOLD, LO)
    tn[0] = hn[0] = HI  # one clean false alarm per normal set, both variants
    clean = (an, tn, hn)
    s = {v: {c: tuple(a.copy() for a in clean) for c in ("P0", "G1", "B1", "N1", "blur", "shift")}
         for v in ("base", "cand")}
    # G1: baseline flags 18 more of the 45 unseen normals, candidate none -> DiD ~ -0.38
    s["base"]["G1"][1][1:9] = HI
    s["base"]["G1"][2][1:11] = HI
    # N1: both variants flag 14 more normals -> candidate rise 31 pts > 10 -> limitation exposed
    for v in ("base", "cand"):
        s[v]["N1"][1][10:16] = HI
        s[v]["N1"][2][12:20] = HI
    return s


def build_world(root: Path, scores=None, mutate_cfg=None):
    """Write configs, manifest, images, masks and all score artifacts. Returns (cfg_base, cfg_cand)."""
    scores = scores or default_scores()
    rows, anno = _rows(root)
    _write_csv(root / "manifests" / "split_pcb3.csv", rows, ["image", "official_split", "label", "role", "sha256"])
    _write_csv(root / "data" / "visa" / OBJ / "image_anno.csv", anno, ["image", "label", "mask"])
    cfgs = {}
    for v, name in (("base", "pcb3_baseline"), ("cand", "pcb3_illum")):
        raw = yaml.safe_load((REPO_ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
        raw["paths"]["root"] = str(root)
        if mutate_cfg:
            mutate_cfg(v, raw)
        p = root / "config" / f"{name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        cfgs[v] = load_config(p)
    manifest = root / "manifests" / "split_pcb3.csv"
    names = {"P0": None}
    for v, cfg in cfgs.items():
        rep = root / cfg["paths"]["reports"]
        rep.mkdir(parents=True, exist_ok=True)
        perts = cfg["robustness"]["perturbations"]
        names = {p.get("protocol_id", p["kind"]): p["name"] for p in perts}
        fz = {"primary": "P95", "thresholds": {"P95": {"threshold": THR}}, "frozen_at_utc": "2026-01-01T00:00:00+00:00",
              "config_hash": config_hash(cfg), "split_manifest_sha256": sha256_text(manifest),
              "pretrained_weights": {"name": "synthetic"}}
        (rep / "frozen_operating_point.json").write_text(json.dumps(fz), encoding="utf-8")
        an, tn, hn = scores[v]["P0"]
        by_role = {"calib": np.full(N_CALIB, 0.4)}
        test_rows = [r for r in rows if r["role"] == "test"]
        by_role["test"] = np.r_[an, tn]
        by_role["holdout"] = hn
        for role in ("calib", "test", "holdout"):
            rr = [r for r in rows if r["role"] == role]
            s = by_role[role]
            write_scores(rep / f"scores_{role}.csv", rr, s,
                         {"score_full": s + 0.1, "roi_coverage": np.ones(len(s)), "roi_fallback": np.zeros(len(s), bool)})
        y = np.r_[np.ones(N_TA, bool), np.zeros(N_TN, bool)]
        st = by_role["test"]
        m = {"frozen_operating_point": {"primary": "P95", "config_hash": fz["config_hash"]},
             "official_test": {"auroc": {"value": 0.95}, "operating_points": {"P95": {
                 "recall": {"value": float((st[y] > THR).mean())}, "fpr": {"value": float((st[~y] > THR).mean())}}}},
             "supplementary_holdout": {"operating_points": {"P95": {"fpr": {"value": float((hn > THR).mean())}}}},
             "latency": {"median_ms": 100.0, "p95_ms": 120.0}, "git_head_at_evaluation": "synthetic",
             "roi_fallback": {"test": {"n": len(st), "n_fallback": 0}, "holdout": {"n": len(hn), "n_fallback": 0}}}
        (rep / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
        hold_rows = [r for r in rows if r["role"] == "holdout"]
        rob_rows = test_rows + hold_rows
        base = np.r_[an, tn, hn]
        ry = np.array([r["label"] == "anomaly" for r in rob_rows])
        per = {"image": [r["image"] for r in rob_rows], "label": [r["label"] for r in rob_rows], "base": base}
        rb = {"threshold": THR, "primary": "P95", "perturbations": {}}
        for p in perts:
            key = p.get("protocol_id") or ("blur" if p["kind"] == "blur" else "shift")
            a2, t2, h2 = scores[v][key]
            s2 = np.r_[a2, t2, h2]
            per[p["name"]] = s2
            rb["perturbations"][p["name"]] = {"spec": p, "paired": {
                "delta_recall": {"value": float((s2[ry] > THR).mean() - (base[ry] > THR).mean())},
                "delta_fpr": {"value": float((s2[~ry] > THR).mean() - (base[~ry] > THR).mean())}}}
        (rep / "robustness.json").write_text(json.dumps(rb), encoding="utf-8")
        keys = list(per)
        with open(rep / "robustness_scores.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(keys)
            for i in range(len(rob_rows)):
                w.writerow([per[k][i] if isinstance(per[k][i], str) else f"{per[k][i]:.6f}" for k in keys])
    return cfgs["base"], cfgs["cand"]


@pytest.fixture
def world(tmp_path):
    return tmp_path, build_world(tmp_path)


# --------------------------------------------------------------------------- #
# Positive end-to-end runs
# --------------------------------------------------------------------------- #
def test_e2e_supported(world, monkeypatch):
    root, (cb, cc) = world
    seen = []
    real = illum.defect_descriptors

    def spy(cfg, images):
        seen.extend(images)
        return real(cfg, images)
    monkeypatch.setattr(illum, "defect_descriptors", spy)
    out = illum.illum_compare(cb, cc)

    # GT masks are read only for the clean TEST anomalies (post-freeze description)
    manifest = {r["image"]: r for r in csv.DictReader(open(root / "manifests" / "split_pcb3.csv", encoding="utf-8"))}
    assert len(seen) == N_TA and all(manifest[i]["role"] == "test" and manifest[i]["label"] == "anomaly" for i in seen)

    assert all(out["pairing_checks"].values())
    p = out["primary_endpoint"]
    assert p["condition"] == "G1"
    assert p["delta_base"] == pytest.approx(18 / 45) and p["delta_cand"] == 0
    assert p["did"] == pytest.approx(-18 / 45) and p["ci_high"] < 0
    d = out["decision"]
    assert d["outcome"] == "ROBUSTNESS IMPROVEMENT SUPPORTED"
    assert d["clean_recall_gate_pass"] and all(d["clean_fpr_gate_pass"].values())
    assert not d["baseline_rise_weakly_informative"]
    c = out["clean"]
    assert c["recall_test_anomalies"]["base"] == c["recall_test_anomalies"]["new"] == pytest.approx(27 / 30)
    assert c["fpr_test_normals"]["new"] == pytest.approx(1 / N_TN)
    assert c["fpr_holdout_normals"]["new"] == pytest.approx(1 / N_HOLD)
    # N1 reported, limitation exposed (candidate rise 14/45 > 10 pts), DiD 0
    nu = out["nonuniform_limitation"]
    assert nu["limitation_exposed"] and nu["cand_fpr_rise"] == pytest.approx(14 / 45)
    assert out["conditions"]["N1"]["fpr_normals"]["did"] == 0
    assert out["conditions"]["B1"]["fpr_normals"]["did"] == 0
    # subgroups: 100% agreement, the misses are the three largest defects
    sg = out["subgroups"]
    assert sg["size_clean"]["large"]["detected_base"] == sg["size_clean"]["large"]["n"] - 3
    assert sg["size_clean"]["small"]["recall_cand"] == 1.0
    assert not sg["sensitivity_flag"]["flagged"]
    # acceptance excludes N1 from P5
    assert out["acceptance"]["cand"]["robustness_max_fpr_rise"] == pytest.approx(0.0)
    # reports written
    dst = root / cc["paths"]["illum_comparison"]
    js = json.loads((dst / "comparison.json").read_text(encoding="utf-8"))
    md = (dst / "comparison.md").read_text(encoding="utf-8")
    assert js["decision"]["outcome"] == "ROBUSTNESS IMPROVEMENT SUPPORTED"
    assert "**Pre-registered outcome: ROBUSTNESS IMPROVEMENT SUPPORTED**" in md
    assert "limitation exposed: **True**" in md and "G1 (primary)" in md


def test_e2e_unfavourable_with_small_defect_flag(tmp_path):
    s = default_scores()
    for cond in s["cand"]:  # candidate misses the 6 smallest defects in every condition
        s["cand"][cond][0][:6] = LO
    cb, cc = build_world(tmp_path, s)
    out = illum.illum_compare(cb, cc)
    d = out["decision"]
    assert out["clean"]["recall_test_anomalies"]["delta"] == pytest.approx(-0.2)
    assert not d["clean_recall_gate_pass"] and d["outcome"] == "UNFAVOURABLE TRADE-OFF"
    sf = out["subgroups"]["sensitivity_flag"]
    assert sf["flagged"] and sf["deficits"]["small"] == 6


def test_e2e_weak_baseline_and_inconclusive(tmp_path):
    s = default_scores()
    s["base"]["G1"] = tuple(a.copy() for a in s["base"]["P0"])
    s["base"]["G1"][2][1] = HI  # baseline rise 1/45 < 5 pts
    cb, cc = build_world(tmp_path, s)
    out = illum.illum_compare(cb, cc)
    assert out["decision"]["baseline_rise_weakly_informative"]
    assert out["decision"]["outcome"] == "INCONCLUSIVE"
    md = (tmp_path / cc["paths"]["illum_comparison"] / "comparison.md").read_text(encoding="utf-8")
    assert "weakly informative" in md


def test_e2e_fpr_gate_on_holdout(tmp_path):
    s = default_scores()
    for cond in s["cand"]:
        s["cand"][cond][2][1:4] = HI  # candidate +3/25 = +12 pts on holdout normals only
    cb, cc = build_world(tmp_path, s)
    d = illum.illum_compare(cb, cc)["decision"]
    assert d["clean_fpr_gate_pass"] == {"official_test": True, "holdout": False}
    assert d["outcome"] == "UNFAVOURABLE TRADE-OFF"


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def _rewrite(path, fn):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = list(rows[0])
    rows = fn(rows)
    _write_csv(path, rows, keys)


def _rep(root, cfg):
    return root / cfg["paths"]["reports"]


def test_refuses_mismatched_image_ordering(world):
    root, (cb, cc) = world
    _rewrite(_rep(root, cc) / "scores_test.csv", lambda r: [r[1], r[0]] + r[2:])
    with pytest.raises(RuntimeError, match="split manifest's test rows"):
        illum.illum_compare(cb, cc)


def test_refuses_reordered_robustness_rows(world):
    root, (cb, cc) = world
    _rewrite(_rep(root, cb) / "robustness_scores.csv", lambda r: r[::-1])
    with pytest.raises(RuntimeError, match="robustness rows"):
        illum.illum_compare(cb, cc)


def test_refuses_missing_condition_scores(world):
    root, (cb, cc) = world
    p = _rep(root, cc) / "robustness_scores.csv"
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = [k for k in rows[0] if k != "illum_ramp_lr_1.0_0.8"]
    _write_csv(p, [{k: r[k] for k in keys} for r in rows], keys)
    with pytest.raises(RuntimeError, match="lacks conditions"):
        illum.illum_compare(cb, cc)


def test_refuses_config_without_predeclared_condition(tmp_path):
    def drop_n1(v, raw):
        raw["robustness"]["perturbations"] = [p for p in raw["robustness"]["perturbations"]
                                              if p.get("protocol_id") != "N1"]
    s = default_scores()
    cb, cc = build_world(tmp_path, s, drop_n1)
    with pytest.raises(RuntimeError, match="predeclared conditions missing"):
        illum.illum_compare(cb, cc)


def test_refuses_mismatched_split(world):
    root, (cb, cc) = world
    p = _rep(root, cc) / "frozen_operating_point.json"
    fz = json.loads(p.read_text(encoding="utf-8"))
    fz["split_manifest_sha256"] = "0" * 64
    p.write_text(json.dumps(fz), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different split manifest"):
        illum.illum_compare(cb, cc)


def test_refuses_split_manifest_changed_after_freeze(world):
    root, (cb, cc) = world
    _rewrite(root / "manifests" / "split_pcb3.csv", lambda r: r[:-1])
    with pytest.raises(RuntimeError, match="different split manifest"):
        illum.illum_compare(cb, cc)


def test_refuses_config_changed_after_freeze(world):
    root, (cb, cc) = world
    cc = copy.deepcopy(cc)
    cc["preprocess"]["photometric"]["ref_mean"] = 100.0
    with pytest.raises(RuntimeError, match="different config"):
        illum.illum_compare(cb, cc)


@pytest.mark.parametrize("damage", ["missing_holdout", "nan_score", "missing_column", "missing_robustness_json",
                                    "metrics_other_freeze", "threshold_mismatch", "base_scores_differ"])
def test_refuses_invalid_or_incomplete_artifacts(world, damage):
    root, (cb, cc) = world
    rep = _rep(root, cc)
    if damage == "missing_holdout":
        (rep / "scores_holdout.csv").unlink()
    elif damage == "nan_score":
        _rewrite(rep / "scores_test.csv", lambda r: [{**r[0], "score": "nan"}] + r[1:])
    elif damage == "missing_column":
        p = rep / "scores_test.csv"
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        _write_csv(p, [{k: v for k, v in r.items() if k != "score"} for r in rows],
                   [k for k in rows[0] if k != "score"])
    elif damage == "missing_robustness_json":
        (rep / "robustness.json").unlink()
    elif damage == "metrics_other_freeze":
        m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
        m["frozen_operating_point"]["config_hash"] = "x"
        (rep / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    elif damage == "threshold_mismatch":
        r = json.loads((rep / "robustness.json").read_text(encoding="utf-8"))
        r["threshold"] = 0.9
        (rep / "robustness.json").write_text(json.dumps(r), encoding="utf-8")
    elif damage == "base_scores_differ":
        _rewrite(rep / "robustness_scores.csv", lambda r: [{**r[0], "base": "9.000000"}] + r[1:])
    with pytest.raises(RuntimeError):
        illum.illum_compare(cb, cc)


@pytest.mark.parametrize("damage", ["coverage_above_one", "fallback_contradicts_coverage", "fallback_not_binary"])
def test_refuses_invalid_roi_metadata(world, damage):
    root, (cb, cc) = world
    p = _rep(root, cb) / "scores_holdout.csv"
    patch = {"coverage_above_one": {"roi_coverage": "1.200000"},
             "fallback_contradicts_coverage": {"roi_fallback": "1"},
             "fallback_not_binary": {"roi_fallback": "yes"}}[damage]
    _rewrite(p, lambda r: [{**r[0], **patch}] + r[1:])
    with pytest.raises(RuntimeError, match="ROI"):
        illum.illum_compare(cb, cc)


def test_real_visa_annotation_format():
    """Regression: the real PCB3 image_anno.csv uses defect types as labels (attempt-1 refusal)."""
    anno = REPO_ROOT / "data" / "visa" / "pcb3" / "image_anno.csv"
    if not anno.exists():
        pytest.skip("VisA PCB3 not extracted")
    rows = list(csv.DictReader(open(anno, encoding="utf-8")))
    an = [r for r in rows if "/Anomaly/" in r["image"]]
    assert an and all(r["label"] not in ("", "normal", "anomaly") and r["mask"] for r in an)


@pytest.mark.parametrize("damage", ["empty_mask", "missing_mask_path", "missing_mask_file", "size_mismatch",
                                    "labelled_normal"])
def test_refuses_invalid_gt_mask_metadata(world, damage):
    root, (cb, cc) = world
    vis = root / "data" / "visa"
    anno = vis / OBJ / "image_anno.csv"
    first = f"{OBJ}/Data/Masks/Anomaly/000.png"
    if damage == "empty_mask":
        Image.new("L", (64, 48), 0).save(vis / first)
    elif damage == "missing_mask_path":
        _rewrite(anno, lambda r: [{**r[0], "mask": ""}] + r[1:])
    elif damage == "missing_mask_file":
        (vis / first).unlink()
    elif damage == "size_mismatch":
        Image.new("L", (32, 24), 255).save(vis / first)
    elif damage == "labelled_normal":
        _rewrite(anno, lambda r: [{**r[0], "label": "normal"}] + r[1:])
    with pytest.raises(ValueError, match="invalid mask metadata"):
        illum.illum_compare(cb, cc)


def test_gt_masks_never_reach_fit_calibration_or_roi_code():
    """Ground-truth anomaly masks (image_anno.csv) are read only by post-freeze reporting/analysis code."""
    src = REPO_ROOT / "src" / "ivad"
    for name in ("pipeline.py", "calibration.py", "roi.py", "align.py", "photometric.py", "gaussian.py",
                 "features.py", "robustness.py", "data.py", "fitdiag.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "image_anno" not in text and "Masks" not in text, name
    readers = sorted(p.name for p in src.glob("*.py") if "image_anno" in p.read_text(encoding="utf-8"))
    assert readers == ["illum.py", "report.py"]
