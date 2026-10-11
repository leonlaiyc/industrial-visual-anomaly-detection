"""Exploratory E1 (exploratory/yolo_pcb4/): the recorded results recompute from the raw per-image scores, the split
is leak-free, and the study stays out of the frozen PCB1-PCB3 artefacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ivad.calibration import conformal_threshold

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "exploratory" / "yolo_pcb4" / "results"


def _rows(name):
    return list(csv.DictReader(open(E1 / name, newline="", encoding="utf-8")))


@pytest.fixture(scope="module")
def results():
    if not (E1 / "results.json").exists():
        pytest.skip("exploratory analysis not run")
    return json.loads((E1 / "results.json").read_text(encoding="utf-8"))


def test_split_is_disjoint_and_complete():
    split = _rows("split.csv")
    roles = {}
    for r in split:
        assert r["image"] not in roles
        roles[r["image"]] = r["role"]
    assert len({r["sha256"] for r in split}) == len(split)  # no exact duplicate across the whole split
    counts = {k: sum(v == k for v in roles.values()) for k in set(roles.values())}
    assert counts["fit"] == 600 and counts["calib"] == 150
    assert all(counts[f"fold{k}"] == 20 for k in range(1, 6))
    assert json.loads((E1 / "split_audit.json").read_text(encoding="utf-8"))["ok"] is True


def test_reference_recall_recomputes_from_scores(results):
    s = _rows("scores_m.csv")
    thr = conformal_threshold([float(r["score"]) for r in s if r["role"] == "calib"], 0.05)[0]
    hits = sum(float(r["score"]) > thr for r in s if r["label"] == "anomaly")
    assert hits == results["methods"]["m"]["recall_x"]


@pytest.mark.parametrize("arch", ["yolov8s", "yolo11s"])
def test_yolo_out_of_fold_recall_recomputes(results, arch):
    seen, hits = set(), 0
    for k in range(1, 6):
        rows = _rows(f"scores_{arch}_cv_fold{k}.csv")
        thr = conformal_threshold([float(r["score"]) for r in rows if r["set"] == "calib"], 0.05)[0]
        test = [r for r in rows if r["set"] == "test"]
        assert all(r["role"] == f"fold{k}" for r in test)  # a fold model scores only its own held-out defects
        assert not seen & {r["image"] for r in test}
        seen |= {r["image"] for r in test}
        hits += sum(float(r["score"]) > thr for r in test)
    assert len(seen) == 100 and hits == results["methods"][arch]["recall_x"]


def test_exploratory_study_does_not_touch_frozen_artefacts():
    """No commit that touched exploratory/ changed a frozen config, split manifest or evaluation report."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    commits = git("log", "--format=%H", "--", "exploratory/").split()
    assert commits
    frozen = ("config/", "manifests/", "reports/", "src/")
    for c in commits:
        files = git("show", "--name-only", "--format=", c).split()
        bad = [f for f in files if f.startswith(frozen) and not f.startswith("reports/executive_summary_")]
        assert not bad, (c[:7], bad)
    for cfg in (ROOT / "exploratory" / "yolo_pcb4").glob("config_*.yaml"):
        text = cfg.read_text(encoding="utf-8")
        assert "artifacts/exploratory_yolo_pcb4/" in text and "reports/pcb" not in text


AUDIT = ROOT / "exploratory" / "yolo_pcb4" / "audit"


def _run_audit(script, tmp_path):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(AUDIT / script), str(tmp_path)], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_audit_independent_recomputation_matches_recorded_results(tmp_path):
    """audit/recompute_metrics.py (no ivad import) reproduces every recorded number from the saved scores, and its
    committed output is what the script produces now."""
    _run_audit("recompute_metrics.py", tmp_path)
    fresh = json.loads((tmp_path / "recomputed_metrics.json").read_text(encoding="utf-8"))
    assert fresh["all_checks_ok"] and len(fresh["checks_vs_recorded_results_json"]) >= 50
    assert fresh == json.loads((AUDIT / "recomputed_metrics.json").read_text(encoding="utf-8"))
    assert fresh["denominators"]["test_normals_unique"] == 255
    assert fresh["denominators"]["yolo_test_normal_evaluations_per_arch"] == 5 * 255


def test_audit_fold_construction_and_leakage(tmp_path):
    _run_audit("check_leakage.py", tmp_path)
    fresh = json.loads((tmp_path / "leakage_audit.json").read_text(encoding="utf-8"))
    assert fresh["all_ok"] and len(fresh["checks"]) >= 70
    assert fresh == json.loads((AUDIT / "leakage_audit.json").read_text(encoding="utf-8"))


def test_audit_localization_reproduces_the_original_post_hoc_counts():
    """The audit re-ran the frozen weights (inference only): the original post-hoc counts reproduce exactly, and no
    image-level flag changes with the re-predicted score."""
    loc = json.loads((E1 / "posthoc_localization.json").read_text(encoding="utf-8"))
    la = json.loads((AUDIT / "localization_audit.json").read_text(encoding="utf-8"))
    for a in ("yolov8s", "yolo11s"):
        p = la["pooled_cv"][a]
        assert p["n_flagged"] == loc[a]["n_flagged_defects"]
        assert p["top_touch_dilated16 (original definition)"] == loc[a]["top_box_on_defect"]
        assert p["any_above_touch_dilated16 (original 'any', int-truncated boxes)"] == loc[a]["any_flagging_box_on_defect"]
        assert p["flag_changes_if_reproduced_score_used"] == 0
        assert p["top_iou_ge_0.5"] < p["top_touch_dilated16 (original definition)"]  # the original rule is the lenient one


def test_public_text_does_not_rank_latency_or_overclaim_localisation():
    for lang in ("en", "zh-TW"):
        text = (ROOT / "site" / "content" / f"{lang}.yaml").read_text(encoding="utf-8")
        for bad in ("it was also faster", "速度也較快", "so the recall is not a shortcut", "因此高檢出率並不是靠拍攝批次之類的捷徑"):
            assert bad not in text, (lang, bad)
        assert "f.explore.audit.fpr_dark" in text and "f.explore.audit.iou" in text
