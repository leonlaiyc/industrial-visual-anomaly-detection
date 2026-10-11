"""PCB3: independent recomputation, freeze chronology and the post-evaluation repair disclosure.

Git-history checks need the private development history (commits eb4e7a9, 115c54b, 7dbd302, 9c9f7c5).
In a clean export without that history they are SKIPPED, never passed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
RECORD = REP / "pcb3_illum_comparison" / "repair_record.json"
REPAIR_DOC = ROOT / "docs" / "pcb3_post_evaluation_repair.md"
SCORE_FILES = [f"reports/{v}/{f}" for v in ("pcb3_baseline", "pcb3_illum")
               for f in ("scores_calib.csv", "frozen_operating_point.json", "scores_test.csv", "scores_holdout.csv",
                         "robustness_scores.csv", "metrics.json", "robustness.json")]


def _recompute_module():
    spec = importlib.util.spec_from_file_location("recompute_pcb3", ROOT / "verification" / "recompute_pcb3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def _has_commit(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def _need_history(*shas: str) -> None:
    missing = [s for s in shas if not _has_commit(s)]
    if missing:
        pytest.skip(f"development history not present (commits {missing}); clean export")


@pytest.fixture(scope="module")
def recomputed():
    mod = _recompute_module()
    o = mod.recompute()
    mod.compare(o)
    return mod, o


def test_independent_recomputation_agrees(recomputed):
    _, o = recomputed
    bad = [c for c in o["checks"] if c["ok"] is False]
    assert not bad, bad
    assert sum(c["ok"] is True for c in o["checks"]) >= 80


def test_verifier_does_not_import_project_code():
    src = (ROOT / "verification" / "recompute_pcb3.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+(ivad|src)\b", src, re.M)


def test_clean_metrics_and_denominators(recomputed):
    _, o = recomputed
    b, c = o["variants"]["baseline"], o["variants"]["candidate"]
    assert (b["recall"]["x"], b["recall"]["n"], c["recall"]["x"]) == (90, 100, 91)
    assert (b["fpr_official_test"]["x"], b["fpr_official_test"]["n"], c["fpr_official_test"]["x"]) == (3, 101, 3)
    assert (b["fpr_holdout"]["x"], b["fpr_holdout"]["n"], c["fpr_holdout"]["x"]) == (8, 155, 6)
    assert (b["fpr_pooled_unseen_normals"]["n"], b["fpr_pooled_unseen_normals"]["x"], c["fpr_pooled_unseen_normals"]["x"]) == (256, 11, 9)
    assert round(b["auroc"], 3) == 0.982 and round(c["auroc"], 3) == 0.983
    for v in (b, c):
        assert v["rank_k"] == 144 and v["n_calib"] == 150
        assert abs(v["threshold_recomputed"] - v["threshold_frozen"]) < 1e-6


def test_robustness_counts(recomputed):
    _, o = recomputed
    want = {"G1": (22, 8), "B1": (50, 16), "N1": (16, 15), "blur": (12, 46), "shift": (11, 11)}
    for pid, (fb, fc) in want.items():
        assert o["variants"]["baseline"]["conditions"][pid]["fpr_perturbed"]["x"] == fb
        assert o["variants"]["candidate"]["conditions"][pid]["fpr_perturbed"]["x"] == fc
    assert round(o["variants"]["candidate"]["conditions"]["blur"]["fpr_change_pts"], 1) == 14.5
    assert round(o["variants"]["baseline"]["conditions"]["B1"]["fpr_change_pts"], 1) == 15.2


def test_primary_endpoint_and_rule(recomputed):
    _, o = recomputed
    g1 = o["conditions"]["G1"]["fpr_normals"]
    assert round(100 * g1["did"], 1) == -4.7
    assert round(100 * g1["ci_low"], 1) == -8.6 and round(100 * g1["ci_high"], 1) == -0.8
    assert o["decision"]["outcome"] == "ROBUSTNESS IMPROVEMENT SUPPORTED"
    assert o["decision"]["weakly_informative"] is True
    assert o["decision"]["outcome_robust_to_bootstrap_seed"] is True
    acc = {v: o["variants"][v]["acceptance"] for v in ("baseline", "candidate")}
    assert acc["baseline"]["verdict"] == acc["candidate"]["verdict"] == "INCONCLUSIVE"
    assert acc["baseline"]["robustness"] is False and acc["candidate"]["robustness"] is False


def test_subgroup_table_from_masks():
    mod = _recompute_module()
    o = {"checks": []}
    mod.recompute_subgroups(o)
    if o["subgroups"]["status"] == "SKIPPED":
        pytest.skip(o["subgroups"]["reason"])
    assert o["checks"][-1]["ok"] is True
    assert "anomaly" not in o["subgroups"]["labels_seen"]  # VisA stores defect types (the repaired premise)


def test_artifact_hashes_in_repair_record_match_files():
    doc = REPAIR_DOC.read_text(encoding="utf-8")
    rows = dict(re.findall(r"\| `(reports/[^`]+)` \| `[0-9a-f]{7}` \| `([0-9a-f]{64})` \|", doc))
    assert set(SCORE_FILES) <= set(rows), sorted(set(SCORE_FILES) - set(rows))
    for rel, h in rows.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h, rel


def test_repair_record_chronology_order():
    r = json.loads(RECORD.read_text(encoding="utf-8"))
    t = lambda s: datetime.fromisoformat(s)  # noqa: E731
    inv = r["analysis_invocations"]
    seq = [r["freeze"]["committed_at"], r["evaluation"]["started_at"], r["evaluation"]["ended_at"],
           inv[0]["started_at"], inv[1]["started_at"], r["evaluation"]["outputs_committed_at"],
           r["repair"]["committed_at"], r["paired_analysis"]["committed_at"]]
    assert [t(x) for x in seq] == sorted(t(x) for x in seq)
    assert [i["result"].split()[0] for i in inv] == ["refused", "refused", "completed:"]
    assert r["repair"]["scores_regenerated"] is False and r["repair"]["test_predictions_rerun"] is False
    assert not any(v is True for v in r["repair"]["affects"].values())


def test_repair_patch_touches_only_the_label_check():
    diff = (REP / "pcb3_illum_comparison" / "repair_7dbd302.diff").read_text(encoding="utf-8")
    files = re.findall(r"^diff --git a/(\S+)", diff, re.M)
    assert files == ["src/ivad/illum.py", "tests/test_illum_e2e.py"]
    src = diff.split("diff --git a/tests/")[0]
    hunks = re.findall(r"^@@ .*?@@ (.*)$", src, re.M)
    assert hunks == ["def defect_descriptors(cfg: dict[str, Any], images: list[str]) -> dict[str, np.n"]
    changed = [ln for ln in src.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert len(changed) == 5 and all("label" in ln or "raise ValueError" in ln or ln.startswith("+        #")
                                     for ln in changed)


def test_commits_and_times_match_history():
    r = json.loads(RECORD.read_text(encoding="utf-8"))
    pairs = [(r["freeze"]["commit"], r["freeze"]["committed_at"]),
             (r["evaluation"]["outputs_commit"], r["evaluation"]["outputs_committed_at"]),
             (r["repair"]["commit"], r["repair"]["committed_at"]),
             (r["paired_analysis"]["commit"], r["paired_analysis"]["committed_at"])]
    _need_history(*(c for c, _ in pairs))
    for sha, when in pairs:
        assert _git("show", "-s", "--format=%cI", sha).stdout.strip() == when


def test_patch_file_equals_git_diff():
    _need_history("eb4e7a9", "7dbd302")
    want = _git("diff", "eb4e7a9", "7dbd302", "--", "src", "tests").stdout
    got = (REP / "pcb3_illum_comparison" / "repair_7dbd302.diff").read_text(encoding="utf-8")
    assert got == want


def test_scores_and_thresholds_written_once_and_unchanged():
    _need_history("eb4e7a9", "115c54b")
    for rel in SCORE_FILES:
        commits = _git("log", "--format=%h", "--", rel).stdout.split()
        assert len(commits) == 1, (rel, commits)
        assert commits[0] in ("eb4e7a9", "115c54b"), (rel, commits)


def test_no_score_affecting_change_since_freeze():
    _need_history("eb4e7a9")
    paths = ["config", "manifests"] + [f"src/ivad/{m}.py" for m in
                                       ("pipeline", "photometric", "align", "roi", "robustness", "features",
                                        "gaussian", "calibration", "data", "config", "metrics")]
    assert _git("diff", "--stat", "eb4e7a9", "HEAD", "--", *paths).stdout.strip() == ""
    # the only src change between freeze and paired analysis is the disclosed repair
    assert _git("diff", "--name-only", "eb4e7a9", "9c9f7c5", "--", "src").stdout.split() == ["src/ivad/illum.py"]
    # the protocol itself is frozen; post-test events live in the deviation log
    assert _git("diff", "--stat", "eb4e7a9", "HEAD", "--", "docs/illumination_experiment_protocol.md").stdout.strip() == ""


def test_disclosure_present_everywhere():
    link = "pcb3_post_evaluation_repair.md"
    for rel in ("README.md", "docs/pcb3_illumination_results.md", "docs/illumination_experiment_deviations.md"):
        assert link in (ROOT / rel).read_text(encoding="utf-8"), rel
    doc = REPAIR_DOC.read_text(encoding="utf-8")
    for must in ("eb4e7a9", "7dbd302", "16:58:51", "17:14:22", "twice", "not presented as unchanged",
                 "Summary **test** metrics", "judgment for the independent auditor"):
        assert must in doc, must
    for content in ("site/content/en.yaml", "site/content/zh-TW.yaml"):
        assert "f.repair.doc" in (ROOT / content).read_text(encoding="utf-8")
