"""End-to-end CLI smoke test on a tiny synthetic dataset (CPU, random weights).

Checks the workflow runs, results are repeatable with fixed seeds, and the
freeze guard blocks evaluation after a config change.
"""
import csv
import json

import pytest
import yaml

from ivad.cli import main
from ivad.config import DEFAULT_CONFIG

from .conftest import make_board

N_TRAIN, N_TN, N_TA = 24, 6, 6


@pytest.fixture
def project(tmp_path):
    rows = []
    for i in range(N_TRAIN):
        rows.append(("train", "normal", f"syn/N/{i:03d}.png", make_board(i, flip=i % 5 == 0, dx=i % 7 - 3)))
    for i in range(N_TN):
        rows.append(("test", "normal", f"syn/TN/{i:03d}.png", make_board(100 + i, dx=2)))
    for i in range(N_TA):
        rows.append(("test", "anomaly", f"syn/A/{i:03d}.png", make_board(200 + i, anomaly=True, flip=i % 2 == 0)))
    for split, label, rel, img in rows:
        p = tmp_path / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
    man = tmp_path / "manifests" / "official.csv"
    man.parent.mkdir(parents=True)
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "split", "label", "image", "mask"])
        for split, label, rel, _ in rows:
            w.writerow(["syn", split, label, rel, ""])

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"].update({"root": str(tmp_path), "data_dir": "data", "official_manifest": "manifests/official.csv"})
    cfg["dataset"].update({"object": "syn", "n_fit": 14, "n_calib": 6, "n_holdout": 4,
                           "n_test_normal": N_TN, "n_test_anomaly": N_TA})
    cfg["preprocess"].update({"crop_wh": [240, 160], "size_hw": [64, 96]})
    cfg["model"].update({"backbone": "resnet18", "pretrained": False, "n_features": 16, "map_sigma": 1.0})
    cfg["calibration"].update({"candidates": {"P80": 0.2, "P70": 0.3}, "primary": "P80"})
    cfg["evaluation"].update({"n_bootstrap": 50, "latency_warmup": 1})
    cfg["robustness"].update({"n_bootstrap": 50, "perturbations": [
        {"name": "dark", "kind": "photometric", "brightness": 0.9, "contrast": 0.9},
        {"name": "shift", "kind": "shift", "dx_px": 4, "dy_px": 3}]})
    cfg["runtime"].update({"device": "cpu", "batch_size": 4, "num_workers": 2})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp_path, path


def test_cli_end_to_end_repeatable_and_guarded(project):
    root, cfg_path = project
    c = ["--config", str(cfg_path)]
    for step in ("split", "fit"):
        assert main(c + [step]) == 0
    with pytest.raises(RuntimeError, match="calibrate"):
        main(c + ["evaluate"])  # no freeze yet
    for step in ("calibrate", "evaluate", "robustness"):
        assert main(c + [step]) == 0

    m = json.loads((root / "reports" / "metrics.json").read_text())
    ot = m["official_test"]
    assert ot["n_normal"] == N_TN and ot["n_anomaly"] == N_TA
    assert 0 <= ot["auroc"]["value"] <= 1
    cm = ot["operating_points"]["P80"]["confusion"]
    assert sum(cm.values()) == N_TN + N_TA
    assert m["supplementary_holdout"]["n_normal"] == 4
    rb = json.loads((root / "reports" / "robustness.json").read_text())
    assert set(rb["perturbations"]) == {"dark", "shift"}

    # report runs on a foreign project root and must not touch the real README
    from ivad.config import REPO_ROOT
    real_readme = (REPO_ROOT / "README.md").read_bytes()
    assert main(c + ["report"]) == 0
    assert (REPO_ROOT / "README.md").read_bytes() == real_readme
    assert (root / "reports" / "figures" / "fig_cost_sensitivity.png").exists()
    assert "Official VisA SYN test" in (root / "reports" / "results.md").read_text(encoding="utf-8")

    img = root / "data" / "syn" / "A" / "000.png"
    assert main(c + ["predict", str(img), "--out", str(root / "hm.png")]) == 0
    assert (root / "hm.png").exists()

    # Repeatability: re-fitting with the same seed reproduces identical scores.
    first = (root / "reports" / "scores_test.csv").read_text()
    assert main(c + ["fit"]) == 0 and main(c + ["calibrate"]) == 0 and main(c + ["evaluate"]) == 0
    assert (root / "reports" / "scores_test.csv").read_text() == first

    # Changing a semantic setting after freezing must block evaluation.
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["model"]["map_sigma"] = 2.0
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError):
        main(c + ["evaluate"])


def _variant(cfg_path, name, scoring, **eval_extra):
    """Same data/model settings, own artifacts/reports, given scoring section."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["paths"].update({"artifacts": f"artifacts/{name}", "reports": f"reports/{name}",
                         "figures": f"reports/{name}/figures"})
    cfg["scoring"] = scoring
    cfg["evaluation"].update(eval_extra)
    out = cfg_path.with_name(f"{name}.yaml")
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


ROI = {"aggregation": "roi_max", "roi": {"vote_fraction": 0.5, "margin_px": 4, "coverage_min": 0.95,
                                         "structure_k": 10, "structure_far_px": 6, "closing_px": 2,
                                         "fit_pass_frac_min": 0.99, "min_area_frac": 0.05,
                                         "max_area_frac": 0.95, "calib_fallback_frac_max": 0.05}}


def test_cli_v1_vs_v2_share_model_and_differ_only_in_aggregation(project):
    root, cfg_path = project
    base = _variant(cfg_path, "v1", {"aggregation": "full_max"})
    new = _variant(cfg_path, "v2", ROI)
    assert main(["--config", str(base), "split"]) == 0
    for cfg in (base, new):
        for step in ("fit", "calibrate", "evaluate"):
            assert main(["--config", str(cfg), step]) == 0

    import torch
    s1 = torch.load(root / "artifacts" / "v1" / "model.pt", weights_only=True)
    s2 = torch.load(root / "artifacts" / "v2" / "model.pt", weights_only=True)
    assert "roi" not in s1 and s2["roi"].dtype == torch.bool and s2["roi_qc"]["status"] == "PASS"
    for k in ("mean", "whitener"):  # identical Gaussian state: only the aggregation differs
        assert torch.equal(s1["gaussian"][k], s2["gaussian"][k])
    assert torch.equal(s1["template"], s2["template"])

    def rows(name, role):
        with open(root / "reports" / name / f"scores_{role}.csv", newline="") as f:
            return list(csv.DictReader(f))
    for role in ("calib", "test", "holdout"):
        a, b = rows("v1", role), rows("v2", role)
        assert [r["image"] for r in a] == [r["image"] for r in b]
        assert set(b[0]) == {"image", "role", "label", "score", "score_full", "roi_coverage", "roi_fallback"}
        for ra, rb in zip(a, b):
            assert ra["score"] == rb["score_full"]  # v2's full-frame max reproduces v1 exactly
            assert float(rb["score"]) <= float(rb["score_full"])
    fz = json.loads((root / "reports" / "v2" / "frozen_operating_point.json").read_text())
    assert fz["aggregation"] == "roi_max" and fz["roi"]["fit_qc"]["status"] == "PASS"
    m = json.loads((root / "reports" / "v2" / "metrics.json").read_text())
    assert set(m["roi_fallback"]) == {"test", "holdout"}

    assert main(["--config", str(new), "roi"]) == 0
    assert (root / "reports" / "v2" / "roi_qc.png").exists()
    assert main(["--config", str(new), "report"]) == 0
    assert "Board ROI" in (root / "reports" / "v2" / "results.md").read_text(encoding="utf-8")
    img = root / "data" / "syn" / "A" / "000.png"
    assert main(["--config", str(new), "predict", str(img)]) == 0

    # A model and a config that disagree about the ROI are rejected.
    from ivad.config import load_config
    from ivad.pipeline import load_detector
    cross = load_config(new)
    cross["paths"]["artifacts"] = "artifacts/v1"
    with pytest.raises(RuntimeError, match="config hash|different config|ROI"):
        load_detector(cross)


def test_committed_freeze_guard(project, tmp_path):
    import subprocess
    root, cfg_path = project
    cfg = _variant(cfg_path, "g", {"aggregation": "full_max"}, require_committed_freeze=True)
    (root / "src").mkdir()
    (root / "src" / "x.py").write_text("x = 1\n")
    c = ["--config", str(cfg)]
    for step in ("split", "fit", "calibrate"):
        assert main(c + [step]) == 0

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=root, check=True,
                       capture_output=True)
    git("init", "-q")
    with pytest.raises(RuntimeError, match="not committed"):
        main(c + ["evaluate"])
    git("add", "-A")
    git("commit", "-q", "-m", "freeze")
    assert main(c + ["evaluate"]) == 0
    m = json.loads((root / "reports" / "g" / "metrics.json").read_text())
    assert len(m["git_head_at_evaluation"]) == 40
    (root / "src" / "x.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="src/"):
        main(c + ["evaluate"])
