"""The single source of every number on the website and in the executive PDF.

Reads only recorded project outputs (metrics.json, frozen_operating_point.json,
robustness.json, comparison.json, diagnostics.json, inspection statistics and,
once it exists, the PCB3 illumination comparison). Nothing here is typed in by
hand: content files reference these values as {{ f.<path> }} and the build
fails if a content file contains a literal metric (build.py, `lint_content`).

Each value is kept raw (for charts and checks) and pre-formatted (for text), so
the English and Chinese pages cannot drift apart numerically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def _j(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


MINUS = "−"


def pts(x: float, nd: int = 0) -> str:
    """Signed percentage points with a typographic minus ("+0" for zero)."""
    v = round(100 * x, nd)
    v = 0.0 if v == 0 else v  # no "-0"
    return f"{v:+.{nd}f}".replace("-", MINUS)


def pval(p: float) -> str:
    return f"{p:.2f}" if p >= 0.1 else f"{p:.2g}"


def num(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def rate(r: dict, nd: int = 0) -> dict[str, Any]:
    """A proportion with its exact CI, raw and formatted."""
    return {"value": r["value"], "x": r["x"], "n": r["n"], "pct": pct(r["value"], nd),
            "frac": f"{r['x']}/{r['n']}", "ci": f"{pct(r['ci_low'], nd)}–{pct(r['ci_high'], nd)}",
            "ci_low_pct": pct(r["ci_low"], nd), "ci_high_pct": pct(r["ci_high"], nd)}


def run(name: str) -> dict[str, Any]:
    d = REP if name == "pcb1_v1" else REP / name
    m, fz = _j(d / "metrics.json"), _j(d / "frozen_operating_point.json")
    pr = fz["primary"]
    op = m["official_test"]["operating_points"][pr]
    hold = m["supplementary_holdout"]["operating_points"][pr]["fpr"]
    au = m["official_test"]["auroc"]
    c = op["confusion"] if "confusion" in op else None
    out = {"recall": rate(op["recall"]), "fpr": rate(op["fpr"]), "fpr_holdout": rate(hold, 1),
           "auroc": {"value": au["value"], "fmt": num(au["value"], 3),
                     "ci": f"{num(au['ci_low'], 3)}–{num(au['ci_high'], 3)}"},
           "latency_ms": {"value": m["latency"]["median_ms"], "fmt": f"{m['latency']['median_ms']:.0f}"},
           "threshold": {k: {"value": v["threshold"], "fmt": num(v["threshold"], 2)}
                         for k, v in fz["thresholds"].items()},
           "primary": pr, "n_calib": fz["thresholds"][pr]["n_calib"], "rank_k": fz["thresholds"][pr]["rank_k"],
           "frozen_at_utc": fz["frozen_at_utc"], "evaluated_at_utc": m["evaluated_at_utc"]}
    if c:
        out["confusion"] = c
    rb = d / "robustness.json"
    if rb.exists():
        r = _j(rb)
        out["robustness"] = {
            "n_normal": r["images"]["normal"], "n_anomaly": r["images"]["anomaly"],
            "base_fpr": rate(r["base"]["fpr"], 1),
            **{k: {"fpr": rate(v["fpr"], 1), "d_fpr": pts(v["paired"]["delta_fpr"]["value"], 1),
                   "d_fpr_value": v["paired"]["delta_fpr"]["value"],
                   "d_recall": pts(v["paired"]["delta_recall"]["value"]),
                   "d_recall_value": v["paired"]["delta_recall"]["value"]}
               for k, v in r["perturbations"].items()}}
    return out


def paired(c: dict) -> dict[str, Any]:
    def d(r, nd=0):
        return {"delta": pts(r["delta"], nd),
                "ci_low": pts(r["ci_low"], nd), "ci_high": pts(r["ci_high"], nd),
                "p": pval(r["p_value"]),
                "p_value": r["p_value"], "disc_new": r["flagged_by_new_only"], "disc_base": r["flagged_by_base_only"]}
    p = c["paired"]
    a = p["auroc"]
    def sg(x):
        return f"{x:+.3f}".replace("-", MINUS)
    return {"recall": d(p["recall_test_anomalies"]), "fpr_test": d(p["fpr_test_normals"]),
            "fpr_holdout": d(p["fpr_holdout_normals"], 1),
            "auroc": {"delta": sg(a["delta"]), "ci_low": sg(a["ci_low"]), "ci_high": sg(a["ci_high"])},
            "outcome": c["decision"]["outcome"]}


def mechanism(c: dict) -> dict[str, Any]:
    t = c["threshold_mechanism"]
    return {"top10_off_roi": t["top10_calib_full_frame_scores_off_roi"],
            "threshold_base": num(t["threshold_base"], 1), "threshold_new": num(t["threshold_new"], 1),
            "anomalies_between": t["test_anomalies_between_thresholds"]}


def cost(recall: float, fpr: float) -> dict[str, Any]:
    """Break-even interval of k against review-none (cost k) and review-all (cost 1)."""
    lo, hi = fpr / recall, (1 - fpr) / (1 - recall)
    return {"miss": num(1 - recall, 2), "fpr": num(fpr, 2), "k_low": lo, "k_high": hi,
            "k_low_fmt": f"{lo:.4f}".rstrip("0"), "k_high_fmt": f"{hi:.2f}"}


def _did(d: dict, nd: int = 1) -> dict[str, Any]:
    r = d["rates"]
    return {"base_clean": pct(r["base_clean"], nd), "base_pert": pct(r["base_pert"], nd),
            "cand_clean": pct(r["cand_clean"], nd), "cand_pert": pct(r["cand_pert"], nd),
            "d_base": pts(d["delta_base"], nd), "d_cand": pts(d["delta_cand"], nd), "did": pts(d["did"], nd),
            "ci_low": pts(d["ci_low"], nd), "ci_high": pts(d["ci_high"], nd),
            "d_base_value": d["delta_base"], "d_cand_value": d["delta_cand"], "did_value": d["did"],
            "ci_low_value": d["ci_low"], "ci_high_value": d["ci_high"]}


def _pcb3_result(c: dict) -> dict[str, Any]:
    """Formatted PCB3 comparison (reports/pcb3_illum_comparison/comparison.json)."""
    cl, d = c["clean"], c["decision"]

    def pr(r, nd=0):
        return {"base": pct(r["base"], nd), "cand": pct(r["new"], nd), "delta": pts(r["delta"], nd),
                "ci_low": pts(r["ci_low"], nd), "ci_high": pts(r["ci_high"], nd), "p": pval(r["p_value"]),
                "base_value": r["base"], "cand_value": r["new"], "delta_value": r["delta"]}
    a = cl["auroc"]
    sg = c["subgroups"]
    nu = c["nonuniform_limitation"]
    acc = c["acceptance"]
    cond = {pid: {"fpr": _did(v["fpr_normals"]), "recall": _did(v["recall_anomalies"], 0)}
            for pid, v in c["conditions"].items()}
    return {
        "outcome": d["outcome"], "weak": d["baseline_rise_weakly_informative"],
        "recall_gate": d["clean_recall_gate_pass"], "fpr_gate": all(d["clean_fpr_gate_pass"].values()),
        "threshold": {"base": num(c["primary_threshold"]["base"][1], 1), "cand": num(c["primary_threshold"]["cand"][1], 1)},
        "recall": pr(cl["recall_test_anomalies"]), "fpr_test": pr(cl["fpr_test_normals"]),
        "fpr_holdout": pr(cl["fpr_holdout_normals"], 1),
        "auroc": {"base": num(a["base"], 3), "cand": num(a["new"], 3), "delta": f"{a['delta']:+.3f}".replace("-", MINUS),
                  "ci_low": f"{a['ci_low']:+.3f}".replace("-", MINUS), "ci_high": f"{a['ci_high']:+.3f}".replace("-", MINUS)},
        "cond": cond, "n_pooled": c["images"]["pooled_normals_for_conditions"],
        "n_anom": c["images"]["test_anomalies"], "n_test_normal": c["images"]["test_normals"],
        "n_holdout": c["images"]["holdout_normals"],
        "n1_exposed": nu["limitation_exposed"] if nu else None,
        "small": {k: {"n": v["n"], "base": v["detected_base"], "cand": v["detected_cand"]}
                  for k, v in sg["size_clean"].items()},
        "low_contrast": {k: {"n": v["n"], "base": v["detected_base"], "cand": v["detected_cand"]}
                         for k, v in sg["contrast_clean"].items()},
        "sensitivity_flag": sg["sensitivity_flag"]["flagged"],
        "latency": {"base": f"{c['runtime']['latency_median_ms']['base']:.0f}",
                    "cand": f"{c['runtime']['latency_median_ms']['cand']:.0f}",
                    "corr": f"{c['runtime']['correction_only_cpu']['ms_median']:.0f}"},
        "verdict": {"base": acc["base"]["verdict"], "cand": acc["cand"]["verdict"]},
        "fallback": {v: {role: c["roi_fallback"][v][role]["n_fallback"] for role in ("test", "holdout")}
                     for v in ("base", "cand")},
    }


def _pcb3_diag() -> dict[str, Any]:
    """Addendum B2 FIT-only diagnostics (reports/pcb3_inspection/fit_diagnostics_B2.json)."""
    r = _j(REP / "pcb3_inspection" / "fit_diagnostics_B2.json")
    s, roi, dom = r["summary"], r["roi_specificity"]["baseline"], r["background_dominance"]
    out = {"n_fit": r["data"]["n_images"]}
    for v, k in (("baseline", "base"), ("candidate", "cand")):
        for c in ("G1", "B1", "N1"):
            x = s[v][c]
            out[f"{k}_{c}"] = {"warp_median": num(x["warp_discrepancy_px"]["p50"], 1),
                               "warp_p95": num(x["warp_discrepancy_px"]["p95"], 0),
                               "over20": x["warp_discrepancy_over_20px"],
                               "iou_median": num(x["iou_vs_p0"]["p50"], 2), "fallback": x["roi_fallback_count"]}
    out["p0_residual_median"] = num(s["baseline"]["P0"]["template_residual_shift_px"]["p50"], 1)
    out["roi_area"] = pct(roi["roi_area_frac_of_model_frame"])
    out["roi_core"] = pct(roi["consensus_core_area_frac"])
    out["roi_cloth"] = pct(s["baseline"]["P0"]["roi_cloth_frac"]["p50"])
    out["roi_outside_mask"] = pct(s["baseline"]["P0"]["roi_outside_mask_frac"]["p50"])
    out["outside_roi"] = pct(1 - roi["roi_area_frac_of_model_frame"])
    out["board_share"] = pct(dom["board_pixel_share_of_frame_mean"])
    out["corr_bg"] = num(dom["corr_frame_mean_vs_background_mean"], 2)
    out["corr_board"] = num(dom["corr_frame_mean_vs_board_mean"], 2)
    return out


def pcb3() -> dict[str, Any]:
    """PCB3 illumination experiment: pre-test facts always; results only once the run exists."""
    insp = REP / "pcb3_inspection"
    ph = _j(insp / "photometric_fit_stats.json")
    hue = _j(insp / "background_hue_fit.json")
    reg = _j(insp / "inspection_stats.json")["registration_all_fit"]
    split = _j(ROOT / "manifests" / "split_pcb3_audit.json")["counts"]
    out: dict[str, Any] = {
        "split": {"fit": split["fit/normal"], "calib": split["calib/normal"], "holdout": split["holdout/normal"],
                  "test_normal": split["test/normal"], "test_anomaly": split["test/anomaly"]},
        "ref_mean": num(ph["ref_mean"], 1), "ref_std": num(ph["ref_std"], 1),
        "fit_mean_p05": num(ph["per_image_mean"]["p05"], 1), "fit_mean_p95": num(ph["per_image_mean"]["p95"], 1),
        "hue_band": f"{hue['exclude_hue'][0]}–{hue['exclude_hue'][1]}",
        "board_area_after": f"{pct(reg['board_area_fraction']['0'])}–{pct(reg['board_area_fraction']['100'])}",
        "diag": _pcb3_diag(),
        "status": "pending",
    }
    comp = REP / "pcb3_illum_comparison" / "comparison.json"
    if comp.exists():
        out["status"] = "complete"
        out["result"] = _pcb3_result(_j(comp))
        out["detail"] = _pcb3_detail()
        # Acceptance-only conditions (blur, shift) are not in the paired analysis: read each run's robustness.json.
        rb = {v: _j(REP / d / "robustness.json") for v, d in (("base", "pcb3_baseline"), ("cand", "pcb3_illum"))}
        for kind in ("blur", "shift"):
            out["result"][kind] = {}
            for v, r in rb.items():
                p = next(x for x in r["perturbations"].values() if x["spec"]["kind"] == kind)
                out["result"][kind][v] = {"fpr": pct(p["fpr"]["value"], 1), "rise": pts(p["paired"]["delta_fpr"]["value"], 1),
                                          "frac": f"{p['fpr']['x']}/{p['fpr']['n']}",
                                          "clean_frac": f"{r['base']['fpr']['x']}/{r['base']['fpr']['n']}"}
    return out


def _pcb3_detail() -> dict[str, Any]:
    """Exact counts per condition and the original acceptance verdicts, from each run's robustness.json and
    the paired comparison. Absolute rates and changes from clean are kept as separate fields."""
    c = _j(REP / "pcb3_illum_comparison" / "comparison.json")
    acc = c["acceptance"]
    out: dict[str, Any] = {"cond": {}, "acceptance": {}}
    for v, d in (("base", "pcb3_baseline"), ("cand", "pcb3_illum")):
        r = _j(REP / d / "robustness.json")
        m = _j(REP / d / "metrics.json")
        b = r["base"]
        out["cond"].setdefault("P0", {})[v] = {"fpr": pct(b["fpr"]["value"], 1), "fp": f"{b['fpr']['x']}/{b['fpr']['n']}",
                                               "rise": None, "tp": f"{b['recall']['x']}/{b['recall']['n']}"}
        worst = None
        for name, p in r["perturbations"].items():
            pid = p["spec"].get("protocol_id", p["spec"]["kind"])
            out["cond"].setdefault(pid, {})[v] = {
                "fpr": pct(p["fpr"]["value"], 1), "fp": f"{p['fpr']['x']}/{p['fpr']['n']}",
                "rise": pts(p["paired"]["delta_fpr"]["value"], 1), "tp": f"{p['recall']['x']}/{p['recall']['n']}",
                "recall_change": pts(p["paired"]["delta_recall"]["value"])}
            if p["spec"].get("acceptance", True) and (worst is None or p["paired"]["delta_fpr"]["value"] > worst[1]):
                worst = (pid, p["paired"]["delta_fpr"]["value"])
        op = m["official_test"]["operating_points"]["P95"]
        hold = m["supplementary_holdout"]["operating_points"]["P95"]["fpr"]
        a = acc[v]
        out["acceptance"][v] = {
            "recall": a["recall"], "fpr_test": a["fpr_test"], "fpr_holdout": a["fpr_holdout"], "auroc": a["auroc"],
            "latency": a["latency"], "robustness": a["robustness"], "verdict": a["verdict"],
            "no_go": a["no_go_triggers"], "worst_pid": worst[0], "worst_rise": pts(worst[1], 1),
            "recall_v": f"{pct(op['recall']['value'])} ({op['recall']['x']}/{op['recall']['n']})",
            "recall_ci": f"{pct(op['recall']['ci_low'])}–{pct(op['recall']['ci_high'])}",
            "fpr_test_v": f"{pct(op['fpr']['value'])} ({op['fpr']['x']}/{op['fpr']['n']})",
            "fpr_holdout_v": f"{pct(hold['value'], 1)} ({hold['x']}/{hold['n']})",
            "auroc_v": num(m["official_test"]["auroc"]["value"], 3),
            "latency_v": f"{m['latency']['median_ms']:.0f} ms"}
    return out


def boards() -> list[dict[str, Any]]:
    """One row per (board type, version/arm). Separate datasets: rows on different board types are NOT a
    matched comparison, and the site renders them without arrows."""
    rows = [("pcb1_v1", "PCB1", "pcb1_v1"), ("pcb1_v2_retro", "PCB1", "pcb1_v2_retrospective"),
            ("pcb2_v1", "PCB2", "pcb2_v1"), ("pcb2_v2", "PCB2", "pcb2_v2"),
            ("pcb3_base", "PCB3", "pcb3_baseline"), ("pcb3_cand", "PCB3", "pcb3_illum")]
    out = []
    for key, board, d in rows:
        r = run(d)
        out.append({"key": key, "board": board, "recall": r["recall"]["pct"], "recall_frac": r["recall"]["frac"],
                    "recall_ci": r["recall"]["ci"], "fpr": f"{r['fpr']['pct']} ({r['fpr']['frac']})",
                    "fpr_holdout": f"{r['fpr_holdout']['pct']} ({r['fpr_holdout']['frac']})", "auroc": r["auroc"]["fmt"]})
    return out


def repair() -> dict[str, Any]:
    """Post-evaluation analysis-code repair (reports/pcb3_illum_comparison/repair_record.json) and the
    independent recomputation result (reports/pcb3_verification/recompute.json)."""
    from datetime import datetime
    r = _j(REP / "pcb3_illum_comparison" / "repair_record.json")

    def hm(s: str) -> str:
        return datetime.fromisoformat(s).strftime("%H:%M")
    inv = r["analysis_invocations"]
    out = {"freeze_sha": r["freeze"]["commit"][:7], "freeze_time": hm(r["freeze"]["committed_at"]),
           "eval_start": hm(r["evaluation"]["started_at"]), "eval_end": hm(r["evaluation"]["ended_at"]),
           "refusal1": hm(inv[0]["started_at"]), "refusal2": hm(inv[1]["started_at"]),
           "n_refused": sum(i["result"].startswith("refused") for i in inv),
           "fix_sha": r["repair"]["commit"][:7], "fix_time": hm(r["repair"]["committed_at"]),
           "analysis_sha": r["paired_analysis"]["commit"][:7], "analysis_time": hm(r["paired_analysis"]["committed_at"]),
           "date": datetime.fromisoformat(r["freeze"]["committed_at"]).strftime("%Y-%m-%d"),
           "doc": "docs/pcb3_post_evaluation_repair.md", "deviations": "docs/illumination_experiment_deviations.md",
           "recompute_doc": "reports/pcb3_verification/recompute.md"}
    rc = REP / "pcb3_verification" / "recompute.json"
    if rc.exists():
        ch = _j(rc)["checks"]
        out["checks_ok"] = sum(c["ok"] is True for c in ch)
        out["checks_total"] = len(ch)
        out["checks_bad"] = sum(c["ok"] is False for c in ch)
    return out


def milestones() -> list[dict[str, Any]]:
    """Date-level milestones for the site (reports/project_milestones.json). Exact commit times stay in the
    repository's audit history; the site shows only verified dates. The AI-assisted cross-check is dated from
    its recorded verdict (`reviews`, not a commit; not a human sign-off); the owner's review carries no date: pending."""
    from datetime import datetime
    rec = _j(REP / "project_milestones.json")
    ms = {m["key"]: datetime.fromisoformat(m["committed_at"]) for m in rec["milestones"]}
    reviews = {r["key"]: r for r in rec.get("reviews", [])}
    groups = [("poc", ("first_commit", "pcb1_eval")), ("pcb2", ("v2_prereg", "pcb2_freeze", "pcb2_eval")),
              ("pcb3", ("pcb3_prereg", "pcb3_b2", "pcb3_freeze", "pcb3_eval"))]
    out = []
    for key, ks in groups:
        d0, d1 = ms[ks[0]].strftime("%Y-%m-%d"), ms[ks[-1]].strftime("%Y-%m-%d")
        out.append({"key": key, "iso": d0, "date": d0 if d0 == d1 else f"{d0} – {d1}", "pending": False})
    er = reviews.get("ai_cross_check")
    out.append({"key": "audit", "iso": er["date"], "date": er["date"], "pending": False} if er else
               {"key": "audit", "iso": "", "date": "", "pending": True})
    out.append({"key": "owner_review", "iso": "", "date": "", "pending": True})
    return out


def site_assets() -> dict[str, Any]:
    """Images and rescored workflow examples (site/make_assets.py; scores verified against recorded CSVs)."""
    a = _j(Path(__file__).parent / "assets_manifest.json")
    m = a["pcb3_mask_rules"]
    return {"workflow": {"cases": a["workflow_cases"]},
            "assets": {"pcb3_mask_rules": {"w": m["w"], "h": m["h"]},
                       "pcb3_roi": {"w": a["pcb3_roi"]["w"], "h": a["pcb3_roi"]["h"]}},
            "pcb3_mask": {"original": pct(m["original_rule_area_median"]), "adapted": pct(m["adapted_rule_area_median"]),
                          "n_fit": m["n_fit"]}}


# Repository files cited as evidence on the site (engineering-discipline section). The build fails if one is
# missing, so the site cannot cite a file that does not exist.
EVIDENCE_FILES = {
    "requirements": "docs/requirements.md", "acceptance": "config/acceptance_criteria.yaml",
    "decisions": "docs/engineering_decisions.md", "decision_record": "docs/v2_decision_record.md",
    "prereg_scoring": "docs/v2_experiment_protocol.md", "prereg_lighting": "docs/illumination_experiment_protocol.md",
    "milestones": "reports/project_milestones.json", "handoff": "docs/handoff.md",
    "config_v2": "config/pcb2_v2.yaml", "freeze_v2": "reports/pcb2_v2/frozen_operating_point.json",
    "pipeline": "src/ivad/pipeline.py", "deviations": "docs/illumination_experiment_deviations.md",
    "repair": "docs/pcb3_post_evaluation_repair.md", "tests": "tests/", "split_audit": "manifests/split_pcb2_audit.json",
    "data": "src/ivad/data.py", "consistency": "src/ivad/consistency.py", "recompute": "verification/recompute_pcb3.py",
    "lock": "requirements.lock.txt", "align": "src/ivad/align.py", "roi": "src/ivad/roi.py",
}


def files() -> dict[str, str]:
    missing = [p for p in EVIDENCE_FILES.values() if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"site cites missing evidence files: {missing}")
    return dict(EVIDENCE_FILES)


def explore() -> dict[str, Any]:
    """Exploratory E1 (exploratory/yolo_pcb4/): counts from the committed split audit; results only once the
    protocol's analysis has written results.json (`ready`)."""
    d = ROOT / "exploratory" / "yolo_pcb4" / "results"
    a = _j(d / "split_audit.json")
    c = a["counts"]
    n_def = sum(v for k, v in c.items() if k.startswith("fold"))
    out: dict[str, Any] = {"dir": "exploratory/yolo_pcb4/", "n_defects": n_def, "n_calib": c["calib"],
                           "n_test_normal": c["test_n"], "train_defects": n_def - n_def // 5,
                           "train_val": round(0.2 * (n_def - n_def // 5)),
                           "train_fit": (n_def - n_def // 5) - round(0.2 * (n_def - n_def // 5)),
                           "unseen_n_total": sum(v["test_single_label"] for v in a["unseen_type_runs"].values()),
                           "ready": False}
    rp = d / "results.json"
    if rp.exists():
        r = _j(rp)
        out.update(r["site"])
        out["ready"] = True
        out["n_test_normal"] = r["n_test_normal"]
    lp = d / "posthoc_localization.json"  # post-hoc check, reported as such
    if lp.exists():
        loc = _j(lp)
        out["loc"] = {a: {"top": f"{loc[a]['top_box_on_defect']}/{loc[a]['n_flagged_defects']}",
                          "any": f"{loc[a]['any_flagging_box_on_defect']}/{loc[a]['n_flagged_defects']}",
                          "tol": loc["tolerance_px"]} for a in ("yolov8s", "yolo11s")}
    au = ROOT / "exploratory" / "yolo_pcb4" / "audit"  # audit-time checks (after the experiment), reported as such
    if (au / "localization_audit.json").exists() and (au / "acquisition_confound.json").exists():
        la, ac = _j(au / "localization_audit.json"), _j(au / "acquisition_confound.json")
        rm = _j(au / "recomputed_metrics.json")
        um = la["runs"]["yolo11s_unseen_missing"]
        bm = ac["brightness_matched_fpr_sensitivity"]["methods"]
        mc = rm["mcnemar_defects"]["yolov8s_vs_yolo11s"]
        tn = rm["methods"]["yolov8s"]["test_normal_evaluations_pooled"]
        out["audit"] = {
            "iou_min": "0.5",
            "iou": {a: f"{la['pooled_cv'][a]['top_iou_ge_0.5']}/{la['pooled_cv'][a]['n_flagged']}" for a in ("yolov8s", "yolo11s")},
            "unseen_missing_flagged": um["n_flagged"], "unseen_missing_top_touch": um["top_touch_dilated16 (original definition)"],
            "unseen_missing_any_iou": um["any_above_iou_ge_0.5"],
            "dark_auroc": f"{1 - ac['features']['grey_mean']['auroc']:.2f}",
            "fpr_dark": {k: f"{100 * bm[k]['fpr_brightness_matched']:.1f}%" for k in bm},
            "recall_matched_cal": {k: f"{100 * bm[k]['matched_calibration']['recall']:.0f}%" for k in bm},
            "v8_only": mc["only_first"], "v11_only": mc["only_second"],
            "evals": f"{tn['evaluations']:,}", "unique_normals": tn["unique_images"],
            "yolo_imgsz": _j(d / "train_args_yolov8s_cv_fold1.json")["train_args"]["imgsz"],
            "ref_size": "×".join(str(v) for v in reversed(a["reference_size_hw"])),  # width × height
            "lat_n": _j(d / "latency_yolov8s.json")["n_images"],
            "dir": "exploratory/yolo_pcb4/audit/"}
    return out


def collect() -> dict[str, Any]:
    v1, b, n = run("pcb1_v1"), run("pcb2_v1"), run("pcb2_v2")
    c2 = _j(REP / "pcb2_comparison" / "comparison.json")
    c1 = _j(REP / "pcb1_comparison" / "comparison.json")
    d2 = _j(REP / "pcb2_v2" / "diagnostics.json")["defects"]
    d1 = _j(REP / "diagnostics.json")["defects"]
    reg1 = _j(REP / "inspection" / "inspection_stats.json")["registration_all_fit"]
    pc = c2["premise_check"]
    small = d2["by_mask_area_tercile"]["small"]
    p1r = c1["paired"]
    acc = c2["acceptance"]
    f = {
        "pcb1": {"v1": v1,
                 "v2_retro": {"recall": pct(p1r["recall_test_anomalies"]["new"]),
                              "fpr": pct(p1r["fpr_test_normals"]["new"]),
                              "auroc": num(p1r["auroc"]["new"], 3)},
                 "mechanism": mechanism(c1),
                 "fa_offboard": "13/13",
                 "rotated_180": reg1["rotated_180_count"], "n_fit": reg1["n_images"],
                 "rotated_180_pct": pct(reg1["rotated_180_count"] / reg1["n_images"]),
                 "melt_recall": pct(d1["by_type_multilabel"]["melt"]["recall"])},
        "pcb2": {"v1": b, "v2": n, "paired": paired(c2), "mechanism": mechanism(c2),
                 "premise": {"outside": pc["test"]["peak_outside_roi"] + pc["holdout"]["peak_outside_roi"],
                             "total": pc["test"]["n_false_alarms"] + pc["holdout"]["n_false_alarms"]},
                 "small": {"recall": pct(small["recall"]), "frac": f"{small['detected']}/{small['n']}"},
                 "verdict": {"v1": acc["base"]["verdict"], "v2": acc["new"]["verdict"]}},
        "cost": {"v2": cost(n["recall"]["value"], n["fpr"]["value"]),
                 "v1": cost(b["recall"]["value"], b["fpr"]["value"])},
        "criteria": _criteria(),
        "ci_level": pct(c2["paired"]["recall_test_anomalies"]["level"]),
        "target_fpr": pct(_j(REP / "pcb2_v2" / "frozen_operating_point.json")["thresholds"]["P95"]["target_fpr"]),
        "pcb3": pcb3(),
        "milestones": milestones(),
        "boards": boards(),
        "repair": repair(),
        "files": files(),
        "explore": explore(),
        **site_assets(),
    }
    f["pcb2"]["v2"]["roi_margin_px"] = _yaml(ROOT / "config" / "pcb2_v2.yaml")["scoring"]["roi"]["margin_px"]
    f["pcb2"]["v2"]["coverage_min"] = num(_yaml(ROOT / "config" / "pcb2_v2.yaml")["scoring"]["roi"]["coverage_min"], 2)
    dark = "darker_lower_contrast"
    import yaml
    spec = next(p for p in yaml.safe_load((ROOT / "config" / "pcb2_v2.yaml").read_text(encoding="utf-8"))
                ["robustness"]["perturbations"] if p["name"] == dark)
    f["pcb2"]["dark_pct"] = pct(1 - spec["brightness"])
    f["pcb2"]["dark"] = {v: {"clean": f["pcb2"][v]["robustness"]["base_fpr"]["pct"],
                             "dark": f["pcb2"][v]["robustness"][dark]["fpr"]["pct"],
                             "rise": f["pcb2"][v]["robustness"][dark]["d_fpr"]}
                         for v in ("v1", "v2")}
    return f


def _yaml(p: Path) -> Any:
    import yaml
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _criteria() -> dict[str, Any]:
    import yaml
    a = yaml.safe_load((ROOT / "config" / "acceptance_criteria.yaml").read_text(encoding="utf-8"))
    p, ng = a["performance"], a["no_go"]
    return {"recall_min": pct(p["recall_min"]), "fpr_max": pct(p["fpr_max"]), "auroc_min": num(p["auroc_min"], 2),
            "latency_max": f"{p['latency_ms_max'] / 1000:.0f} s", "rob_pts": f"{100 * p['robustness_fpr_rise_max']:.0f}",
            "recall_floor": pct(ng["recall_floor"]), "fpr_ceiling": pct(ng["fpr_ceiling"])}


def write(path: Path | None = None) -> dict[str, Any]:
    f = collect()
    path = path or Path(__file__).parent / "data" / "facts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(f, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return f


if __name__ == "__main__":
    write()
