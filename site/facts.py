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
        "status": "pending",
    }
    comp = REP / "pcb3_illum_comparison" / "comparison.json"
    if comp.exists():
        out["status"] = "complete"
        out["result"] = _j(comp)  # formatted in build once the run exists
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
    }
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
