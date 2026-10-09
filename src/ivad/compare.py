"""Paired v1-vs-v2 comparison on the same images (docs/v2_experiment_protocol.md §7-8).

Both runs must have been evaluated on identical test and holdout images. Each
model is judged at its own frozen primary threshold, as it would be operated.

* Δ Recall / Δ FPR: exact two-sided McNemar test on the discordant images
  (binomial test, p = 0.5) and a percentile bootstrap CI that resamples images,
  so each image keeps its v1 and v2 decisions together.
* Δ AUROC: stratified paired bootstrap; each replicate draws one set of images
  and scores both models on it.
* The pre-registered outcome (improvement supported / no meaningful effect /
  unfavourable trade-off) is computed by `classify`, nothing is judged by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from . import metrics
from .acceptance import acceptance_status, load_criteria
from .config import resolve_path, write_json
from .pipeline import model_path, read_scores


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def mcnemar_exact(base: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test for paired binary outcomes (True = flagged)."""
    base, new = np.asarray(base, bool), np.asarray(new, bool)
    if base.shape != new.shape or base.ndim != 1:
        raise ValueError("paired outcomes must be 1-D arrays of equal length")
    gained = int((~base & new).sum())  # flagged by new only
    lost = int((base & ~new).sum())  # flagged by base only
    n_disc = gained + lost
    p = 1.0 if n_disc == 0 else float(stats.binomtest(min(gained, lost), n_disc, 0.5).pvalue)
    return {"flagged_by_new_only": gained, "flagged_by_base_only": lost, "n_discordant": n_disc,
            "p_value": min(p, 1.0), "test": "exact McNemar (two-sided binomial on discordant pairs)"}


def paired_rate_delta(base: np.ndarray, new: np.ndarray, n_boot: int, seed: int,
                      level: float = 0.95) -> dict[str, Any]:
    """Δ rate (new − base) of paired binary outcomes with an image-resampling bootstrap CI."""
    base, new = np.asarray(base, bool), np.asarray(new, bool)
    d = new.astype(float) - base.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"base": float(base.mean()), "new": float(new.mean()), "delta": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "n": int(len(d)), "level": level,
            "ci_method": f"paired percentile bootstrap over images, B={n_boot}, seed={seed}",
            **mcnemar_exact(base, new)}


def paired_auroc_delta(labels: np.ndarray, s_base: np.ndarray, s_new: np.ndarray, n_boot: int, seed: int,
                       level: float = 0.95) -> dict[str, Any]:
    """Δ AUROC (new − base) with a stratified paired bootstrap over images."""
    y = np.asarray(labels, bool)
    ia, inn = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    lab = np.r_[np.ones(len(ia), bool), np.zeros(len(inn), bool)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[ia[rng.integers(0, len(ia), len(ia))], inn[rng.integers(0, len(inn), len(inn))]]
        boots[b] = metrics.auroc(lab, s_new[idx]) - metrics.auroc(lab, s_base[idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    a0, a1 = metrics.auroc(y, s_base), metrics.auroc(y, s_new)
    return {"base": a0, "new": a1, "delta": a1 - a0, "ci_low": float(lo), "ci_high": float(hi),
            "level": level, "ci_method": f"stratified paired percentile bootstrap, B={n_boot}, seed={seed}"}


def classify(recall: dict, fpr_test: dict, fpr_holdout: dict, alpha: float, margin: float) -> dict[str, Any]:
    """Pre-registered outcome (protocol §8) from the paired results."""
    noninf = {"official_test": fpr_test["new"] <= fpr_test["base"] + margin + 1e-12,
              "holdout": fpr_holdout["new"] <= fpr_holdout["base"] + margin + 1e-12}
    sig = recall["p_value"] < alpha
    if not all(noninf.values()) or (recall["delta"] < 0 and sig):
        outcome = "UNFAVOURABLE TRADE-OFF"
    elif recall["delta"] > 0 and sig:
        outcome = "IMPROVEMENT SUPPORTED"
    else:
        outcome = "NO MEANINGFUL EFFECT DEMONSTRATED"
    return {"outcome": outcome, "recall_significant": sig, "fpr_noninferior": noninf,
            "rule": f"supported iff Δrecall > 0 with McNemar p < {alpha} and v2 FPR <= v1 FPR + {margin} "
                    "on official-test and holdout normals; unfavourable iff FPR non-inferiority fails or "
                    f"Δrecall < 0 with p < {alpha}; otherwise no meaningful effect demonstrated"}


# --------------------------------------------------------------------------- #
# Run loading and consistency
# --------------------------------------------------------------------------- #
def _run(cfg) -> dict[str, Any]:
    rep = resolve_path(cfg, "reports")
    out = {"cfg": cfg, "rep": rep,
           "m": json.loads((rep / "metrics.json").read_text(encoding="utf-8")),
           "fz": json.loads((rep / "frozen_operating_point.json").read_text(encoding="utf-8"))}
    rb = rep / "robustness.json"
    out["rb"] = json.loads(rb.read_text(encoding="utf-8")) if rb.exists() else None
    for role in ("calib", "test", "holdout"):
        out[role] = read_scores(rep / f"scores_{role}.csv")
    return out


def _full_scores(path: Path) -> np.ndarray | None:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["score_full"]) for r in rows]) if rows and "score_full" in rows[0] else None


def pairing_checks(b: dict, n: dict) -> dict[str, Any]:
    """Same images, labels and split; identical Gaussian state; v2 full-frame scores == v1 scores."""
    out: dict[str, Any] = {}
    for role in ("calib", "test", "holdout"):
        rb_, rn = b[role][0], n[role][0]
        out[f"same_{role}_images_and_labels"] = (
            [(r["image"], r["label"]) for r in rb_] == [(r["image"], r["label"]) for r in rn])
        full = _full_scores(n["rep"] / f"scores_{role}.csv")
        out[f"v2_full_frame_equals_v1_score_{role}"] = (full is not None and len(full) == len(b[role][1])
                                                        and bool(np.array_equal(full, b[role][1])))
    out["same_split_manifest"] = b["fz"]["split_manifest_sha256"] == n["fz"]["split_manifest_sha256"]
    out["same_pretrained_weights"] = b["fz"]["pretrained_weights"] == n["fz"]["pretrained_weights"]
    try:
        sb = torch.load(model_path(b["cfg"]), map_location="cpu", weights_only=True)
        sn = torch.load(model_path(n["cfg"]), map_location="cpu", weights_only=True)
        out["identical_gaussian_state"] = all(torch.equal(sb["gaussian"][k], sn["gaussian"][k])
                                              for k in ("mean", "whitener"))
        out["identical_registration_template"] = torch.equal(sb["template"], sn["template"])
        out["identical_fit_images"] = sb["fit_images"] == sn["fit_images"]
    except FileNotFoundError:
        out["identical_gaussian_state"] = None  # model files absent (e.g. fresh clone); recorded, not assumed
    return out


def threshold_mechanism(b: dict, n: dict) -> dict[str, Any]:
    """How much did the ROI move the calibrated threshold, and how many anomalies sit in the gap?

    Uses only calibration scores (unlabelled normals) plus the test-anomaly scores that the
    paired endpoints already use. Descriptive.
    """
    rows, sc = n["calib"]
    full = _full_scores(n["rep"] / "scores_calib.csv")
    off = full > sc + 1e-9  # the full-frame maximum lies outside the ROI
    top = np.argsort(full)[-10:]
    tb = b["fz"]["thresholds"][b["fz"]["primary"]]["threshold"]
    tn = n["fz"]["thresholds"][n["fz"]["primary"]]["threshold"]
    trows, ts = n["test"]
    a = np.array([v for r, v in zip(trows, ts) if r["label"] == "anomaly"])
    return {"calib_normals_peak_off_roi": int(off.sum()), "n_calib": int(len(sc)),
            "top10_calib_full_frame_scores_off_roi": int(off[top].sum()),
            "threshold_base": tb, "threshold_new": tn, "threshold_drop": tb - tn,
            "test_anomalies_between_thresholds": int(((a > tn) & (a <= tb)).sum()),
            "note": "The ROI can only raise recall at a calibrated threshold if background peaks set the upper "
                    "tail of the calibration normals."}


def premise_check(b: dict, n: dict) -> dict[str, Any]:
    """Do v1's false alarms peak off the board ROI on PCB2? (Descriptive; needs the fitted v2 model.)"""
    from .pipeline import load_detector, score_images

    det = load_detector(n["cfg"])
    roi = det.roi.cpu().numpy()
    root = resolve_path(n["cfg"], "data_dir")
    t = b["fz"]["thresholds"][b["fz"]["primary"]]["threshold"]
    out = {}
    for role in ("test", "holdout"):
        rows, s = b[role]
        fps = [r for r, v in zip(rows, s) if r["label"] == "normal" and v > t]
        if not fps:
            out[role] = {"n_false_alarms": 0, "peak_outside_roi": 0}
            continue
        _, maps = score_images(det, [root / r["image"] for r in fps], keep_maps=True)
        outside = [bool(not roi[np.unravel_index(np.argmax(m), m.shape)]) for m in maps.astype(np.float32)]
        out[role] = {"n_false_alarms": len(fps), "peak_outside_roi": int(sum(outside)),
                     "images": [{"image": r["image"], "peak_outside_roi": o} for r, o in zip(fps, outside)]}
    out["note"] = ("v1 false alarms (normals above the v1 P95 threshold); peak = argmax of the full-frame "
                   "anomaly map (identical for v1 and v2); outside = not in the v2 ROI. Descriptive only.")
    return out


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def compare_runs(cfg_base, cfg_new, with_premise: bool = True) -> dict[str, Any]:
    b, n = _run(cfg_base), _run(cfg_new)
    c = cfg_new["comparison"]
    B, seed, level = c["n_bootstrap"], c["seed"], c["level"]
    checks = pairing_checks(b, n)
    hard = [k for k, v in checks.items() if v is False]
    if hard:
        raise RuntimeError(f"runs are not a valid paired comparison: {hard}")

    pb, pn = b["fz"]["primary"], n["fz"]["primary"]
    tb, tn = b["fz"]["thresholds"][pb]["threshold"], n["fz"]["thresholds"][pn]["threshold"]
    (trows, sb), (_, sn) = b["test"], n["test"]
    y = np.array([r["label"] == "anomaly" for r in trows])
    hb, hn = b["holdout"][1], n["holdout"][1]
    fb, fn = sb > tb, sn > tn

    recall = paired_rate_delta(fb[y], fn[y], B, seed, level)
    fpr_test = paired_rate_delta(fb[~y], fn[~y], B, seed + 1, level)
    fpr_hold = paired_rate_delta(hb > tb, hn > tn, B, seed + 2, level)
    auroc = paired_auroc_delta(y, sb, sn, B, seed + 3, level)
    decision = classify(recall, fpr_test, fpr_hold, c["alpha"], c["fpr_noninferiority_margin"])

    acc = load_criteria()
    retro = "RETROSPECTIVE" in (cfg_new.get("run_name") or "")
    out: dict[str, Any] = {
        "evidence_status": ("RETROSPECTIVE: discovery data that motivated v2; not evidence for the hypothesis"
                            if retro else "CONFIRMATORY: pre-registered, single evaluation on held-back data"),
        "base_run": cfg_base.get("run_name") or f"{cfg_base['dataset']['object'].upper()} v1 (original)",
        "new_run": cfg_new.get("run_name"),
        "dataset": cfg_new["dataset"]["object"], "primary_threshold": {"base": [pb, tb], "new": [pn, tn]},
        "pairing_checks": checks,
        "paired": {"recall_test_anomalies": recall, "fpr_test_normals": fpr_test,
                   "fpr_holdout_normals": fpr_hold, "auroc": auroc},
        "decision": decision,
        "acceptance": {"base": acceptance_status(b["m"], b["rb"], pb, acc),
                       "new": acceptance_status(n["m"], n["rb"], pn, acc)},
        "latency_median_ms": {"base": b["m"]["latency"]["median_ms"], "new": n["m"]["latency"]["median_ms"]},
        "roi_fallback": n["m"].get("roi_fallback"),
        "threshold_mechanism": threshold_mechanism(b, n),
    }
    if with_premise:
        out["premise_check"] = premise_check(b, n)
    out_dir = comparison_dir(cfg_new)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "comparison.json", out)
    write_comparison_md(out, out_dir / "comparison.md")
    r = recall
    print(f"compare: Δrecall {r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_value']:.4f}; "
          f"ΔFPR test {fpr_test['delta']:+.3f}, holdout {fpr_hold['delta']:+.3f}; ΔAUROC {auroc['delta']:+.4f} "
          f"-> {decision['outcome']}")
    return out


def comparison_dir(cfg_new) -> Path:
    return resolve_path(cfg_new, "reports").parent / f"{cfg_new['dataset']['object']}_comparison"


def _pp(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def write_comparison_md(o: dict, path: Path) -> None:
    p = o["paired"]
    L = [f"# Paired comparison on {o['dataset'].upper()}: {o['base_run']} vs {o['new_run']}",
         "(generated by `ivad compare`; do not edit by hand)", "", f"**Evidence status: {o['evidence_status']}.**", "",
         f"Primary thresholds (each frozen from its own calibration normals): v1 {o['primary_threshold']['base'][0]} = "
         f"{o['primary_threshold']['base'][1]:.2f}, v2 {o['primary_threshold']['new'][0]} = {o['primary_threshold']['new'][1]:.2f}.", "",
         "| Endpoint | v1 | v2 | Δ (v2 − v1) [95% CI] | Discordant (v2 only / v1 only) | McNemar p |",
         "|---|---|---|---|---|---|"]
    for name, key in (("Recall, test anomalies (primary)", "recall_test_anomalies"),
                      ("FPR, official test normals", "fpr_test_normals"),
                      ("FPR, supplementary holdout normals", "fpr_holdout_normals")):
        d = p[key]
        L.append(f"| {name} | {100 * d['base']:.1f}% | {100 * d['new']:.1f}% | {_pp(d['delta'])} "
                 f"[{_pp(d['ci_low'])}, {_pp(d['ci_high'])}] | {d['flagged_by_new_only']} / {d['flagged_by_base_only']} | "
                 f"{d['p_value']:.4f} |")
    a = p["auroc"]
    L.append(f"| AUROC, official test | {a['base']:.3f} | {a['new']:.3f} | {a['delta']:+.3f} "
             f"[{a['ci_low']:+.3f}, {a['ci_high']:+.3f}] | — | — |")
    d = o["decision"]
    label = ("Outcome under the pre-registered rule (retrospective, NOT evidence)"
             if o["evidence_status"].startswith("RETRO") else "Pre-registered outcome")
    tm = o["threshold_mechanism"]
    L += ["", f"**{label}: {d['outcome']}.** Rule: {d['rule']}.", "",
          "## Threshold mechanism (descriptive)", "",
          f"Calibration normals whose full-frame peak lies outside the ROI: {tm['calib_normals_peak_off_roi']}/{tm['n_calib']} "
          f"(top 10 by full-frame score: {tm['top10_calib_full_frame_scores_off_roi']}/10). Primary threshold "
          f"{tm['threshold_base']:.2f} -> {tm['threshold_new']:.2f} (drop {tm['threshold_drop']:.2f}); test anomalies "
          f"scoring between the two thresholds: {tm['test_anomalies_between_thresholds']}. {tm['note']}", "",
          "## Pairing checks", ""] + [f"- {k}: {v}" for k, v in o["pairing_checks"].items()]
    L += ["", "## Acceptance against the original proposed criteria (point estimates)", "",
          "| Criterion | v1 | v2 |", "|---|---|---|"]
    for k in ("auroc", "recall", "fpr_test", "fpr_holdout", "latency", "robustness"):
        f = lambda v: "Met" if v is True else "NOT met" if v is False else "not evaluated"  # noqa: E731
        L.append(f"| {k} | {f(o['acceptance']['base'][k])} | {f(o['acceptance']['new'][k])} |")
    L.append(f"| **Verdict** | **{o['acceptance']['base']['verdict']}** | **{o['acceptance']['new']['verdict']}** |")
    L.append(f"| NO-GO triggers | {', '.join(o['acceptance']['base']['no_go_triggers']) or 'none'} | "
             f"{', '.join(o['acceptance']['new']['no_go_triggers']) or 'none'} |")
    pc = o.get("premise_check")
    if pc:
        L += ["", "## Premise check (descriptive)", "", pc["note"], ""]
        for role in ("test", "holdout"):
            L.append(f"- {role}: {pc[role]['peak_outside_roi']}/{pc[role]['n_false_alarms']} v1 false alarms peak "
                     "outside the ROI")
    if o.get("roi_fallback"):
        L += ["", "ROI fail-safe fallbacks (v2): " + ", ".join(f"{k} {v['n_fallback']}/{v['n']}"
                                                            for k, v in o["roi_fallback"].items() if v)]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
