"""PCB3 illumination-correction experiment: reference statistics and the paired analysis.

Pre-registered in docs/illumination_experiment_protocol.md. Two runs on the same
PCB3 images: the v2 pipeline (baseline) and the same pipeline with the global
luminance standardisation of photometric.py (candidate). Each is judged at its
own frozen P95 threshold.

* `photostats`  FIT-only luminance statistics -> the frozen reference values (§3).
* `illum_compare`  the single planned paired analysis (§8):
    primary   DiD = [FPR_c(G1) - FPR_c(P0)] - [FPR_b(G1) - FPR_b(P0)] on the pooled
              unseen normals, paired bootstrap over image identities;
    gates     clean recall (Δ >= -5 pts and no significant loss) and clean FPR
              non-inferiority (+5 pts, official test and holdout);
    outcome   ROBUSTNESS IMPROVEMENT SUPPORTED / INCONCLUSIVE / UNFAVOURABLE TRADE-OFF,
              computed by `classify_illum`, nothing judged by hand;
    plus N1 (non-uniform) limitation check, size/contrast terciles, runtime.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from . import metrics, photometric
from .acceptance import acceptance_status, load_criteria
from .compare import _run, mcnemar_exact, paired_auroc_delta, paired_rate_delta
from .config import resolve_path, write_json
from .data import select
from .features import load_rgb

# Protocol condition IDs (§7) -> perturbation names in the PCB3 configs.
PRIMARY = "G1"
CLEAN_RECALL_MARGIN = 0.05  # §8: Δ recall point estimate below -5 pts fails the clean gate
CLEAN_FPR_MARGIN = 0.05  # §8: candidate FPR <= baseline FPR + 5 pts
ALPHA = 0.05
WEAK_BASELINE_RISE = 0.05  # §8: baseline G1 FPR rise below 5 pts -> primary endpoint weakly informative
SUBGROUP_DEFICIT_FLAG = 3  # §8 C: candidate detects >= 3 fewer in the small or low-contrast tercile
RING_PX = 12


# --------------------------------------------------------------------------- #
# Reference statistics (FIT normals only)
# --------------------------------------------------------------------------- #
def photostats(cfg: dict[str, Any]) -> dict[str, Any]:
    from .pipeline import load_rows, pmap

    rows = select(load_rows(cfg), "fit", "normal")
    if len(rows) != cfg["dataset"]["n_fit"]:
        raise ValueError(f"expected {cfg['dataset']['n_fit']} fit images, found {len(rows)}")
    root = resolve_path(cfg, "data_dir")
    ref = photometric.fit_reference(pmap(cfg)(lambda r: load_rgb(root / r["image"]), rows))
    ref["source"] = "FIT normals only (role=fit); no calibration, holdout or test image read"
    out = (resolve_path(cfg, "inspection") if "inspection" in cfg["paths"]
           else resolve_path(cfg, "reports")) / "photometric_fit_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, ref)
    print(f"photostats: ref_mean {ref['ref_mean']:.4f}, ref_std {ref['ref_std']:.4f} -> {out}")
    return ref


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def did_bootstrap(b0: np.ndarray, b1: np.ndarray, c0: np.ndarray, c1: np.ndarray, n_boot: int, seed: int,
                  level: float = 0.95) -> dict[str, Any]:
    """Difference in differences of paired flag rates, resampling image identities.

    b0/b1: baseline flags clean/perturbed; c0/c1: candidate flags clean/perturbed (same images).
    DiD = mean(c1 - c0) - mean(b1 - b0) = mean of the per-image d_i = (c1-c0)_i - (b1-b0)_i.
    """
    arrs = [np.asarray(a, bool).astype(float) for a in (b0, b1, c0, c1)]
    n = len(arrs[0])
    if any(a.shape != (n,) for a in arrs) or n == 0:
        raise ValueError("flag arrays must be 1-D, non-empty and of equal length")
    b0f, b1f, c0f, c1f = arrs
    d = (c1f - c0f) - (b1f - b0f)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"delta_base": float((b1f - b0f).mean()), "delta_cand": float((c1f - c0f).mean()),
            "did": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(n), "level": level,
            "rates": {"base_clean": float(b0f.mean()), "base_pert": float(b1f.mean()),
                      "cand_clean": float(c0f.mean()), "cand_pert": float(c1f.mean())},
            "ci_method": f"paired percentile bootstrap over image identities, B={n_boot}, seed={seed}"}


def classify_illum(did: dict, recall_clean: dict, fpr_test: dict, fpr_hold: dict) -> dict[str, Any]:
    """Pre-registered outcome rule (protocol §8)."""
    eps = 1e-12
    recall_gate = not (recall_clean["delta"] < -CLEAN_RECALL_MARGIN - eps
                       or (recall_clean["delta"] < 0 and recall_clean["p_value"] < ALPHA))
    fpr_gate = {"official_test": fpr_test["new"] <= fpr_test["base"] + CLEAN_FPR_MARGIN + eps,
                "holdout": fpr_hold["new"] <= fpr_hold["base"] + CLEAN_FPR_MARGIN + eps}
    gates_ok = recall_gate and all(fpr_gate.values())
    if not gates_ok or did["ci_low"] > 0:
        outcome = "UNFAVOURABLE TRADE-OFF"
    elif did["ci_high"] < 0:
        outcome = "ROBUSTNESS IMPROVEMENT SUPPORTED"
    else:
        outcome = "INCONCLUSIVE"
    return {"outcome": outcome, "clean_recall_gate_pass": recall_gate, "clean_fpr_gate_pass": fpr_gate,
            "primary_did_ci_excludes_zero": did["ci_high"] < 0 or did["ci_low"] > 0,
            "baseline_rise_weakly_informative": did["delta_base"] < WEAK_BASELINE_RISE,
            "rule": "supported iff DiD 95% CI upper < 0 and both clean gates pass; unfavourable iff a clean "
                    "gate fails or DiD 95% CI lower > 0; otherwise inconclusive. Clean recall gate fails iff "
                    f"Δrecall < -{CLEAN_RECALL_MARGIN:.2f} or (Δrecall < 0 and McNemar p < {ALPHA}); clean FPR "
                    f"gate fails iff candidate FPR > baseline FPR + {CLEAN_FPR_MARGIN:.2f} on official-test or "
                    "holdout normals."}


def thirds(values: np.ndarray, keys: list[str]) -> np.ndarray:
    """Tercile labels 0/1/2 by rank (ties broken by image path), sizes as equal as possible."""
    order = sorted(range(len(values)), key=lambda i: (values[i], keys[i]))
    lab = np.empty(len(values), int)
    for g, idx in enumerate(np.array_split(np.array(order), 3)):
        lab[idx] = g
    return lab


def defect_descriptors(cfg: dict[str, Any], images: list[str]) -> dict[str, np.ndarray]:
    """Mask area fraction and local contrast of each test anomaly (clean native image, GT mask).

    Contrast = |mean grey inside the mask - mean grey in a ring 1..RING_PX px outside it|.
    Descriptive only (protocol §8 C); never used for any setting.
    """
    root = resolve_path(cfg, "data_dir")
    anno_path = root / cfg["dataset"]["object"] / "image_anno.csv"
    with open(anno_path, encoding="utf-8") as f:
        anno = {r["image"]: r for r in csv.DictReader(f)}
    area, contrast = [], []
    for rel in images:
        mk = np.asarray(Image.open(root / anno[rel]["mask"])) > 0
        if mk.ndim == 3:
            mk = mk.any(axis=2)
        g = np.asarray(load_rgb(root / rel).convert("L"), np.float64)
        if mk.shape != g.shape:
            raise ValueError(f"mask/image size mismatch for {rel}")
        dist = ndimage.distance_transform_edt(~mk)
        ring = (dist > 0) & (dist <= RING_PX)
        area.append(float(mk.mean()))
        contrast.append(abs(float(g[mk].mean()) - float(g[ring].mean())) if mk.any() and ring.any() else 0.0)
    return {"mask_area_frac": np.array(area), "contrast": np.array(contrast)}


def subgroup_table(lab: np.ndarray, values: np.ndarray, fb: np.ndarray, fc: np.ndarray,
                   names=("low", "medium", "high")) -> dict[str, Any]:
    out = {}
    for g, name in enumerate(names):
        sel = lab == g
        out[name] = {"n": int(sel.sum()), "value_range": [float(values[sel].min()), float(values[sel].max())],
                     "detected_base": int(fb[sel].sum()), "detected_cand": int(fc[sel].sum()),
                     "recall_base": float(fb[sel].mean()), "recall_cand": float(fc[sel].mean()),
                     "cand_only": int((fc & ~fb)[sel].sum()), "base_only": int((fb & ~fc)[sel].sum())}
    return out


# --------------------------------------------------------------------------- #
# Run loading
# --------------------------------------------------------------------------- #
def _robustness_scores(rep: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    with open(rep / "robustness_scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = [k for k in rows[0] if k not in ("image", "label")]
    return rows, {k: np.array([float(r[k]) for r in rows]) for k in cols}


def condition_ids(cfg: dict[str, Any]) -> dict[str, str]:
    """protocol_id -> perturbation name, from the config."""
    return {p["protocol_id"]: p["name"] for p in cfg["robustness"]["perturbations"] if "protocol_id" in p}


def correction_runtime(cfg: dict[str, Any], n: int = 50) -> dict[str, Any]:
    """CPU time of the correction step alone on FIT images (no test data needed)."""
    from .pipeline import load_rows

    corr = photometric.from_config(cfg)
    if corr is None:
        return {"ms_median": 0.0, "n_images": 0}
    root = resolve_path(cfg, "data_dir")
    imgs = [load_rgb(root / r["image"]) for r in select(load_rows(cfg), "fit", "normal")[:n]]
    t = []
    for im in imgs:
        t0 = time.perf_counter()
        corr(im)
        t.append((time.perf_counter() - t0) * 1000)
    return {"ms_median": float(np.median(t)), "ms_p95": float(np.percentile(t, 95)), "n_images": len(t),
            "definition": "luminance statistics + affine map + uint8 rounding on the native frame, CPU, "
                          "single thread"}


# --------------------------------------------------------------------------- #
# The planned paired analysis
# --------------------------------------------------------------------------- #
def illum_compare(cfg_base: dict[str, Any], cfg_cand: dict[str, Any]) -> dict[str, Any]:
    b, c = _run(cfg_base), _run(cfg_cand)
    st = cfg_cand["illumination_study"]
    B, seed, level = st["n_bootstrap"], st["seed"], st["level"]

    # Pairing: same images, labels and roles; same split; different preprocessing only.
    checks = {f"same_{role}_images_and_labels": [(r["image"], r["label"]) for r in b[role][0]]
              == [(r["image"], r["label"]) for r in c[role][0]] for role in ("calib", "test", "holdout")}
    checks["same_split_manifest"] = b["fz"]["split_manifest_sha256"] == c["fz"]["split_manifest_sha256"]
    checks["same_pretrained_weights"] = b["fz"]["pretrained_weights"] == c["fz"]["pretrained_weights"]
    rrows_b, rs_b = _robustness_scores(b["rep"])
    rrows_c, rs_c = _robustness_scores(c["rep"])
    checks["same_robustness_images"] = ([(r["image"], r["label"]) for r in rrows_b]
                                        == [(r["image"], r["label"]) for r in rrows_c])
    ids_b, ids_c = condition_ids(cfg_base), condition_ids(cfg_cand)
    checks["same_condition_definitions"] = (ids_b == ids_c and
                                            cfg_base["robustness"]["perturbations"]
                                            == cfg_cand["robustness"]["perturbations"])
    bad = [k for k, v in checks.items() if v is not True]
    if bad:
        raise RuntimeError(f"runs are not a valid paired comparison: {bad}")

    tb = b["fz"]["thresholds"][b["fz"]["primary"]]["threshold"]
    tc = c["fz"]["thresholds"][c["fz"]["primary"]]["threshold"]
    (trows, sb), (_, sc) = b["test"], c["test"]
    y = np.array([r["label"] == "anomaly" for r in trows])
    hb, hc = b["holdout"][1], c["holdout"][1]

    # Clean gates (question B) and clean secondary metrics
    recall = paired_rate_delta(sb[y] > tb, sc[y] > tc, B, seed, level)
    fpr_test = paired_rate_delta(sb[~y] > tb, sc[~y] > tc, B, seed, level)
    fpr_hold = paired_rate_delta(hb > tb, hc > tc, B, seed, level)
    auroc = paired_auroc_delta(y, sb, sc, B, seed, level)

    # Every condition on the pooled unseen normals and the anomalies (robustness rows)
    ry = np.array([r["label"] == "anomaly" for r in rrows_b])
    fb0, fc0 = rs_b["base"] > tb, rs_c["base"] > tc
    conditions: dict[str, Any] = {}
    # Every bootstrap uses the protocol seed with its own generator (protocol §8: B = 10000, seed 2026).
    for pid, name in sorted(ids_c.items()):
        fb1, fc1 = rs_b[name] > tb, rs_c[name] > tc
        conditions[pid] = {
            "perturbation": name,
            "spec": next(p for p in cfg_cand["robustness"]["perturbations"] if p["name"] == name),
            "fpr_normals": did_bootstrap(fb0[~ry], fb1[~ry], fc0[~ry], fc1[~ry], B, seed, level),
            "recall_anomalies": did_bootstrap(fb0[ry], fb1[ry], fc0[ry], fc1[ry], B, seed, level),
            "perturbed_flags_cand_vs_base_normals": mcnemar_exact(fb1[~ry], fc1[~ry]),
            "perturbed_flags_cand_vs_base_anomalies": mcnemar_exact(fb1[ry], fc1[ry]),
        }
    if PRIMARY not in conditions:
        raise RuntimeError("primary condition G1 missing from the configs")
    decision = classify_illum(conditions[PRIMARY]["fpr_normals"], recall, fpr_test, fpr_hold)

    # Question D: non-uniform illumination against the original P5 limits
    acc = load_criteria()
    lim_fpr, lim_rec = (acc["performance"]["robustness_fpr_rise_max"],
                        acc["performance"]["robustness_recall_drop_max"])
    n1 = conditions.get("N1")
    nonuniform = None if n1 is None else {
        "cand_fpr_rise": n1["fpr_normals"]["delta_cand"], "cand_recall_change": n1["recall_anomalies"]["delta_cand"],
        "limitation_exposed": (n1["fpr_normals"]["delta_cand"] > lim_fpr + 1e-12
                               or -n1["recall_anomalies"]["delta_cand"] > lim_rec + 1e-12),
        "rule": f"exposed iff candidate N1 FPR rise > {lim_fpr} or recall drop > {lim_rec} (original P5 limits)"}

    # Question C: size and contrast terciles of the clean test anomalies
    an_rows = [r for r in trows if r["label"] == "anomaly"]
    desc = defect_descriptors(cfg_cand, [r["image"] for r in an_rows])
    keys = [r["image"] for r in an_rows]
    fb_a, fc_a = sb[y] > tb, sc[y] > tc
    g1 = conditions[PRIMARY]["perturbation"]
    an_idx = np.flatnonzero(ry)  # robustness rows: anomalies in the same order as the test anomalies
    if [rrows_b[i]["image"] for i in an_idx] != keys:
        raise RuntimeError("robustness anomaly rows are not in test order")
    size_lab, con_lab = thirds(desc["mask_area_frac"], keys), thirds(desc["contrast"], keys)
    subgroups = {
        "size_clean": subgroup_table(size_lab, desc["mask_area_frac"], fb_a, fc_a, ("small", "medium", "large")),
        "contrast_clean": subgroup_table(con_lab, desc["contrast"], fb_a, fc_a),
        "size_G1": subgroup_table(size_lab, desc["mask_area_frac"], rs_b[g1][an_idx] > tb, rs_c[g1][an_idx] > tc,
                                  ("small", "medium", "large")),
        "contrast_G1": subgroup_table(con_lab, desc["contrast"], rs_b[g1][an_idx] > tb, rs_c[g1][an_idx] > tc),
        "note": "POST-FREEZE DESCRIPTION, pre-declared in protocol §8 C: GT masks used for description only.",
    }
    deficit = {k: subgroups[t][g]["detected_base"] - subgroups[t][g]["detected_cand"]
               for k, t, g in (("small", "size_clean", "small"), ("low_contrast", "contrast_clean", "low"))}
    subgroups["sensitivity_flag"] = {"deficits": deficit,
                                     "flagged": any(v >= SUBGROUP_DEFICIT_FLAG for v in deficit.values()),
                                     "rule": f"flag iff candidate detects >= {SUBGROUP_DEFICIT_FLAG} fewer anomalies "
                                             "than the baseline in the small or the low-contrast tercile"}

    out: dict[str, Any] = {
        "evidence_status": "CONFIRMATORY: pre-registered (docs/illumination_experiment_protocol.md), "
                           "single planned evaluation on PCB3, never used before",
        "base_run": cfg_base.get("run_name"), "cand_run": cfg_cand.get("run_name"),
        "dataset": cfg_cand["dataset"]["object"],
        "primary_threshold": {"base": [b["fz"]["primary"], tb], "cand": [c["fz"]["primary"], tc]},
        "git_head_at_evaluation": {"base": b["m"].get("git_head_at_evaluation"),
                                   "cand": c["m"].get("git_head_at_evaluation")},
        "pairing_checks": checks,
        "images": {"test_anomalies": int(y.sum()), "test_normals": int((~y).sum()), "holdout_normals": len(hb),
                   "pooled_normals_for_conditions": int((~ry).sum())},
        "clean": {"recall_test_anomalies": recall, "fpr_test_normals": fpr_test,
                  "fpr_holdout_normals": fpr_hold, "auroc": auroc},
        "conditions": conditions,
        "primary_endpoint": {"condition": PRIMARY, **conditions[PRIMARY]["fpr_normals"]},
        "decision": decision,
        "nonuniform_limitation": nonuniform,
        "subgroups": subgroups,
        "runtime": {"latency_median_ms": {"base": b["m"]["latency"]["median_ms"],
                                          "cand": c["m"]["latency"]["median_ms"]},
                    "latency_p95_ms": {"base": b["m"]["latency"]["p95_ms"], "cand": c["m"]["latency"]["p95_ms"]},
                    "correction_only_cpu": correction_runtime(cfg_cand)},
        "acceptance": {"base": acceptance_status(b["m"], b["rb"], b["fz"]["primary"], acc),
                       "cand": acceptance_status(c["m"], c["rb"], c["fz"]["primary"], acc)},
        "roi_fallback": {"base": b["m"].get("roi_fallback"), "cand": c["m"].get("roi_fallback")},
    }
    out_dir = resolve_path(cfg_cand, "illum_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "comparison.json", out)
    write_illum_md(out, out_dir / "comparison.md")
    p = out["primary_endpoint"]
    print(f"illum-compare: G1 ΔFPR base {p['delta_base']:+.3f}, cand {p['delta_cand']:+.3f}, DiD {p['did']:+.3f} "
          f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]; clean Δrecall {recall['delta']:+.3f} "
          f"(p={recall['p_value']:.3f}) -> {decision['outcome']}")
    return out


def _pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def write_illum_md(o: dict[str, Any], path: Path) -> None:
    c, d = o["clean"], o["decision"]
    L = [f"# PCB3 illumination experiment: {o['base_run']} vs {o['cand_run']}", "",
         f"*{o['evidence_status']}.* Generated by `ivad illum-compare` from recorded per-image scores.", "",
         f"**Pre-registered outcome: {d['outcome']}**", "",
         f"Thresholds (P95, frozen): baseline {o['primary_threshold']['base'][1]:.3f}, "
         f"candidate {o['primary_threshold']['cand'][1]:.3f}.", "",
         "## Clean condition (official test 100 anomalies + 101 normals; holdout 155 normals)", "",
         "| Metric | Baseline | Candidate | Δ (pts) [95% CI] | McNemar p |", "|---|---|---|---|---|"]
    for name, r in (("Recall", c["recall_test_anomalies"]), ("FPR, test normals", c["fpr_test_normals"]),
                    ("FPR, holdout normals", c["fpr_holdout_normals"])):
        L.append(f"| {name} | {100 * r['base']:.1f}% | {100 * r['new']:.1f}% | {_pp(r['delta'])} "
                 f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}] | {r['p_value']:.3f} |")
    a = c["auroc"]
    L += [f"| AUROC | {a['base']:.3f} | {a['new']:.3f} | {a['delta']:+.3f} [{a['ci_low']:+.3f}, "
          f"{a['ci_high']:+.3f}] | — |", "",
          f"Clean recall gate: {'pass' if d['clean_recall_gate_pass'] else 'FAIL'}; clean FPR gate: "
          f"{'pass' if all(d['clean_fpr_gate_pass'].values()) else 'FAIL'}.", "",
          "## Conditions (pooled unseen normals; test anomalies)", "",
          "| ID | Perturbation | ΔFPR baseline | ΔFPR candidate | DiD FPR [95% CI] | ΔRecall baseline | "
          "ΔRecall candidate |", "|---|---|---|---|---|---|---|"]
    for pid, v in o["conditions"].items():
        f_, r_ = v["fpr_normals"], v["recall_anomalies"]
        L.append(f"| {pid}{' (primary)' if pid == PRIMARY else ''} | {v['perturbation']} | {_pp(f_['delta_base'])} | "
                 f"{_pp(f_['delta_cand'])} | {_pp(f_['did'])} [{_pp(f_['ci_low'])}, {_pp(f_['ci_high'])}] | "
                 f"{_pp(r_['delta_base'])} | {_pp(r_['delta_cand'])} |")
    if d["baseline_rise_weakly_informative"]:
        L += ["", "**Note:** the baseline's own G1 false-alarm rise is below 5 pts, so PCB3 did not reproduce the "
                  "PCB2 failure and the primary endpoint is weakly informative on this board type."]
    nu = o["nonuniform_limitation"]
    if nu:
        L += ["", f"Non-uniform illumination (N1): candidate FPR rise {_pp(nu['cand_fpr_rise'])} pts, recall change "
                  f"{_pp(nu['cand_recall_change'])} pts -> limitation exposed: **{nu['limitation_exposed']}** "
                  f"({nu['rule']})."]
    L += ["", "## Small and low-contrast anomalies (clean; descriptive)", "",
          "| Tercile | n | Baseline detected | Candidate detected | candidate-only / baseline-only |",
          "|---|---|---|---|---|"]
    for t in ("size_clean", "contrast_clean"):
        for g, v in o["subgroups"][t].items():
            L.append(f"| {t.split('_')[0]}: {g} | {v['n']} | {v['detected_base']} | {v['detected_cand']} | "
                     f"{v['cand_only']} / {v['base_only']} |")
    sf = o["subgroups"]["sensitivity_flag"]
    L += ["", f"Sensitivity flag: **{sf['flagged']}** (deficits {sf['deficits']}; {sf['rule']}).", "",
          "## Runtime", "",
          f"Median end-to-end latency: baseline {o['runtime']['latency_median_ms']['base']:.0f} ms, candidate "
          f"{o['runtime']['latency_median_ms']['cand']:.0f} ms. Correction step alone (CPU): "
          f"{o['runtime']['correction_only_cpu']['ms_median']:.1f} ms median.", "",
          "## Original acceptance criteria (unchanged)", "",
          f"Baseline verdict: {o['acceptance']['base']['verdict']}; candidate verdict: "
          f"{o['acceptance']['cand']['verdict']}. N1 is excluded from P5 (not in the original definition).", "",
          f"Rule: {d['rule']}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
