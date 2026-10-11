"""Independent recomputation of the PCB3 illumination experiment from the recorded per-image scores.

    python verification/recompute_pcb3.py            # writes reports/pcb3_verification/recompute.{json,md}
    python verification/recompute_pcb3.py --check    # exit 1 if any recorded value disagrees

Independence: this file imports nothing from `src/ivad`. It reads only
  * reports/pcb3_{baseline,illum}/scores_{calib,test,holdout}.csv and robustness_scores.csv (raw scores,
    written once by the single planned evaluation, commit 115c54b),
  * reports/pcb3_{baseline,illum}/frozen_operating_point.json (thresholds frozen at eb4e7a9),
  * config/pcb3_illum.yaml (condition names and protocol IDs) and config/acceptance_criteria.yaml,
and re-derives every count, rate, AUROC, paired statistic and decision with its own code. It then
compares each value with the recorded outputs (metrics.json, robustness.json,
pcb3_illum_comparison/comparison.json). Nothing is re-scored, re-fitted or re-calibrated, and no
recorded file is modified.

Bootstrap intervals: (a) re-run with the protocol's stated method and seed (paired percentile bootstrap
over image identities, B = 10000, seed 2026) to confirm the recorded interval; (b) re-run with ten other
seeds to show the interval's sensitivity to the random stream. Point estimates and counts are exact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = REP / "pcb3_verification"
VARIANTS = {"baseline": "pcb3_baseline", "candidate": "pcb3_illum"}
B, SEED, LEVEL = 10_000, 2026, 0.95
ALT_SEEDS = list(range(1, 11))
TOL = 1e-9

# Protocol §8 constants, restated here from the protocol text (not imported from the analysis code).
CLEAN_RECALL_MARGIN, CLEAN_FPR_MARGIN, ALPHA, WEAK_RISE = 0.05, 0.05, 0.05, 0.05


# --------------------------------------------------------------------------- io
def read_csv(p: Path) -> list[dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(p: Path) -> str:
    """SHA-256 of the file bytes as stored in the working tree (LF line endings via .gitattributes)."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_variant(d: Path) -> dict[str, Any]:
    fz = json.loads((d / "frozen_operating_point.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"fz": fz, "thr": fz["thresholds"][fz["primary"]]["threshold"], "primary": fz["primary"]}
    for role in ("calib", "test", "holdout"):
        rows = read_csv(d / f"scores_{role}.csv")
        out[role] = {"rows": rows, "score": np.array([float(r["score"]) for r in rows]),
                     "anom": np.array([r["label"] == "anomaly" for r in rows])}
    rr = read_csv(d / "robustness_scores.csv")
    out["rob_rows"] = rr
    out["rob"] = {k: np.array([float(r[k]) for r in rr]) for k in rr[0] if k not in ("image", "label")}
    out["rob_anom"] = np.array([r["label"] == "anomaly" for r in rr])
    return out


# --------------------------------------------------------------------------- statistics (own code)
def clopper_pearson(x: int, n: int, level: float = LEVEL) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a, x + 1, n - x))
    return lo, hi


def prop(flags: np.ndarray) -> dict[str, Any]:
    x, n = int(flags.sum()), int(flags.size)
    lo, hi = clopper_pearson(x, n)
    return {"x": x, "n": n, "value": x / n, "ci_low": lo, "ci_high": hi, "ci_method": "Clopper-Pearson exact"}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney U / (n1 n0) with mid-ranks for ties."""
    r = stats.rankdata(s)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar on discordant pairs. a = baseline flags, b = candidate flags."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return {"baseline_only": n10, "candidate_only": n01, "p_value": p}


def boot_mean(d: np.ndarray, seed: int, n_boot: int = B, level: float = LEVEL) -> tuple[float, float]:
    """Percentile bootstrap of a per-image mean, resampling image identities with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    m = d[idx].mean(axis=1)
    lo, hi = np.quantile(m, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def did(b0, b1, c0, c1) -> dict[str, Any]:
    f = [np.asarray(v, bool).astype(float) for v in (b0, b1, c0, c1)]
    d = (f[3] - f[2]) - (f[1] - f[0])
    lo, hi = boot_mean(d, SEED)
    alt = [boot_mean(d, s) for s in ALT_SEEDS]
    return {"n": int(d.size),
            "counts": {"base_clean": int(f[0].sum()), "base_pert": int(f[1].sum()),
                       "cand_clean": int(f[2].sum()), "cand_pert": int(f[3].sum())},
            "delta_base": float((f[1] - f[0]).mean()), "delta_cand": float((f[3] - f[2]).mean()),
            "did": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "alt_seed_ci_high_range": [min(a[1] for a in alt), max(a[1] for a in alt)],
            "alt_seed_ci_low_range": [min(a[0] for a in alt), max(a[0] for a in alt)]}


def paired_delta(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = b.astype(float) - a.astype(float)
    lo, hi = boot_mean(d, SEED)
    return {"base": float(a.mean()), "cand": float(b.mean()), "delta": float(d.mean()), "ci_low": lo, "ci_high": hi,
            **mcnemar(a, b)}


def order_stat_threshold(calib: np.ndarray, target_fpr: float) -> tuple[float, int]:
    """Split-conformal rank k = ceil((n + 1)(1 - alpha)), the k-th smallest calibration score."""
    n = calib.size
    k = min(n, math.ceil((n + 1) * (1 - target_fpr)))
    return float(np.sort(calib)[k - 1]), k


# --------------------------------------------------------------------------- recomputation
def recompute() -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "config" / "pcb3_illum.yaml").read_text(encoding="utf-8"))
    acc = yaml.safe_load((ROOT / "config" / "acceptance_criteria.yaml").read_text(encoding="utf-8"))
    perts = cfg["robustness"]["perturbations"]
    ids = {p.get("protocol_id", p["kind"]): p["name"] for p in perts}  # G1 B1 N1 blur shift
    in_p5 = {p["name"]: p.get("acceptance", True) for p in perts}

    V = {k: load_variant(REP / d) for k, d in VARIANTS.items()}
    out: dict[str, Any] = {"inputs": {}, "variants": {}, "conditions": {}, "checks": []}

    for k, d in VARIANTS.items():
        for fn in ("scores_calib.csv", "scores_test.csv", "scores_holdout.csv", "robustness_scores.csv",
                   "frozen_operating_point.json", "metrics.json", "robustness.json"):
            out["inputs"][f"reports/{d}/{fn}"] = sha256(REP / d / fn)
    for fn in ("comparison.json", "comparison.md"):
        out["inputs"][f"reports/pcb3_illum_comparison/{fn}"] = sha256(REP / "pcb3_illum_comparison" / fn)

    # Pairing: same images, labels and order in both variants
    for role in ("calib", "test", "holdout"):
        same = [(r["image"], r["label"]) for r in V["baseline"][role]["rows"]] == \
               [(r["image"], r["label"]) for r in V["candidate"][role]["rows"]]
        out["checks"].append({"name": f"pairing: identical {role} images/labels/order", "ok": same})
    out["checks"].append({"name": "pairing: identical robustness images/labels/order",
                          "ok": [(r["image"], r["label"]) for r in V["baseline"]["rob_rows"]]
                          == [(r["image"], r["label"]) for r in V["candidate"]["rob_rows"]]})

    for k, v in V.items():
        thr = v["thr"]
        t, h = v["test"], v["holdout"]
        y = t["anom"]
        tfl, hfl = t["score"] > thr, h["score"] > thr
        thr_re, k_re = order_stat_threshold(v["calib"]["score"], v["fz"]["thresholds"][v["primary"]]["target_fpr"])
        rob_base_ok = bool(np.allclose(v["rob"]["base"], np.r_[t["score"], h["score"]], rtol=0, atol=1e-6))
        rec = prop(tfl[y])
        fpr_t = prop(tfl[~y])
        fpr_h = prop(hfl)
        pooled = prop(np.r_[tfl[~y], hfl])
        out["variants"][k] = {
            "threshold_frozen": thr, "threshold_recomputed": thr_re, "rank_k": k_re, "n_calib": int(v["calib"]["score"].size),
            "confusion_official_test": {"TP": rec["x"], "FN": rec["n"] - rec["x"], "FP": fpr_t["x"], "TN": fpr_t["n"] - fpr_t["x"]},
            "recall": rec, "fpr_official_test": fpr_t, "fpr_holdout": fpr_h, "fpr_pooled_unseen_normals": pooled,
            "auroc": auroc(y, t["score"]),
            "roi_fallback": {"test": int(sum(r["roi_fallback"] == "1" for r in t["rows"])),
                             "holdout": int(sum(r["roi_fallback"] == "1" for r in h["rows"]))},
            "robustness_base_equals_clean_scores": rob_base_ok,
        }
        out["checks"].append({"name": f"{k}: threshold = order statistic k={k_re} of {v['calib']['score'].size} calib scores",
                              "ok": abs(thr_re - thr) <= 1e-6})
        out["checks"].append({"name": f"{k}: robustness 'base' column = recorded clean test+holdout scores",
                              "ok": rob_base_ok})

        # Every condition (absolute FPR / recall, then change vs clean, kept separate)
        ra = v["rob_anom"]
        f0 = v["rob"]["base"] > thr
        cond = {}
        for pid, name in ids.items():
            f1 = v["rob"][name] > thr
            fp0, fp1 = int(f0[~ra].sum()), int(f1[~ra].sum())
            tp0, tp1 = int(f0[ra].sum()), int(f1[ra].sum())
            n0, n1 = int((~ra).sum()), int(ra.sum())
            cond[pid] = {"perturbation": name, "in_P5": in_p5[name],
                         "fpr_clean": prop(f0[~ra]), "fpr_perturbed": prop(f1[~ra]),
                         "fpr_change_pts": 100 * (fp1 - fp0) / n0,
                         "recall_clean": prop(f0[ra]), "recall_perturbed": prop(f1[ra]),
                         "recall_change_pts": 100 * (tp1 - tp0) / n1}
        out["variants"][k]["conditions"] = cond

        # Original acceptance criteria (P1-P5 and NO-GO), re-derived
        p, ng = acc["performance"], acc["no_go"]
        p5 = [c for c in cond.values() if c["in_P5"]]
        drop = max(-c["recall_change_pts"] / 100 for c in p5)
        rise = max(c["fpr_change_pts"] / 100 for c in p5)
        m = json.loads((REP / VARIANTS[k] / "metrics.json").read_text(encoding="utf-8"))
        st = {"auroc": out["variants"][k]["auroc"] >= p["auroc_min"], "recall": rec["value"] >= p["recall_min"],
              "fpr_test": fpr_t["value"] <= p["fpr_max"], "fpr_holdout": fpr_h["value"] <= p["fpr_max"],
              "latency": m["latency"]["median_ms"] <= p["latency_ms_max"],
              "robustness": drop <= p["robustness_recall_drop_max"] + TOL and rise <= p["robustness_fpr_rise_max"] + TOL,
              "robustness_max_fpr_rise": rise, "robustness_max_recall_drop": drop,
              "robustness_worst_fpr_condition": max(p5, key=lambda c: c["fpr_change_pts"])["perturbation"]}
        nog = [n for n, hit in (("recall_below_floor", rec["value"] < ng["recall_floor"]),
                                ("fpr_above_ceiling", max(fpr_t["value"], fpr_h["value"]) > ng["fpr_ceiling"]),
                                ("robustness_collapse", drop > ng["robustness_recall_collapse"])) if hit]
        st["no_go_triggers"] = nog
        keys = ("auroc", "recall", "fpr_test", "fpr_holdout", "latency", "robustness")
        st["verdict"] = "NO-GO" if nog else ("GO" if all(st[x] for x in keys) else "INCONCLUSIVE")
        out["variants"][k]["acceptance"] = st

    # Paired analysis (protocol §8)
    b, c = V["baseline"], V["candidate"]
    tb, tc = b["thr"], c["thr"]
    y = b["test"]["anom"]
    clean = {"recall": paired_delta(b["test"]["score"][y] > tb, c["test"]["score"][y] > tc),
             "fpr_official_test": paired_delta(b["test"]["score"][~y] > tb, c["test"]["score"][~y] > tc),
             "fpr_holdout": paired_delta(b["holdout"]["score"] > tb, c["holdout"]["score"] > tc)}
    ra = b["rob_anom"]
    fb0, fc0 = b["rob"]["base"] > tb, c["rob"]["base"] > tc
    for pid, name in ids.items():
        fb1, fc1 = b["rob"][name] > tb, c["rob"][name] > tc
        out["conditions"][pid] = {"perturbation": name,
                                  "fpr_normals": did(fb0[~ra], fb1[~ra], fc0[~ra], fc1[~ra]),
                                  "recall_anomalies": did(fb0[ra], fb1[ra], fc0[ra], fc1[ra]),
                                  "mcnemar_perturbed_normals": mcnemar(fb1[~ra], fc1[~ra])}
    g1 = out["conditions"]["G1"]["fpr_normals"]
    recall_gate = not (clean["recall"]["delta"] < -CLEAN_RECALL_MARGIN - 1e-12
                       or (clean["recall"]["delta"] < 0 and clean["recall"]["p_value"] < ALPHA))
    fpr_gate = (clean["fpr_official_test"]["cand"] <= clean["fpr_official_test"]["base"] + CLEAN_FPR_MARGIN + 1e-12
                and clean["fpr_holdout"]["cand"] <= clean["fpr_holdout"]["base"] + CLEAN_FPR_MARGIN + 1e-12)
    if not (recall_gate and fpr_gate) or g1["ci_low"] > 0:
        outcome = "UNFAVOURABLE TRADE-OFF"
    elif g1["ci_high"] < 0:
        outcome = "ROBUSTNESS IMPROVEMENT SUPPORTED"
    else:
        outcome = "INCONCLUSIVE"
    out["clean_paired"] = clean
    out["decision"] = {"outcome": outcome, "clean_recall_gate_pass": recall_gate, "clean_fpr_gate_pass": fpr_gate,
                       "baseline_g1_rise_pts": 100 * g1["delta_base"],
                       "weakly_informative": g1["delta_base"] < WEAK_RISE,
                       "outcome_robust_to_bootstrap_seed": all(x < 0 for x in g1["alt_seed_ci_high_range"])}
    return out


RING_PX = 12  # protocol §8 C: ring 1-12 px outside the mask


def recompute_subgroups(out: dict[str, Any]) -> None:
    """Question C terciles (the only output that passes through the repaired function, `defect_descriptors`).

    Re-derived from the official VisA ground-truth masks with this file's own code. Needs the PCB3 images
    and masks under data/visa/pcb3 (not in the repository); otherwise recorded as SKIPPED, never as passed.
    """
    from PIL import Image
    from scipy import ndimage

    vis = ROOT / "data" / "visa"
    anno_p = vis / "pcb3" / "image_anno.csv"
    if not anno_p.is_file():
        out["subgroups"] = {"status": "SKIPPED", "reason": "VisA PCB3 images/masks not present (data/ is never committed)"}
        out["checks"].append({"name": "question C subgroup terciles from GT masks", "ok": None,
                              "skipped": out["subgroups"]["reason"]})
        return
    anno = {r["image"]: r for r in read_csv(anno_p)}
    b = load_variant(REP / VARIANTS["baseline"])
    c = load_variant(REP / VARIANTS["candidate"])
    y = b["test"]["anom"]
    imgs = [r["image"] for r, a in zip(b["test"]["rows"], y) if a]
    area, con = [], []
    for rel in imgs:
        a = anno[rel]
        with Image.open(vis / a["mask"]) as im:
            mk = np.asarray(im) > 0
        if mk.ndim == 3:
            mk = mk.any(axis=2)
        with Image.open(vis / rel) as im:
            g = np.asarray(im.convert("RGB").convert("L"), np.float64)
        ring = (ndimage.distance_transform_edt(~mk) > 0) & (ndimage.distance_transform_edt(~mk) <= RING_PX)
        area.append(float(mk.mean()))
        con.append(abs(float(g[mk].mean()) - float(g[ring].mean())))
    fb, fc = b["test"]["score"][y] > b["thr"], c["test"]["score"][y] > c["thr"]

    def terciles(vals):
        order = sorted(range(len(vals)), key=lambda i: (vals[i], imgs[i]))
        lab = np.empty(len(vals), int)
        for gi, idx in enumerate(np.array_split(np.array(order), 3)):
            lab[idx] = gi
        return lab

    res = {}
    for key, vals, names in (("size_clean", area, ("small", "medium", "large")),
                             ("contrast_clean", con, ("low", "medium", "high"))):
        lab = terciles(vals)
        res[key] = {n: {"n": int((lab == gi).sum()), "detected_base": int(fb[lab == gi].sum()),
                        "detected_cand": int(fc[lab == gi].sum())} for gi, n in enumerate(names)}
    res["labels_seen"] = sorted({anno[r]["label"] for r in imgs})
    res["status"] = "computed"
    out["subgroups"] = res
    rec = json.loads((REP / "pcb3_illum_comparison" / "comparison.json").read_text(encoding="utf-8"))["subgroups"]
    same = all(res[k][n][f] == rec[k][n][f] for k in ("size_clean", "contrast_clean") for n in res[k]
               for f in ("n", "detected_base", "detected_cand"))
    out["checks"].append({"name": "question C subgroup terciles from GT masks equal the recorded table", "ok": same})


# --------------------------------------------------------------------------- comparison with recorded outputs
def compare(out: dict[str, Any]) -> None:
    chk = out["checks"]

    def eq(name, got, want, tol=TOL):
        ok = (got == want) if isinstance(want, (str, bool)) else abs(float(got) - float(want)) <= tol
        chk.append({"name": name, "ok": bool(ok), "recomputed": got, "recorded": want})

    cmpj = json.loads((REP / "pcb3_illum_comparison" / "comparison.json").read_text(encoding="utf-8"))
    for k, d in VARIANTS.items():
        m = json.loads((REP / d / "metrics.json").read_text(encoding="utf-8"))
        rb = json.loads((REP / d / "robustness.json").read_text(encoding="utf-8"))
        v = out["variants"][k]
        op = m["official_test"]["operating_points"]["P95"]
        for kk in ("TP", "FN", "FP", "TN"):
            eq(f"{k}: official-test {kk}", v["confusion_official_test"][kk], op["confusion"][kk])
        eq(f"{k}: recall CI low", v["recall"]["ci_low"], op["recall"]["ci_low"], 1e-9)
        eq(f"{k}: recall CI high", v["recall"]["ci_high"], op["recall"]["ci_high"], 1e-9)
        eq(f"{k}: holdout FP", v["fpr_holdout"]["x"], m["supplementary_holdout"]["operating_points"]["P95"]["fpr"]["x"])
        eq(f"{k}: AUROC", v["auroc"], m["official_test"]["auroc"]["value"], 1e-12)
        for pid, cnd in v["conditions"].items():
            r = rb["perturbations"][cnd["perturbation"]]
            eq(f"{k}: {pid} FP (pooled normals)", cnd["fpr_perturbed"]["x"], r["fpr"]["x"])
            eq(f"{k}: {pid} FPR change", cnd["fpr_change_pts"] / 100, r["paired"]["delta_fpr"]["value"], 1e-12)
            eq(f"{k}: {pid} recall change", cnd["recall_change_pts"] / 100, r["paired"]["delta_recall"]["value"], 1e-12)
        a = cmpj["acceptance"]["base" if k == "baseline" else "cand"]
        eq(f"{k}: P5 verdict", v["acceptance"]["verdict"], a["verdict"])
        eq(f"{k}: P5 robustness criterion", v["acceptance"]["robustness"], a["robustness"])
        eq(f"{k}: P5 max FPR rise", v["acceptance"]["robustness_max_fpr_rise"], a["robustness_max_fpr_rise"], 1e-12)
    for pid, cnd in out["conditions"].items():
        if pid not in cmpj["conditions"]:
            continue
        r = cmpj["conditions"][pid]["fpr_normals"]
        for kk in ("delta_base", "delta_cand", "did"):
            eq(f"paired {pid}: {kk}", cnd["fpr_normals"][kk], r[kk], 1e-12)
        eq(f"paired {pid}: DiD CI low (B={B}, seed {SEED})", cnd["fpr_normals"]["ci_low"], r["ci_low"], 1e-12)
        eq(f"paired {pid}: DiD CI high (B={B}, seed {SEED})", cnd["fpr_normals"]["ci_high"], r["ci_high"], 1e-12)
        mc = cmpj["conditions"][pid]["perturbed_flags_cand_vs_base_normals"]
        eq(f"paired {pid}: McNemar p (perturbed normals)", cnd["mcnemar_perturbed_normals"]["p_value"], mc["p_value"], 1e-9)
    for kk, rk in (("recall", "recall_test_anomalies"), ("fpr_official_test", "fpr_test_normals"),
                   ("fpr_holdout", "fpr_holdout_normals")):
        eq(f"clean paired {kk}: delta", out["clean_paired"][kk]["delta"], cmpj["clean"][rk]["delta"], 1e-12)
        eq(f"clean paired {kk}: McNemar p", out["clean_paired"][kk]["p_value"], cmpj["clean"][rk]["p_value"], 1e-9)
    eq("pre-registered outcome", out["decision"]["outcome"], cmpj["decision"]["outcome"])
    eq("weakly-informative flag", out["decision"]["weakly_informative"], cmpj["decision"]["baseline_rise_weakly_informative"])
    eq("clean recall gate", out["decision"]["clean_recall_gate_pass"], cmpj["decision"]["clean_recall_gate_pass"])
    eq("clean FPR gate", out["decision"]["clean_fpr_gate_pass"], all(cmpj["decision"]["clean_fpr_gate_pass"].values()))
    chk.append({"name": "G1 DiD CI upper < 0 under 10 alternative bootstrap seeds",
                "ok": out["decision"]["outcome_robust_to_bootstrap_seed"]})


# --------------------------------------------------------------------------- report
def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def pts(x: float, nd: int = 1) -> str:
    return f"{100 * x:+.{nd}f}"


def write_md(o: dict[str, Any], path: Path) -> None:
    L = ["# PCB3 independent recomputation from recorded per-image scores", "",
         "Generated by `python verification/recompute_pcb3.py` (imports nothing from `src/ivad`). Inputs are the",
         "raw score CSVs of the single planned evaluation (`115c54b`) and the thresholds frozen at `eb4e7a9`.",
         "Nothing was re-scored, re-fitted or re-calibrated.", "",
         f"**Checks: {sum(c['ok'] is True for c in o['checks'])}/{len(o['checks'])} agree with the recorded outputs; "
         f"{sum(c['ok'] is False for c in o['checks'])} disagree; {sum(c['ok'] is None for c in o['checks'])} SKIPPED.**", "",
         "## Clean condition, each variant at its own frozen P95 threshold", "",
         "| Quantity | Baseline | Candidate |", "|---|---|---|"]
    vb, vc = o["variants"]["baseline"], o["variants"]["candidate"]
    L.append(f"| Frozen threshold (rank k of n calib) | {vb['threshold_frozen']:.4f} (k={vb['rank_k']}/{vb['n_calib']}) "
             f"| {vc['threshold_frozen']:.4f} (k={vc['rank_k']}/{vc['n_calib']}) |")
    for lab, key in (("Recall, official-test anomalies", "recall"), ("FPR, official-test normals", "fpr_official_test"),
                     ("FPR, supplementary holdout normals", "fpr_holdout"),
                     ("FPR, pooled unseen normals (robustness reference)", "fpr_pooled_unseen_normals")):
        cell = [f"{pct(v[key]['value'])} ({v[key]['x']}/{v[key]['n']}) [{pct(v[key]['ci_low'])}, {pct(v[key]['ci_high'])}]"
                for v in (vb, vc)]
        L.append(f"| {lab} | {cell[0]} | {cell[1]} |")
    L.append(f"| TP / FN / FP / TN (official test) | {' / '.join(str(vb['confusion_official_test'][k]) for k in ('TP','FN','FP','TN'))} "
             f"| {' / '.join(str(vc['confusion_official_test'][k]) for k in ('TP','FN','FP','TN'))} |")
    L.append(f"| AUROC (official test) | {vb['auroc']:.4f} | {vc['auroc']:.4f} |")
    L += ["", "Intervals: Clopper-Pearson exact 95%. The official-test split (100 anomalies + 101 normals) is the",
          "benchmark result; the 155 holdout normals are a supplementary check; the 256 pooled normals (101 + 155)",
          "are the population of every robustness endpoint.", "",
          "## Every condition: absolute rates and change from clean (kept separate)", "",
          "| Condition | In P5 | FPR base (x/256) | FPR cand (x/256) | Change base (pts) | Change cand (pts) | Recall base (x/100) | Recall cand (x/100) |",
          "|---|---|---|---|---|---|---|---|"]
    for pid in ("G1", "B1", "N1", "blur", "shift"):
        cb, cc = vb["conditions"][pid], vc["conditions"][pid]
        L.append(f"| {pid} `{cb['perturbation']}` | {'yes' if cb['in_P5'] else 'no'} "
                 f"| {pct(cb['fpr_perturbed']['value'])} ({cb['fpr_perturbed']['x']}) "
                 f"| {pct(cc['fpr_perturbed']['value'])} ({cc['fpr_perturbed']['x']}) "
                 f"| {cb['fpr_change_pts']:+.1f} | {cc['fpr_change_pts']:+.1f} "
                 f"| {cb['recall_perturbed']['x']} | {cc['recall_perturbed']['x']} |")
    L.append(f"| clean reference | — | {pct(vb['fpr_pooled_unseen_normals']['value'])} ({vb['fpr_pooled_unseen_normals']['x']}) "
             f"| {pct(vc['fpr_pooled_unseen_normals']['value'])} ({vc['fpr_pooled_unseen_normals']['x']}) | — | — "
             f"| {vb['recall']['x']} | {vc['recall']['x']} |")
    g1 = o["conditions"]["G1"]["fpr_normals"]
    d = o["decision"]
    L += ["", "## Primary endpoint and pre-registered rule", "",
          f"- G1 false-alarm rise: baseline {pts(g1['delta_base'])} pts ({g1['counts']['base_clean']} → {g1['counts']['base_pert']} of {g1['n']}), "
          f"candidate {pts(g1['delta_cand'])} pts ({g1['counts']['cand_clean']} → {g1['counts']['cand_pert']} of {g1['n']}).",
          f"- DiD {pts(g1['did'])} pts, 95% CI [{pts(g1['ci_low'])}, {pts(g1['ci_high'])}] (B = {B}, seed {SEED}); "
          f"upper bound under ten other seeds: {pts(g1['alt_seed_ci_high_range'][0])} to {pts(g1['alt_seed_ci_high_range'][1])}.",
          f"- Clean recall gate: {'pass' if d['clean_recall_gate_pass'] else 'FAIL'}; clean FPR gate: {'pass' if d['clean_fpr_gate_pass'] else 'FAIL'}.",
          f"- **Rule outcome: {d['outcome']}**; baseline G1 rise {d['baseline_g1_rise_pts']:+.1f} pts < 5 pts → "
          f"weakly informative: {d['weakly_informative']}.", "",
          "## Original acceptance criteria (P1–P5, NO-GO), re-derived", "",
          "| Criterion | Baseline | Candidate |", "|---|---|---|"]
    ab, ac = vb["acceptance"], vc["acceptance"]
    for k in ("recall", "fpr_test", "fpr_holdout", "auroc", "latency", "robustness"):
        L.append(f"| {k} | {'met' if ab[k] else 'NOT met'} | {'met' if ac[k] else 'NOT met'} |")
    L.append(f"| P5 worst FPR rise (condition) | {100 * ab['robustness_max_fpr_rise']:+.1f} pts ({ab['robustness_worst_fpr_condition']}) "
             f"| {100 * ac['robustness_max_fpr_rise']:+.1f} pts ({ac['robustness_worst_fpr_condition']}) |")
    L.append(f"| NO-GO triggers | {ab['no_go_triggers'] or 'none'} | {ac['no_go_triggers'] or 'none'} |")
    L.append(f"| **Verdict** | **{ab['verdict']}** | **{ac['verdict']}** |")
    L += ["", "## Input file hashes (SHA-256)", "", "| File | SHA-256 |", "|---|---|"]
    L += [f"| `{k}` | `{v}` |" for k, v in o["inputs"].items()]
    L += ["", "## All checks", "", "| Check | Result |", "|---|---|"]
    L += [f"| {c['name']} | {'PASS' if c['ok'] else ('SKIPPED: ' + c['skipped'] if c['ok'] is None else '**FAIL**')} |"
          for c in o["checks"]]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 on any disagreement; write nothing")
    a = ap.parse_args()
    o = recompute()
    compare(o)
    recompute_subgroups(o)
    bad = [c for c in o["checks"] if c["ok"] is False]
    skipped = [c for c in o["checks"] if c["ok"] is None]
    print(f"PCB3 independent recomputation: {sum(c['ok'] is True for c in o['checks'])}/{len(o['checks'])} checks agree, "
          f"{len(bad)} disagree, {len(skipped)} SKIPPED; "
          f"outcome {o['decision']['outcome']}; verdicts baseline {o['variants']['baseline']['acceptance']['verdict']}, "
          f"candidate {o['variants']['candidate']['acceptance']['verdict']}")
    for c in bad:
        print("  FAIL:", c)
    if not a.check:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "recompute.json").write_text(json.dumps(o, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
        write_md(o, OUT / "recompute.md")
        print(f"-> {OUT.relative_to(ROOT).as_posix()}/recompute.{{json,md}}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
