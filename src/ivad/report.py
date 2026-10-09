"""Figures, post-hoc diagnostics, sample outputs and reports/results.md.

Everything here READS frozen outputs (metrics.json, robustness.json, score
CSVs). Nothing here can change the model or the threshold. Diagnostics that use
test labels (per-defect-type recall, defect size) are explicitly post hoc.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from . import cost
from .align import foreground_mask, warp
from .config import resolve_path, write_json
from .features import center_crop, load_rgb
from .pipeline import check_freeze, load_detector, pmap, read_scores, score_details, score_images

# Restrained palette: one hue per series, consistent across figures.
C_NORMAL, C_ANOM, C_HOLD, C_CAL = "#4C72B0", "#C44E52", "#8172B2", "#999999"
_C_THR = {"P90": "#55A868", "P95": "#222222", "P99": "#CCB974"}


class _ThrColors(dict):
    def __missing__(self, key):  # thresholds other than P90/P95/P99 (e.g. test configs)
        return "#777777"


C_THR = _ThrColors(_C_THR)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def _load(cfg):
    rep = resolve_path(cfg, "reports")
    m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    fz = json.loads((rep / "frozen_operating_point.json").read_text(encoding="utf-8"))
    rb_path = rep / "robustness.json"
    rb = json.loads(rb_path.read_text(encoding="utf-8")) if rb_path.exists() else None
    return rep, m, fz, rb


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_scores(cfg, rep: Path, fz, figdir: Path) -> Path:
    _, cal = read_scores(rep / "scores_calib.csv")
    trows, ts = read_scores(rep / "scores_test.csv")
    _, hs = read_scores(rep / "scores_holdout.csv")
    y = np.array([r["label"] == "anomaly" for r in trows])
    groups = [(f"Calibration normals (n={len(cal)})", cal, C_CAL), (f"Holdout normals (n={len(hs)})", hs, C_HOLD),
              (f"Test normals (n={int((~y).sum())})", ts[~y], C_NORMAL),
              (f"Test anomalies (n={int(y.sum())})", ts[y], C_ANOM)]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    rng = np.random.default_rng(0)
    for i, (name, s, c) in enumerate(groups):
        ax.scatter(s, i + rng.uniform(-0.18, 0.18, len(s)), s=7, color=c, alpha=0.7, linewidths=0)
    for name, t in fz["thresholds"].items():
        ax.axvline(t["threshold"], color=C_THR[name], lw=1.2 if name == fz["primary"] else 0.8,
                   ls="-" if name == fz["primary"] else "--")
        ax.text(t["threshold"], len(groups) - 0.45, f" {name}{' (primary)' if name == fz['primary'] else ''}",
                fontsize=7, color=C_THR[name], va="bottom")
    ax.set_xscale("log")
    ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlabel("Image anomaly score (max Mahalanobis distance, log scale)")
    ax.set_ylim(-0.6, len(groups) - 0.1)
    out = figdir / "fig_score_distributions.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_roc(cfg, rep: Path, m, fz, figdir: Path) -> Path:
    from .metrics import roc_curve
    trows, ts = read_scores(rep / "scores_test.csv")
    y = np.array([r["label"] == "anomaly" for r in trows])
    fpr, tpr = roc_curve(y, ts)
    a = m["official_test"]["auroc"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(fpr, tpr, color=C_NORMAL, lw=1.5)
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls=":")
    for name, op in m["official_test"]["operating_points"].items():
        ax.plot(op["fpr"]["value"], op["recall"]["value"], "o", color=C_THR[name], ms=5)
        ax.annotate(name, (op["fpr"]["value"], op["recall"]["value"]), textcoords="offset points",
                    xytext=(6, -10), fontsize=7)
    ax.set_xlabel("False-positive rate (test normals)")
    ax.set_ylabel("Recall (test anomalies)")
    ax.set_title(f"Official test ROC, AUROC {a['value']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]", fontsize=8)
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
    out = figdir / "fig_roc_official_test.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def cost_table(m) -> list[dict[str, Any]]:
    rows = []
    for name, op in m["official_test"]["operating_points"].items():
        r, f = op["recall"]["value"], op["fpr"]["value"]
        be = cost.breakeven_k(r, f)
        rows.append({"threshold": name, "recall": r, "fpr": f, **be})
    return rows


def fig_cost(cfg, m, fz, figdir: Path) -> Path:
    k = np.logspace(np.log10(cfg["cost"]["k_min"]), np.log10(cfg["cost"]["k_max"]), 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(k, k, color="#BBBBBB", lw=1, ls=":", label="Review none  (C = k)")
    ax.plot(k, np.ones_like(k), color="#888888", lw=1, ls="--", label="Review all  (C = 1)")
    for name, op in m["official_test"]["operating_points"].items():
        r, f = op["recall"]["value"], op["fpr"]["value"]
        ax.plot(k, cost.normalized_cost(k, r, f), color=C_THR[name], lw=2 if name == fz["primary"] else 1.2,
                label=f"{name}{' (primary)' if name == fz['primary'] else ''}: R={r:.2f}, FPR={f:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k = (prevalence / (1 - prevalence)) x (C_FN / C_FP)")
    ax.set_ylabel("Normalised expected cost  C = k(1-R) + FPR")
    ax.set_ylim(min(k.min(), 0.005), 30)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Cost sensitivity (official-test Recall/FPR at frozen thresholds)", fontsize=8)
    out = figdir / "fig_cost_sensitivity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Post-hoc diagnostics (use test labels; descriptive only)
# --------------------------------------------------------------------------- #
def defect_diagnostics(cfg, rep: Path, fz) -> dict[str, Any]:
    root = resolve_path(cfg, "data_dir")
    obj = cfg["dataset"]["object"]
    anno_path = root / obj / "image_anno.csv"
    trows, ts = read_scores(rep / "scores_test.csv")
    t = fz["thresholds"][fz["primary"]]["threshold"]
    out: dict[str, Any] = {"note": "POST HOC: uses official-test labels and masks for description only; "
                                   "nothing here was used to choose the model or threshold."}
    if not anno_path.exists():
        return out
    anno = {r["image"]: r for r in csv.DictReader(open(anno_path, encoding="utf-8"))}
    recs = []
    for r, s in zip(trows, ts):
        if r["label"] != "anomaly":
            continue
        a = anno.get(r["image"], {})
        types = a.get("label", "unknown").split(",")
        area = None
        if a.get("mask"):
            mk = np.asarray(Image.open(root / a["mask"]))
            area = float((mk > 0).mean())
        recs.append({"image": r["image"], "types": types, "score": s, "detected": bool(s > t),
                     "mask_area_frac": area})
    by_type = {}
    for ty in sorted({x for r in recs for x in r["types"]}):
        sel = [r for r in recs if ty in r["types"]]
        by_type[ty] = {"n": len(sel), "detected": sum(r["detected"] for r in sel),
                       "recall": sum(r["detected"] for r in sel) / len(sel)}
    areas = np.array([r["mask_area_frac"] for r in recs if r["mask_area_frac"] is not None])
    by_size = {}
    if len(areas) == len(recs):
        edges = np.quantile(areas, [0, 1 / 3, 2 / 3, 1])
        for i, lab in enumerate(["small", "medium", "large"]):
            sel = [r for r in recs if edges[i] <= r["mask_area_frac"] <= edges[i + 1]
                   and (i == 0 or r["mask_area_frac"] > edges[i])]
            by_size[lab] = {"n": len(sel), "detected": sum(r["detected"] for r in sel),
                            "recall": sum(r["detected"] for r in sel) / max(len(sel), 1),
                            "mask_area_pct_range": [100 * float(edges[i]), 100 * float(edges[i + 1])]}
    out.update({"by_type_multilabel": by_type, "by_mask_area_tercile": by_size,
                "missed": sorted([{"image": r["image"], "types": r["types"], "score": round(r["score"], 2),
                                   "mask_area_pct": None if r["mask_area_frac"] is None
                                   else round(100 * r["mask_area_frac"], 3)}
                                  for r in recs if not r["detected"]], key=lambda d: d["score"])})
    return out


def decision_maps(det, maps: np.ndarray, info: dict[str, np.ndarray]) -> np.ndarray:
    """Anomaly maps restricted to the region the image score is taken from.

    v1: the whole frame. v2: the ROI, except for images where the fail-safe fell back
    to the full frame. The argmax of a decision map is the pixel that set the score.
    """
    if det.roi is None:
        return maps
    roi = det.roi.cpu().numpy()
    fb = info["roi_fallback"].astype(bool)
    return np.where(roi[None] | fb[:, None, None], maps.astype(np.float32), -np.inf)


def _gt_mask_view(cfg, det, image_rel: str, mask_rel: str) -> np.ndarray:
    """Ground-truth defect mask mapped into the model frame (same registration + crop)."""
    root = resolve_path(cfg, "data_dir")
    _, info = det.pre.registrar.register(load_rgb(root / image_rel))
    mk = Image.open(root / mask_rel).convert("L").point(lambda v: 255 if v > 0 else 0)
    mk = center_crop(warp(mk, info, info["flipped"], fill=0), det.pre.crop_wh)
    h, w = det.pre.hw
    return np.asarray(mk.resize((w, h), Image.NEAREST)) > 0


def peak_diagnostics(cfg, det, rep: Path, fz) -> dict[str, Any]:
    """Where is the anomaly-map maximum? (POST HOC, uses test labels/masks.)

    * anomalies: is the peak inside the defect mask (dilated by `tol` px)?
    * flagged normals (false alarms, test + holdout): is the peak on the board
      foreground or on the background cloth?
    """
    root = resolve_path(cfg, "data_dir")
    obj = cfg["dataset"]["object"]
    anno = {r["image"]: r for r in csv.DictReader(open(root / obj / "image_anno.csv", encoding="utf-8"))}
    t = fz["thresholds"][fz["primary"]]["threshold"]
    tol = 16  # model-frame px (= 24 native px)
    trows, ts = read_scores(rep / "scores_test.csv")
    hrows, hs = read_scores(rep / "scores_holdout.csv")
    an = [r for r in trows if r["label"] == "anomaly"]
    _, amaps, ainfo = score_details(det, [root / r["image"] for r in an], keep_maps=True)
    amaps = decision_maps(det, amaps, ainfo)
    inside = {"detected": [0, 0], "missed": [0, 0]}
    for r, m in zip(an, amaps):
        gt = ndimage.binary_dilation(_gt_mask_view(cfg, det, r["image"], anno[r["image"]]["mask"]), iterations=tol)
        py, px = np.unravel_index(np.argmax(m), m.shape)
        key = "detected" if float(m.max()) > t else "missed"
        inside[key][0] += int(gt[py, px])
        inside[key][1] += 1
    fps = [r for r, s in zip(trows + hrows, np.r_[ts, hs]) if r["label"] == "normal" and s > t]
    on_board = 0
    for r in fps:
        img = load_rgb(root / r["image"])
        _, m, info = score_details(det, [root / r["image"]], keep_maps=True)
        m = decision_maps(det, m, info)
        view = det.pre.view(img)
        fg = foreground_mask(view)  # 1/4 scale of the crop
        fg = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize(m[0].shape[::-1], Image.NEAREST)) > 0
        py, px = np.unravel_index(np.argmax(m[0]), m[0].shape)
        on_board += int(fg[py, px])
    return {"note": "POST HOC description; tolerance = 16 model px (24 native px) around the GT mask; "
                    "peak = location that set the image score (inside the ROI for v2)",
            "anomaly_peak_in_defect_mask": {k: {"in_mask": v[0], "n": v[1]} for k, v in inside.items()},
            "false_alarm_peak_location": {"n": len(fps), "on_board": on_board, "on_background": len(fps) - on_board,
                                          "sources": "official test normals + supplementary holdout"}}


def registration_qc(cfg, det) -> tuple[dict[str, Any], dict[str, Any]]:
    """Orientation-decision margin on every scored image (no labels), plus a POST-HOC
    orientation-confound check that uses labels only to describe the benchmark."""
    from scipy import stats as sstats
    from .metrics import auroc, bootstrap_auroc

    rep = resolve_path(cfg, "reports")
    root = resolve_path(cfg, "data_dir")
    res, per = {}, {}
    for role in ("test", "holdout", "calib"):
        rows, sc = read_scores(rep / f"scores_{role}.csv")
        info = list(pmap(cfg)(lambda r: det.pre.registrar.register(load_rgb(root / r["image"]))[1], rows))
        margin = np.array([i["ncc"] - i["ncc_other"] for i in info])
        per[role] = (rows, sc, np.array([i["flipped"] for i in info]))
        res[role] = {"n": len(info), "flipped": int(per[role][2].sum()),
                     "min_orientation_margin": float(margin.min()),
                     "lowest_margin_image": rows[int(np.argmin(margin))]["image"],
                     "n_margin_below_0.1": int((margin < 0.1).sum()),
                     "tilt_deg_range": [float(min(i["angle"] for i in info)), float(max(i["angle"] for i in info))]}
    rows, sc, fl = per["test"]
    y = np.array([r["label"] == "anomaly" for r in rows])
    ns = np.r_[per["calib"][1], per["holdout"][1], sc[~y]]
    nf = np.r_[per["calib"][2], per["holdout"][2], fl[~y]]
    if not (nf.any() and (~nf).any() and (y & fl).any() and (~y & fl).any()):
        return res, {}  # too few rotated images of some kind to compare
    conf = {
        "note": "POST HOC: rotation rate by label, and whether orientation shifts NORMAL scores",
        "test_anomaly_rotated": [int(fl[y].sum()), int(y.sum())],
        "test_normal_rotated": [int(fl[~y].sum()), int((~y).sum())],
        "normals_rotated_vs_not_mannwhitney_p": float(sstats.mannwhitneyu(ns[nf], ns[~nf]).pvalue),
        "normals_orientation_auroc": auroc(nf, ns),
        "auroc_within_rotated_test_subset": bootstrap_auroc(y[fl], sc[fl], cfg["evaluation"]["n_bootstrap"],
                                                            cfg["evaluation"]["bootstrap_seed"]),
    }
    return res, conf


# --------------------------------------------------------------------------- #
# Sample outputs
# --------------------------------------------------------------------------- #
def _overlay(ax, view: Image.Image, amap: np.ndarray, vmin: float, vmax: float, title: str,
             gt: np.ndarray | None = None, roi: np.ndarray | None = None):
    """Image with the anomaly map shown only where it exceeds vmin (normal-level scores stay clear)."""
    ax.imshow(view.resize((amap.shape[1], amap.shape[0]), Image.BILINEAR))
    ax.imshow(np.ma.masked_less(amap, vmin), cmap="inferno", alpha=0.6, vmin=vmin, vmax=vmax)
    if gt is not None and gt.any():
        ax.contour(gt, levels=[0.5], colors="#00E5FF", linewidths=0.8)
    if roi is not None:
        ax.contour(roi.astype(float), levels=[0.5], colors="#7CFC00", linewidths=0.6, linestyles="dashed")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def sample_outputs(cfg, det, rep: Path, fz, figdir: Path) -> Path:
    """Two examples each of TP / FN / FP / TN from the official test set."""
    root = resolve_path(cfg, "data_dir")
    trows, ts = read_scores(rep / "scores_test.csv")
    t = fz["thresholds"][fz["primary"]]["threshold"]
    y = np.array([r["label"] == "anomaly" for r in trows])
    f = ts > t
    cats = {"TP": y & f, "FN": y & ~f, "FP": ~y & f, "TN": ~y & ~f}
    picks = []
    for name, mask in cats.items():
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        order = idx[np.argsort(ts[idx])]
        # FN: the two lowest-scoring misses; FP: two highest false alarms; else median-ish
        chosen = order[:2] if name == "FN" else order[-2:] if name == "FP" else order[len(order) // 2:][:2]
        picks += [(name, int(i)) for i in chosen]
    paths = [root / trows[i]["image"] for _, i in picks]
    scores, maps = score_images(det, paths, keep_maps=True)
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.4))
    root_anno = resolve_path(cfg, "data_dir") / cfg["dataset"]["object"] / "image_anno.csv"
    anno = {r["image"]: r for r in csv.DictReader(open(root_anno, encoding="utf-8"))} if root_anno.exists() else {}
    vmin = fz["calib_score_summary"]["median"]
    roi = det.roi.cpu().numpy() if det.roi is not None else None
    for ax, (name, i), s, amap, p in zip(axes.T.ravel(), picks, scores, maps, paths):
        view = det.pre.view(load_rgb(p))
        rel = trows[i]["image"]
        gt = _gt_mask_view(cfg, det, rel, anno[rel]["mask"]) if anno.get(rel, {}).get("mask") else None
        _overlay(ax, view, amap.astype(np.float32), vmin=vmin, vmax=max(2 * t, vmin + 1), gt=gt, roi=roi,
                 title=f"{name} | {Path(trows[i]['image']).parent.name}/{Path(p).name}\n"
                       f"score {s:.1f} -> {'REVIEW' if s > t else 'no review'}")
    fig.suptitle(f"Official-test examples at the frozen {fz['primary']} threshold ({t:.1f}). Heat map: smoothed "
                 f"Mahalanobis distance where above the median calibration score ({vmin:.0f}); "
                 "cyan outline: ground-truth defect mask (shown for review only)"
                 + ("; dashed green: scoring ROI" if roi is not None else ""), fontsize=8)
    fig.tight_layout()
    out = figdir / "fig_sample_outputs.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    with open(rep / "sample_predictions.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv, lineterminator="\n")
        w.writerow(["category", "image", "label", "score", "threshold", "recommendation"])
        for (name, i), s in zip(picks, scores):
            w.writerow([name, trows[i]["image"], trows[i]["label"], f"{s:.4f}", f"{t:.4f}",
                        "REVIEW" if s > t else "NO_REVIEW"])
    return out


def predict_one(cfg, image: str, out: str | None = None) -> dict[str, Any]:
    fz = check_freeze(cfg)
    det = load_detector(cfg)
    t = fz["thresholds"][fz["primary"]]["threshold"]
    img = load_rgb(image)
    s, amap, extra = det.score_one(img)
    res = {"image": str(image), "score": s, "threshold": t, "threshold_name": fz["primary"],
           "recommendation": "REVIEW" if s > t else "NO_REVIEW", **extra,
           "note": "Decision support only; a human makes the disposition."}
    if out:
        fig, ax = plt.subplots(figsize=(6, 4))
        _overlay(ax, det.pre.view(img), amap.cpu().numpy(), fz["calib_score_summary"]["median"], max(2 * t, 1),
                 f"score {s:.1f} vs threshold {t:.1f} -> {res['recommendation']}",
                 roi=None if det.roi is None else det.roi.cpu().numpy())
        fig.savefig(out, dpi=150)
        plt.close(fig)
        res["heatmap"] = str(out)
    return res


# --------------------------------------------------------------------------- #
# results.md
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{100 * x:.1f}%"


def _rate(r):
    return f"{_pct(r['value'])} ({r['x']}/{r['n']}) [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]"


def write_results_md(cfg, rep: Path, m, fz, rb, diag, qc, ct) -> Path:
    o, h, pooled = m["official_test"], m["supplementary_holdout"], m["pooled_unseen_normals"]
    a = o["auroc"]
    obj = cfg["dataset"]["object"].upper()
    agg = fz.get("aggregation", "full_max")
    L = [f"# Results — {cfg.get('run_name', obj)} (generated by `ivad report`; do not edit by hand)", "",
         f"Image score: **{'max over the board ROI (v2)' if agg == 'roi_max' else 'max over the full frame (v1)'}**. "
         f"Frozen operating point: **{fz['primary']}**, frozen at {fz['frozen_at_utc']} UTC "
         f"(config `{fz['config_hash'][:12]}`, model `{fz['model_sha256'][:12]}`).", "",
         f"## 1. Official VisA {obj} test ({o['n_normal']} normal + {o['n_anomaly']} anomaly)", "",
         f"Image AUROC: **{a['value']:.3f}** [{a['ci_low']:.3f}, {a['ci_high']:.3f}] ({a['ci_method']}).", "",
         "| Threshold | Score cut | Recall [95% CI] | FPR [95% CI] | TP | FN | FP | TN |",
         "|---|---|---|---|---|---|---|---|"]
    for name, op in o["operating_points"].items():
        c = op["confusion"]
        tag = " (primary, pre-registered)" if name == fz["primary"] else " (secondary)"
        L.append(f"| {name}{tag} | {op['threshold']:.2f} | {_rate(op['recall'])} | {_rate(op['fpr'])} | "
                 f"{c['TP']} | {c['FN']} | {c['FP']} | {c['TN']} |")
    L += ["", "CIs: Clopper-Pearson exact for Recall/FPR.", "",
          f"## 2. Supplementary normal holdout ({h['n_normal']} normals; not part of the official protocol)", "",
          "| Threshold | Calibration target FPR | Expected FPR from calibration (95% range) | Measured holdout FPR [95% CI] |",
          "|---|---|---|---|"]
    for name, op in h["operating_points"].items():
        cal = fz["thresholds"][name]
        d = cal["fpr_due_to_calibration_sampling"]
        L.append(f"| {name} | {_pct(cal['target_fpr'])} | {_pct(d['expected'])} ({_pct(d['lower'])}-{_pct(d['upper'])}) "
                 f"| {_rate(op['fpr'])} |")
    L += ["", f"Pooled descriptive FPR over all {pooled['n_normal']} unseen normals (official test + holdout; "
              "supplemental, NOT the benchmark result): " +
          ", ".join(f"{n} {_rate(op['fpr'])}" for n, op in pooled["operating_points"].items()), ""]
    if fz.get("roi"):
        q, cf = fz["roi"]["fit_qc"], fz["roi"]["calib_fallback"]
        fb = m.get("roi_fallback", {})
        L += ["### Board ROI (built from FIT normals only)", "",
              f"ROI area {_pct(q['area_frac'])} of the model frame; FIT images with >= "
              f"{_pct(q['rules']['coverage_min'])} board coverage: {_pct(q['fit_pass_frac'])} "
              f"(min coverage {q['fit_coverage_min']:.3f}). Fail-safe fallbacks to the full frame: "
              f"calib {cf['n_fallback']}/{cf['n']}"
              + "".join(f", {k} {v['n_fallback']}/{v['n']}" for k, v in fb.items() if v) + ".", ""]
    lat = m["latency"]
    L += ["## 3. Latency", "",
          f"Median {lat['median_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms per image on {lat['device_name']} "
          f"({lat['definition']}; n={lat['n_images']}).", ""]
    if rb:
        L += [f"## 4. Robustness at the frozen threshold (paired, {rb['images']['anomaly']} anomalies + "
              f"{rb['images']['normal']} normals)", "",
              f"Baseline: Recall {_pct(rb['base']['recall']['value'])}, FPR {_pct(rb['base']['fpr']['value'])} "
              "(normals = official test + holdout).", "",
              "| Perturbation | Recall | ΔRecall [95% CI] | FPR | ΔFPR [95% CI] | Median Δscore normal / anomaly |",
              "|---|---|---|---|---|---|"]
        for n, p in rb["perturbations"].items():
            dr, df = p["paired"]["delta_recall"], p["paired"]["delta_fpr"]
            L.append(f"| {n} | {_pct(p['recall']['value'])} | {100 * dr['value']:+.1f} pts "
                     f"[{100 * dr['ci_low']:+.1f}, {100 * dr['ci_high']:+.1f}] | {_pct(p['fpr']['value'])} | "
                     f"{100 * df['value']:+.1f} pts [{100 * df['ci_low']:+.1f}, {100 * df['ci_high']:+.1f}] | "
                     f"{p['score_change']['normal_median_delta']:+.1f} / {p['score_change']['anomaly_median_delta']:+.1f} |")
        L += ["", "Δ CIs: percentile bootstrap over ORIGINAL images (perturbed copy resampled with its "
                  "original), stratified by class, B=" + str(cfg["robustness"]["n_bootstrap"]) + ".", ""]
    L += ["## 5. Cost sensitivity (official-test Recall/FPR; retrospective description, not optimisation)", "",
          "| Threshold | Recall | FPR | Beats both reference policies for k in |", "|---|---|---|---|"]
    for r in ct:
        rng = f"({r['k_low']:.3f}, {r['k_high']:.2f})" if r["useful"] else "never"
        L.append(f"| {r['threshold']} | {r['recall']:.2f} | {r['fpr']:.2f} | {rng} |")
    L += ["", "k = (prevalence/(1-prevalence)) x (C_FN/C_FP). Example only: prevalence 1% and a missed defect "
              "costing 50x a review gives k = 0.505.", ""]
    if diag.get("by_type_multilabel"):
        L += ["## 6. Post-hoc diagnostics (test labels used for description only)", "",
              "| Defect type (multi-label) | n | Detected at primary | Recall |", "|---|---|---|---|"]
        for ty, v in diag["by_type_multilabel"].items():
            L.append(f"| {ty} | {v['n']} | {v['detected']} | {_pct(v['recall'])} |")
        if diag.get("by_mask_area_tercile"):
            L += ["", "| Defect size tercile (mask area % of image) | n | Recall |", "|---|---|---|"]
            for k, v in diag["by_mask_area_tercile"].items():
                lo, hi = v["mask_area_pct_range"]
                L.append(f"| {k} ({lo:.3f}-{hi:.3f}%) | {v['n']} | {_pct(v['recall'])} |")
        pk = diag.get("peaks")
        if pk:
            a_, f_ = pk["anomaly_peak_in_defect_mask"], pk["false_alarm_peak_location"]
            L += ["", "Where the anomaly-map maximum falls (peak within 24 native px of the GT mask):", "",
                  f"- Detected anomalies: {a_['detected']['in_mask']}/{a_['detected']['n']} peaks on the defect.",
                  f"- Missed anomalies: {a_['missed']['in_mask']}/{a_['missed']['n']} peaks on the defect.",
                  f"- False alarms ({f_['sources']}): {f_['on_background']}/{f_['n']} peak on the background "
                  f"cloth, {f_['on_board']}/{f_['n']} on the board."]
        L += ["", "Small per-type counts: treat these recalls as indicative only.", ""]
    if qc:
        L += ["## 7. Registration QC (no labels used)", "", "| Split | n | Rotated 180° | Min orientation margin | Margin < 0.1 |",
              "|---|---|---|---|---|"]
        for role, v in qc.items():
            L.append(f"| {role} | {v['n']} | {v['flipped']} | {v['min_orientation_margin']:.3f} | {v['n_margin_below_0.1']} |")
        oc = diag.get("orientation_confound")
        if oc:
            w = oc["auroc_within_rotated_test_subset"]
            L += ["", "**Orientation confound in the benchmark (post hoc):** "
                  f"{oc['test_anomaly_rotated'][0]}/{oc['test_anomaly_rotated'][1]} test anomalies vs "
                  f"{oc['test_normal_rotated'][0]}/{oc['test_normal_rotated'][1]} test normals are rotated 180°. "
                  "After registration, orientation does not shift normal scores (Mann-Whitney "
                  f"p = {oc['normals_rotated_vs_not_mannwhitney_p']:.2f}; orientation-vs-score AUROC "
                  f"{oc['normals_orientation_auroc']:.3f} on calib + holdout + test normals). AUROC within the "
                  f"rotated-only test subset: {w['value']:.3f} [{w['ci_low']:.3f}, {w['ci_high']:.3f}]."]
        L.append("")
    out = rep / "results.md"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    return out


def build_report(cfg) -> None:
    check_freeze(cfg)
    rep, m, fz, rb = _load(cfg)
    figdir = resolve_path(cfg, "figures")
    figdir.mkdir(parents=True, exist_ok=True)
    det = load_detector(cfg)
    figs = [fig_scores(cfg, rep, fz, figdir), fig_roc(cfg, rep, m, fz, figdir), fig_cost(cfg, m, fz, figdir),
            sample_outputs(cfg, det, rep, fz, figdir)]
    diag = defect_diagnostics(cfg, rep, fz)
    qc, conf = registration_qc(cfg, det) if det.pre.registrar is not None else ({}, {})
    diag["orientation_confound"] = conf
    if det.pre.registrar is not None and diag.get("by_type_multilabel"):
        diag["peaks"] = peak_diagnostics(cfg, det, rep, fz)
    ct = cost_table(m)
    write_json(rep / "diagnostics.json", {"defects": diag, "registration_qc": qc, "cost_table": ct})
    md = write_results_md(cfg, rep, m, fz, rb, diag, qc, ct)
    print("report:", *[str(p) for p in figs + [md]], sep="\n  ")


# --------------------------------------------------------------------------- #
# ROI quality figure (v2; FIT normals only)
# --------------------------------------------------------------------------- #
def roi_figure(cfg) -> dict[str, Any]:
    """ROI over the registered FIT mean image, FIT coverage histogram, lowest-coverage FIT images.

    Uses only FIT normals (the images the ROI was built from); no labels, no test data.
    """
    from . import roi as roi_mod
    from .data import select
    from .pipeline import load_rows

    det = load_detector(cfg)
    if det.roi is None:
        raise RuntimeError("`ivad roi` needs a config with scoring.aggregation: roi_max")
    root = resolve_path(cfg, "data_dir")
    rows = select(load_rows(cfg), "fit", "normal")
    roi = det.roi.cpu().numpy()
    h, w = det.pre.hw

    def one(r):
        view = det.pre.view(load_rgb(root / r["image"]))
        small = np.asarray(view.resize((w, h), Image.BILINEAR), np.float32)
        return small, roi_mod.coverage(roi_mod.board_mask(view, det.pre.hw), roi)
    res = list(pmap(cfg)(one, rows))
    cov = np.array([c for _, c in res])
    mean = sum(v for v, _ in res) / len(res)
    worst = np.argsort(cov)[:3]
    rc = cfg["scoring"]["roi"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.0), gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1]})
    axes[0].imshow(mean.astype(np.uint8))
    axes[0].contour(roi.astype(float), levels=[0.5], colors="#7CFC00", linewidths=1.0)
    axes[0].set_title(f"Mean of {len(rows)} registered FIT normals; ROI ({100 * roi.mean():.1f}% of frame)", fontsize=8)
    axes[1].hist(cov, bins=30, color=C_NORMAL)
    axes[1].axvline(rc["coverage_min"], color="#222222", lw=1)
    axes[1].set_title("FIT board-mask coverage by ROI", fontsize=8)
    axes[1].set_xlabel("coverage")
    for ax, i in zip(axes[2:], worst):
        v = res[i][0].astype(np.uint8)
        ax.imshow(v)
        ax.contour(roi.astype(float), levels=[0.5], colors="#7CFC00", linewidths=0.8)
        ax.set_title(f"{Path(rows[i]['image']).name}: coverage {cov[i]:.3f}", fontsize=8)
    for ax in axes[[0, 2, 3, 4]]:
        ax.axis("off")
    fig.tight_layout()
    out_dir = resolve_path(cfg, "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "roi_qc.png", dpi=150)
    plt.close(fig)
    out = {"note": "FIT normals only", "n_fit": int(len(cov)), "roi_area_frac": float(roi.mean()),
           "coverage_min": float(cov.min()), "coverage_p01": float(np.quantile(cov, 0.01)),
           "coverage_median": float(np.median(cov)),
           "fit_pass_frac": float((cov >= rc["coverage_min"]).mean()),
           "lowest_coverage_images": [{"image": rows[i]["image"], "coverage": float(cov[i])} for i in worst]}
    write_json(out_dir / "roi_qc.json", out)
    return out
