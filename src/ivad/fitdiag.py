"""FIT-only descriptive diagnostics for the PCB3 illumination experiment (protocol Addendum B2).

Added after an independent code review, before any PCB3 CALIB, HOLDOUT or test image is scored.
Reads ONLY the 600 role=fit normals and the two fitted model states (template + ROI). Nothing
here is used by fit, calibrate, evaluate or robustness, and nothing here changes a score, a
threshold, a configuration or a decision rule. Every number is descriptive.

Three questions (docs/illumination_experiment_protocol.md, Addendum B2):

A. Registration vs representation. For P0/G1/B1/N1 (the predeclared photometric conditions) and
   both variants, on matched FIT images: native foreground mask area, IoU with the same image's
   P0 mask, centroid displacement, principal-axis angle change, orientation (0/180) decision
   changes, warp discrepancy vs the image's own P0 registration, residual translation vs the
   variant's registration template, model-frame board mask, ROI coverage, ROI fail-safe events,
   invalid/empty masks.
B. ROI specificity. On the SAME registered model frame: per-image board-mask area, fixed ROI
   area, the ROI's construction stages (consensus core / attached structure / closing / margin),
   the fraction of the ROI outside each image's board mask, and the fraction of ROI pixels whose
   colour is the green cloth (heuristic: S > 90 and hue in the FIT background band).
C. Photometric assumptions. Whole-frame, board-region and background-region luminance; the
   candidate's gain/offset; RGB clipping; board saturation and local contrast before/after.

Heuristic quantities (the colour board mask, the cloth-hue class, the template, the P0
registration used as reference) are NOT ground truth and are labelled as such in the output.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from . import photometric
from . import roi as roi_mod
from .align import DOWN, Registrar, _gray_small, board_pose, foreground_mask
from .config import resolve_path, write_json
from .data import select
from .features import center_crop, load_rgb, to_tensor
from .robustness import apply_perturbation

CONDITIONS = ("P0", "G1", "B1", "N1")  # the predeclared photometric conditions (protocol §7); no others
SAT_MIN = 90  # the board-mask saturation rule (align.foreground_mask default)


def _specs(cfg: dict[str, Any]) -> dict[str, dict | None]:
    ids = {p["protocol_id"]: p for p in cfg["robustness"]["perturbations"] if "protocol_id" in p}
    out = {"P0": None}
    for c in CONDITIONS[1:]:
        if c not in ids:
            raise ValueError(f"condition {c} missing from {cfg['_config_path']}")
        out[c] = ids[c]
    return out


def _state(cfg: dict[str, Any]) -> dict[str, Any]:
    from .pipeline import model_path

    st = torch.load(model_path(cfg), map_location="cpu", weights_only=True)
    return {"template": st["template"].numpy(), "roi": st["roi"].numpy().astype(bool),
            "fit_images": list(st["fit_images"])}


def warp_discrepancy(p: dict, q: dict, size_wh: tuple[int, int], crop_wh, step: int = 8) -> float:
    """RMS distance (native px) between where two registrations sample each pixel of the crop.

    Both registrations map an output pixel x to input R(a)(x - o) + c. Photometric perturbations do
    not move the board, so the image's own P0 registration is the reference for its perturbed copies.
    The reference is itself a heuristic estimate, not ground truth.
    """
    w, h = size_wh
    cw, ch = crop_wh
    ox, oy = (w - 1) / 2, (h - 1) / 2
    xs, ys = np.meshgrid(np.arange((w - cw) // 2, (w + cw) // 2, step), np.arange((h - ch) // 2, (h + ch) // 2, step))
    u, v = xs.ravel() - ox, ys.ravel() - oy

    def src(pose):
        a = math.radians(pose["angle"] + (180.0 if pose["flipped"] else 0.0))
        ca, sa = math.cos(a), math.sin(a)
        return ca * u - sa * v + pose["cx"], sa * u + ca * v + pose["cy"]
    (x1, y1), (x2, y2) = src(p), src(q)
    return float(np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2).mean()))


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (dx, dy) of `a` relative to `b` by phase correlation, parabolic sub-pixel peak."""
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-12
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    h, w = c.shape

    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d
    dy = iy + sub(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    dx = ix + sub(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dx), float(dy)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _hsv(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("HSV"), dtype=np.uint8)


def _cloth(hsv: np.ndarray, band) -> np.ndarray:
    """Heuristic cloth class: saturated and in the FIT background hue band (Addendum B1)."""
    return (hsv[..., 1] > SAT_MIN) & (hsv[..., 0] >= band[0]) & (hsv[..., 0] <= band[1])


def _local_contrast(gray: np.ndarray) -> np.ndarray:
    return np.hypot(ndimage.sobel(gray, 1), ndimage.sobel(gray, 0))


class _Variant:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from .pipeline import mask_kwargs, size_hw

        self.cfg = cfg
        self.name = "candidate" if cfg["preprocess"].get("photometric") else "baseline"
        self.corr = photometric.from_config(cfg)
        self.mask_kw = mask_kwargs(cfg)
        st = _state(cfg)
        self.roi, self.fit_images = st["roi"], st["fit_images"]
        self.reg = Registrar(st["template"], self.mask_kw)
        self.crop_wh = cfg["preprocess"]["crop_wh"]
        self.hw = size_hw(cfg)
        self.coverage_min = cfg["scoring"]["roi"]["coverage_min"]
        ph = cfg["preprocess"].get("photometric") or {}
        self.ref = (ph.get("ref_mean"), ph.get("ref_std"))


def _one_image(rel: str, root: Path, variants: list[_Variant], specs: dict, band, region_ref: dict | None = None):
    """All diagnostics for one FIT image; returns (records, accumulators for P0)."""
    raw = load_rgb(root / rel)
    w, h = raw.size
    recs, acc = [], {}
    # Region definitions in the native frame: from the BASELINE P0 mask of the raw image (heuristic).
    base_mask = foreground_mask(raw, **variants[0].mask_kw)
    board_full = np.asarray(Image.fromarray(base_mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    near = ndimage.binary_dilation(base_mask, iterations=6)  # ~24 native px guard band
    raw_small_hsv = _hsv(raw.resize((w // DOWN, h // DOWN), Image.BILINEAR))
    bg_small = ~near & _cloth(raw_small_hsv, band)
    bg_full = np.asarray(Image.fromarray(bg_small.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    p0 = {}
    for var in variants:
        for cond in CONDITIONS:
            img = raw if specs[cond] is None else apply_perturbation(raw, specs[cond])
            r: dict[str, Any] = {"image": rel, "variant": var.name, "condition": cond}
            src_mu, src_sd = photometric.luma_stats(img)
            if var.corr is not None:
                gain = var.ref[1] / src_sd
                r.update({"gain": gain, "offset": var.ref[0] - gain * src_mu})
                img = var.corr(img)
            a = np.asarray(img, dtype=np.uint8)
            y = np.asarray(img.convert("L"), np.float64)
            r.update({
                "luma_mean": float(y.mean()), "luma_std": float(y.std()),
                "luma_board_mean": float(y[board_full].mean()), "luma_board_std": float(y[board_full].std()),
                "luma_bg_mean": float(y[bg_full].mean()) if bg_full.any() else float("nan"),
                "luma_bg_std": float(y[bg_full].std()) if bg_full.any() else float("nan"),
                "clip0_frac": float((a == 0).mean()), "clip255_frac": float((a == 255).mean()),
                "clip_board_frac": float(((a == 0) | (a == 255))[board_full].mean()),
                "board_sat_mean": float(_hsv(img)[..., 1][board_full].mean()),
                "board_local_contrast": float(_local_contrast(y)[board_full].mean()),
            })
            try:
                m = foreground_mask(img, **var.mask_kw)
                pose = board_pose(img, var.mask_kw)
            except ValueError as e:
                r.update({"mask_valid": False, "error": str(e)})
                recs.append(r)
                continue
            view_full, info = var.reg.register(img)
            pose = {**pose, "flipped": info["flipped"]}
            dx, dy = phase_shift(_gray_small(view_full), var.reg.template)
            view = center_crop(view_full, var.crop_wh)
            bm = roi_mod.board_mask(view, var.hw, var.mask_kw)
            n_bm = int(bm.sum())
            cov = float((bm & var.roi).sum() / n_bm) if n_bm else float("nan")
            r.update({
                "mask_valid": n_bm > 0, "mask_area_frac": float(m.mean()), "cx": pose["cx"], "cy": pose["cy"],
                "angle": pose["angle"], "flipped": bool(info["flipped"]), "ncc_margin": info["ncc"] - info["ncc_other"],
                "template_shift_px": float(np.hypot(dx, dy) * DOWN),
                "model_mask_area_frac": float(bm.mean()), "roi_coverage": cov,
                "roi_fallback": bool(not (cov >= var.coverage_min)),
                "roi_outside_mask_frac": float((var.roi & ~bm).sum() / var.roi.sum()),
            })
            if cond == "P0":
                p0[var.name] = (m, pose)
                small = view.resize((var.hw[1], var.hw[0]), Image.BILINEAR)
                cl = _cloth(_hsv(small), band)
                r["roi_cloth_frac"] = float(cl[var.roi].mean())
                r["roi_outside_mask_cloth_frac"] = float(cl[var.roi & ~bm].mean()) if (var.roi & ~bm).any() else 0.0
                g = roi_mod.gray_from_batch(to_tensor(view, var.hw)[None])[0].numpy()
                acc[var.name] = {"mask": bm, "gray": g, "cloth": cl, "rgb": np.asarray(small, np.float64)}
            else:
                m0, pose0 = p0[var.name]
                r.update({"iou_vs_p0": _iou(m, m0),
                          "centroid_shift_px": float(np.hypot(pose["cx"] - pose0["cx"], pose["cy"] - pose0["cy"])),
                          "angle_change_deg": float(abs((pose["angle"] - pose0["angle"] + 90) % 180 - 90)),
                          "flip_changed": bool(pose["flipped"] != pose0["flipped"]),
                          "warp_discrepancy_px": warp_discrepancy(pose, pose0, (w, h), var.crop_wh)})
            recs.append(r)
    return recs, acc


def _q(v, qs=(0, .01, .05, .5, .95, .99, 1)) -> dict[str, float]:
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], np.float64)
    if len(v) == 0:
        return {}
    out = {f"p{int(100 * q):02d}" if 0 < q < 1 else ("min" if q == 0 else "max"): float(np.quantile(v, q)) for q in qs}
    out["mean"] = float(v.mean())
    return out


def _summarise(recs: list[dict], variant: str, cond: str) -> dict[str, Any]:
    rs = [r for r in recs if r["variant"] == variant and r["condition"] == cond]
    ok = [r for r in rs if r.get("mask_valid")]
    col = lambda k: [r.get(k) for r in ok]  # noqa: E731
    s: dict[str, Any] = {"n": len(rs), "invalid_or_empty_mask": len(rs) - len(ok),
                         "native_mask_area_frac": _q(col("mask_area_frac")),
                         "orientation_flips": int(sum(r["flipped"] for r in ok)),
                         "ncc_margin_min": float(min(col("ncc_margin"))) if ok else None,
                         "template_residual_shift_px": _q(col("template_shift_px")),
                         "model_frame_mask_area_frac": _q(col("model_mask_area_frac")),
                         "roi_coverage": _q(col("roi_coverage")),
                         "roi_fallback_count": int(sum(r["roi_fallback"] for r in ok)),
                         "roi_outside_mask_frac": _q(col("roi_outside_mask_frac")),
                         "photometric": {k: _q(col(k)) for k in ("luma_mean", "luma_std", "luma_board_mean",
                                                                 "luma_board_std", "luma_bg_mean", "clip0_frac",
                                                                 "clip255_frac", "clip_board_frac", "board_sat_mean",
                                                                 "board_local_contrast")}}
    if variant == "candidate":
        s["photometric"]["gain"] = _q(col("gain"))
        s["photometric"]["offset"] = _q(col("offset"))
    if cond != "P0":
        ious = np.array(col("iou_vs_p0"))
        warp = np.array(col("warp_discrepancy_px"))
        s.update({"iou_vs_p0": _q(ious), "iou_below_0.95": int((ious < 0.95).sum()),
                  "iou_below_0.90": int((ious < 0.90).sum()),
                  "centroid_shift_px": _q(col("centroid_shift_px")),
                  "angle_change_deg": _q(col("angle_change_deg")),
                  "flip_changed_count": int(sum(r["flip_changed"] for r in ok)),
                  "warp_discrepancy_px": _q(warp), "warp_discrepancy_over_5px": int((warp > 5).sum()),
                  "warp_discrepancy_over_20px": int((warp > 20).sum())})
    else:
        s["roi_cloth_frac"] = _q(col("roi_cloth_frac"))
        s["roi_outside_mask_cloth_frac"] = _q(col("roi_outside_mask_cloth_frac"))
    return s


def roi_stages(var: _Variant, masks_sum: np.ndarray, g1: np.ndarray, g2: np.ndarray, n: int) -> dict[str, Any]:
    """Recompute the ROI from the FIT P0 masks and grey moments; split it into construction stages."""
    rc = var.cfg["scoring"]["roi"]
    freq = masks_sum / n
    core = ndimage.binary_fill_holes(roi_mod._largest_component(freq >= rc["vote_fraction"]))
    gray_std = np.sqrt(np.clip(g2 / n - (g1 / n) ** 2, 0, None))
    struct, st = roi_mod.attached_structure(core, gray_std, rc["structure_k"], rc["structure_far_px"])
    closed = ndimage.binary_fill_holes(roi_mod._closing(struct, rc["closing_px"]))
    roi = ndimage.binary_dilation(closed, structure=roi_mod._disk(rc["margin_px"]))
    stages = {"core": core, "structure": struct & ~core, "closing": closed & ~struct, "margin": roi & ~closed}
    return {"recomputed_roi": roi, "stages": stages, "freq": freq, "gray_std": gray_std, "structure_info": st,
            "recomputed_equals_stored_pixels_differing": int((roi != var.roi).sum())}


def run(cfg_base: dict[str, Any], cfg_cand: dict[str, Any]) -> dict[str, Any]:
    from .pipeline import load_rows, pmap

    variants = [_Variant(cfg_base), _Variant(cfg_cand)]
    if [v.name for v in variants] != ["baseline", "candidate"]:
        raise ValueError("--base must be the uncorrected config and --new the corrected one")
    rows = select(load_rows(cfg_base), "fit", "normal")
    if [r["image"] for r in rows] != variants[0].fit_images or [r["image"] for r in rows] != variants[1].fit_images:
        raise RuntimeError("FIT image list differs from the fitted models' FIT images")
    if any(r["role"] != "fit" for r in rows):
        raise RuntimeError("non-FIT row selected")
    specs = _specs(cfg_cand)
    if _specs(cfg_base) != specs:
        raise RuntimeError("condition definitions differ between the two configs")
    band = cfg_base["preprocess"]["board_mask"]["exclude_hue"]
    root = resolve_path(cfg_base, "data_dir")
    out_dir = resolve_path(cfg_base, "inspection")

    recs: list[dict] = []
    hw = variants[0].hw
    accs = {v.name: {"mask": np.zeros(hw, np.int64), "g1": np.zeros(hw), "g2": np.zeros(hw),
                     "cloth": np.zeros(hw, np.int64), "rgb": np.zeros(hw + (3,))} for v in variants}
    examples = {}
    for i, (rr, acc) in enumerate(pmap(cfg_base)(lambda r: _one_image(r["image"], root, variants, specs, band), rows)):
        recs += rr
        for name, a in acc.items():
            A = accs[name]
            A["mask"] += a["mask"]
            A["g1"] += a["gray"]
            A["g2"] += a["gray"] ** 2
            A["cloth"] += a["cloth"]
            A["rgb"] += a["rgb"]
        if (i + 1) % 100 == 0:
            print(f"fit-diagnostics: {i + 1}/{len(rows)} images")
    n = len(rows)

    summary: dict[str, Any] = {}
    stages_out: dict[str, Any] = {}
    for v in variants:
        summary[v.name] = {c: _summarise(recs, v.name, c) for c in CONDITIONS}
        A = accs[v.name]
        rs = roi_stages(v, A["mask"], A["g1"], A["g2"], n)
        cloth_freq = A["cloth"] / n
        area = {k: float(m.mean()) for k, m in rs["stages"].items()}
        stages_out[v.name] = {
            "roi_area_frac_of_model_frame": float(v.roi.mean()),
            "recomputed_roi_pixels_differing_from_stored": rs["recomputed_equals_stored_pixels_differing"],
            "stage_area_frac_of_model_frame": area,
            "stage_share_of_roi": {k: a / float(v.roi.mean()) for k, a in area.items()},
            "stage_mean_cloth_frac": {k: float(cloth_freq[m].mean()) if m.any() else 0.0
                                      for k, m in rs["stages"].items()},
            "roi_mean_cloth_frac_from_pixel_frequency": float(cloth_freq[v.roi].mean()),
            "consensus_core_area_frac": area["core"],
            "structure_threshold": rs["structure_info"],
        }
        examples[v.name] = (A["rgb"] / n, rs)

    by_image = {}
    for r in recs:
        by_image.setdefault((r["variant"], r["condition"]), {})[r["image"]] = r
    # Matched-image comparison: does the candidate keep the P0 mask better than the baseline?
    matched = {}
    for c in CONDITIONS[1:]:
        b, k = by_image[("baseline", c)], by_image[("candidate", c)]
        ib = np.array([b[x].get("iou_vs_p0", np.nan) for x in b])
        ic = np.array([k[x].get("iou_vs_p0", np.nan) for x in b])
        wb = np.array([b[x].get("warp_discrepancy_px", np.nan) for x in b])
        wc = np.array([k[x].get("warp_discrepancy_px", np.nan) for x in b])
        matched[c] = {"iou_median_base": float(np.nanmedian(ib)), "iou_median_cand": float(np.nanmedian(ic)),
                      "images_cand_iou_higher": int((ic > ib + 1e-9).sum()),
                      "images_cand_iou_lower": int((ic < ib - 1e-9).sum()),
                      "warp_median_base": float(np.nanmedian(wb)), "warp_median_cand": float(np.nanmedian(wc))}

    # Background dominance (FIT P0, raw images = baseline P0 records)
    b0 = [by_image[("baseline", "P0")][r["image"]] for r in rows]
    c0 = [by_image[("candidate", "P0")][r["image"]] for r in rows]
    fm = np.array([r["luma_mean"] for r in b0])
    bm_ = np.array([r["luma_board_mean"] for r in b0])
    gm = np.array([r["luma_bg_mean"] for r in b0])
    cbm = np.array([r["luma_board_mean"] for r in c0])
    board_px = float(np.mean([r["mask_area_frac"] for r in b0]))
    dominance = {
        "board_pixel_share_of_frame_mean": board_px,
        "corr_frame_mean_vs_board_mean": float(np.corrcoef(fm, bm_)[0, 1]),
        "corr_frame_mean_vs_background_mean": float(np.corrcoef(fm, gm)[0, 1]),
        "board_mean_std_across_images_raw": float(bm_.std()),
        "board_mean_std_across_images_corrected": float(cbm.std()),
        "frame_mean_std_across_images_raw": float(fm.std()),
        "note": "board region = baseline P0 colour mask of the raw image (heuristic); background region = "
                "cloth-hue pixels > ~24 px from that mask (heuristic). Not ground truth.",
    }
    # How far does the correction bring each perturbed board back to its own clean corrected board?
    restore = {}
    for c in CONDITIONS[1:]:
        for vname in ("baseline", "candidate"):
            p = by_image[(vname, "P0")]
            q = by_image[(vname, c)]
            d = np.array([q[x]["luma_board_mean"] - p[x]["luma_board_mean"] for x in p])
            restore.setdefault(c, {})[vname] = {"board_mean_change_vs_own_P0": _q(d, (0, .5, 1))}

    report = {
        "status": "FIT-ONLY DESCRIPTIVE DIAGNOSTIC (protocol Addendum B2). Not a model-selection criterion; "
                  "changes no score, threshold, configuration or decision rule.",
        "data": {"role": "fit", "n_images": n, "conditions": list(CONDITIONS),
                 "condition_specs": {c: specs[c] for c in CONDITIONS},
                 "calib_holdout_test_images_read": 0},
        "definitions": {
            "native_mask": "align.foreground_mask (colour heuristic incl. Addendum B1 hue exclusion), 1/4 scale",
            "iou_vs_p0": "IoU of the condition's native mask with the same image's P0 mask, same variant",
            "warp_discrepancy_px": "RMS distance (native px) over the 1152x768 crop between the sampling "
                                   "positions of the condition's registration and the same image's P0 "
                                   "registration (same variant). The P0 registration is a heuristic reference.",
            "template_residual_shift_px": "phase-correlation translation (native px, 1/4-scale grey, full frame) "
                                          "between the registered image and the variant's FIT template "
                                          "(heuristic reference = mean of registered FIT images)",
            "roi_coverage": "fraction of the image's model-frame board mask inside the fixed ROI "
                            "(the fail-safe statistic; fallback iff < coverage_min = 0.95)",
            "roi_outside_mask_frac": "fraction of fixed-ROI pixels outside the image's own model-frame "
                                     "board mask (colour heuristic: misses pins, LED leads, dark parts)",
            "roi_cloth_frac": "fraction of fixed-ROI pixels classified as green cloth (S > 90 and hue in the "
                              "Addendum B1 band) in the registered model frame (heuristic)",
            "gain/offset": "candidate correction I' = gain * I + offset computed from the image it receives",
            "clip": "fraction of RGB channel values equal to 0 or 255 after perturbation (and correction)",
            "board_local_contrast": "mean Sobel gradient magnitude of the grey image inside the board region",
        },
        "summary": summary,
        "matched_baseline_vs_candidate": matched,
        "roi_specificity": stages_out,
        "background_dominance": dominance,
        "board_brightness_change_vs_own_P0": restore,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "fit_diagnostics_B2.json", report)
    keys = sorted({k for r in recs for k in r})
    keys = ["image", "variant", "condition"] + [k for k in keys if k not in ("image", "variant", "condition")]
    with open(out_dir / "fit_diagnostics_B2_per_image.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, keys, lineterminator="\n")
        wr.writeheader()
        for r in recs:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    _figure_roi(examples, variants, rows, by_image, root, band, out_dir / "fit_diag_B2_roi.png")
    _figure_photometric(rows, variants, specs, root, out_dir / "fit_diag_B2_photometric.png")
    print(f"fit-diagnostics -> {out_dir / 'fit_diagnostics_B2.json'}")
    return report


STAGE_COLOURS = {"core": "#ffffff", "structure": "#ffd400", "closing": "#ff8c00", "margin": "#ff2d55"}


def _figure_roi(examples, variants, rows, by_image, root: Path, band, path: Path) -> None:
    v = variants[0]
    mean_rgb, rs = examples["baseline"]
    recs = by_image[("baseline", "P0")]
    order = sorted(rows, key=lambda r: recs[r["image"]]["roi_cloth_frac"])
    picks = [order[0], order[len(order) // 2], order[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    ax = axes[0, 0]
    ax.imshow(mean_rgb.astype(np.uint8))
    for k, m in rs["stages"].items():
        ax.contour(m.astype(float), levels=[0.5], colors=STAGE_COLOURS[k], linewidths=1.0)
    ax.contour(v.roi.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.4)
    st = rs["stages"]
    ax.set_title(f"Baseline, mean of {len(rows)} registered FIT normals (model frame)\n"
                 f"ROI {100 * v.roi.mean():.1f}% = core {100 * st['core'].mean():.1f} + structure "
                 f"{100 * st['structure'].mean():.1f} + closing {100 * st['closing'].mean():.1f} + margin "
                 f"{100 * st['margin'].mean():.1f} (% of frame)", fontsize=9)
    for k, c in STAGE_COLOURS.items():
        ax.plot([], [], color=c, label=k)
    ax.plot([], [], color="#00e5ff", label="ROI boundary")
    ax.legend(loc="lower left", fontsize=7, facecolor="#333333", labelcolor="white")
    for ax, r in zip(axes.ravel()[1:], picks):
        view = center_crop(v.reg(load_rgb(root / r["image"])), v.crop_wh)
        small = view.resize((v.hw[1], v.hw[0]), Image.BILINEAR)
        bm = roi_mod.board_mask(view, v.hw, v.mask_kw)
        cl = _cloth(_hsv(small), band)
        a = np.asarray(small, np.float64)
        tint = a.copy()
        inside_cloth = v.roi & cl
        tint[inside_cloth] = 0.45 * a[inside_cloth] + 0.55 * np.array([255, 0, 200])
        ax.imshow(tint.astype(np.uint8))
        ax.contour(bm.astype(float), levels=[0.5], colors="#ffd400", linewidths=0.9)
        ax.contour(v.roi.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.2)
        rec = recs[r["image"]]
        ax.set_title(f"{Path(r['image']).name}: ROI cloth-hue {100 * rec['roi_cloth_frac']:.0f}%, ROI outside "
                     f"board mask {100 * rec['roi_outside_mask_frac']:.0f}%\n(yellow = board mask, cyan = ROI, "
                     "magenta = cloth-hue pixels inside ROI)", fontsize=8.5)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("PCB3 FIT-only ROI specificity diagnostic (Addendum B2). Heuristic colour classes, not ground truth.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_photometric(rows, variants, specs, root: Path, path: Path) -> None:
    rel = rows[len(rows) // 2]["image"]
    raw = load_rgb(root / rel)
    corr = variants[1].corr
    reg = variants[0].reg
    view0, info = reg.register(raw)
    pose = {**board_pose(raw, variants[0].mask_kw), "flipped": info["flipped"]}
    del view0
    from .align import warp

    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6))
    for j, c in enumerate(CONDITIONS):
        img = raw if specs[c] is None else apply_perturbation(raw, specs[c])
        for i, (lab, im) in enumerate((("as received (baseline input)", img), ("after correction (candidate input)",
                                                                                corr(img)))):
            v = center_crop(warp(im, pose, pose["flipped"]), variants[0].crop_wh)  # same geometry for display
            y = np.asarray(im.convert("L"), np.float64)
            a = np.asarray(im)
            axes[i, j].imshow(v)
            axes[i, j].set_title(f"{c}, {lab}\nframe luma mean {y.mean():.1f} sd {y.std():.1f}; clipped "
                                 f"{100 * ((a == 0) | (a == 255)).mean():.2f}%", fontsize=8)
            axes[i, j].axis("off")
    fig.suptitle(f"{Path(rel).name} (FIT normal): conditions before / after the global luminance correction "
                 "(display uses the clean image's registration)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
