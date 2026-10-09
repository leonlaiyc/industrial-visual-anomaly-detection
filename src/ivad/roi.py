"""Board-region ROI for image-level score aggregation (the single v2 change).

v1 scores an image by the maximum of its anomaly map over the whole model
frame, so dust or lint on the background cloth can set the score. v2 takes the
maximum over a fixed board ROI instead. Everything upstream (registration,
features, Gaussian model, anomaly map) is unchanged.

ROI construction (FIT normals only, fixed rules; docs/v2_experiment_protocol.md §4
and Addendum A1):

1. per-image board mask = align.foreground_mask of the registered, cropped view,
   upsampled to the model frame (nearest);
2. consensus = pixels covered in >= vote_fraction of FIT masks, largest
   component, holes filled;
3. (A1) attached structure: the colour mask misses grey/black parts fixed to the
   board (header pins, header base, dark ICs at the edge). After registration
   such parts sit at the same place in every FIT image, so the pixel-wise std of
   the registered FIT grey images is high there, while the cloth texture is
   uncorrelated with the board pose and its std stays low. Pixels with std above
   the cloth level (median + structure_k * robust SD, measured farther than
   structure_far_px from the consensus) that are connected to the consensus are
   added; a closing (closing_px) and hole fill make one solid object;
4. ROI = object dilated by margin_px.

Per-image fail-safe: if less than `coverage_min` of an image's own board mask
lies inside the ROI, the ROI is not trusted for that image and the v1
full-frame maximum is used (never lower than the ROI maximum, so a
registration failure can only make REVIEW more likely).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .align import foreground_mask

AGGREGATIONS = ("full_max", "roi_max")


def aggregation(cfg: dict[str, Any]) -> str:
    """Configured aggregation; configs without a `scoring` section are v1 (full_max)."""
    agg = cfg.get("scoring", {}).get("aggregation", "full_max")
    if agg not in AGGREGATIONS:
        raise ValueError(f"scoring.aggregation must be one of {AGGREGATIONS}, got {agg!r}")
    return agg


def board_mask(view: Image.Image, size_hw: Sequence[int]) -> np.ndarray:
    """Board mask of a registered, cropped view, in the model frame (H, W) bool."""
    m = foreground_mask(view)
    h, w = int(size_hw[0]), int(size_hw[1])
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(m, dtype=bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _closing(m: np.ndarray, r: int) -> np.ndarray:
    """Binary closing that does not erode objects touching the image border."""
    if r <= 0:
        return m
    p = np.pad(m, r, mode="edge")
    return ndimage.binary_closing(p, structure=_disk(r))[r:-r, r:-r]


def attached_structure(core: np.ndarray, gray_std: np.ndarray, k: float, far_px: int) -> tuple[np.ndarray, dict]:
    """Pixels whose FIT grey-level std exceeds the cloth level and that connect to the board core."""
    if gray_std.shape != core.shape:
        raise ValueError("gray_std and board consensus differ in shape")
    far = ~ndimage.binary_dilation(core, structure=_disk(far_px))
    if not far.any():
        raise ValueError("no background pixels far from the board to measure the cloth level")
    vals = gray_std[far]
    med = float(np.median(vals))
    sd = float(1.4826 * np.median(np.abs(vals - med)))
    thr = med + k * max(sd, 1e-6)
    lab, _ = ndimage.label((gray_std > thr) | core)
    obj = np.isin(lab, np.unique(lab[core]))
    return obj, {"cloth_std_median": med, "cloth_std_robust_sd": sd, "std_threshold": thr}


def build_roi(masks: Iterable[np.ndarray], vote_fraction: float, margin_px: int,
              gray_std: np.ndarray | None = None, structure_k: float = 10.0, structure_far_px: int = 48,
              closing_px: int = 8, info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Consensus board ROI from per-image board masks. Returns (roi, vote_frequency).

    With `gray_std` (pixel-wise std of the registered FIT grey images, model frame) the
    board-attached structure of Addendum A1 is added before the margin. `info`, if
    given, receives the intermediate areas and the structure threshold.
    """
    if not 0 < vote_fraction <= 1:
        raise ValueError("vote_fraction must be in (0, 1]")
    if margin_px < 0:
        raise ValueError("margin_px must be >= 0")
    acc, n = None, 0
    for m in masks:
        m = np.asarray(m, dtype=bool)
        if acc is None:
            acc = np.zeros(m.shape, dtype=np.int64)
        elif m.shape != acc.shape:
            raise ValueError("board masks differ in shape")
        acc += m
        n += 1
    if n == 0:
        raise ValueError("need at least one board mask")
    freq = acc / n
    core = ndimage.binary_fill_holes(_largest_component(freq >= vote_fraction))
    if not core.any():
        raise ValueError("no pixel reaches the board-mask vote fraction")
    obj = core
    if gray_std is not None:
        obj, st = attached_structure(core, np.asarray(gray_std, np.float64), structure_k, structure_far_px)
        obj = ndimage.binary_fill_holes(_closing(obj, closing_px))
        if info is not None:
            info.update(st)
    roi = ndimage.binary_dilation(obj, structure=_disk(margin_px)) if margin_px else obj
    if info is not None:
        info.update({"core_area_frac": float(core.mean()), "object_area_frac": float(obj.mean()),
                     "n_masks": n})
    return roi, freq


def coverage(mask: np.ndarray, roi: np.ndarray) -> float:
    """Fraction of an image's board-mask pixels that lie inside the ROI."""
    mask, roi = np.asarray(mask, bool), np.asarray(roi, bool)
    if mask.shape != roi.shape:
        raise ValueError(f"mask {mask.shape} and ROI {roi.shape} differ in shape")
    n = int(mask.sum())
    if n == 0:
        raise ValueError("empty board mask; cannot assess ROI coverage")
    return float((mask & roi).sum() / n)


def validate_roi(roi: np.ndarray | torch.Tensor, size_hw: Sequence[int],
                 min_area_frac: float = 0.0, max_area_frac: float = 1.0) -> None:
    """Raise unless the ROI is a non-empty, single-region bool mask of the model frame."""
    r = roi.cpu().numpy() if isinstance(roi, torch.Tensor) else np.asarray(roi)
    if r.dtype != bool:
        raise ValueError(f"ROI must be boolean, got {r.dtype}")
    if r.shape != (int(size_hw[0]), int(size_hw[1])):
        raise ValueError(f"ROI shape {r.shape} does not match the model frame {tuple(size_hw)}")
    if not r.any():
        raise ValueError("ROI is empty")
    _, n = ndimage.label(r)
    if n != 1:
        raise ValueError(f"ROI must be one connected region, found {n}")
    frac = float(r.mean())
    if not min_area_frac <= frac <= max_area_frac:
        raise ValueError(f"ROI area fraction {frac:.3f} outside [{min_area_frac}, {max_area_frac}]")


def roi_max(amap: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    """Maximum of each anomaly map (B, H, W) over the ROI (H, W) bool -> (B,)."""
    if roi.dtype != torch.bool:
        raise ValueError("ROI must be a bool tensor")
    if amap.ndim != 3 or tuple(amap.shape[1:]) != tuple(roi.shape):
        raise ValueError(f"anomaly maps {tuple(amap.shape)} do not match ROI {tuple(roi.shape)}")
    if not bool(roi.any()):
        raise ValueError("ROI is empty")
    return amap[:, roi.to(amap.device)].amax(dim=1)


def aggregate(amap: torch.Tensor, roi: torch.Tensor, coverages: Sequence[float],
              coverage_min: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v2 image scores with the per-image fail-safe.

    Returns (score, full_frame_max, fallback) each of shape (B,): score is the
    ROI maximum where coverage >= coverage_min, else the full-frame maximum.
    """
    cov = torch.as_tensor(np.asarray(coverages, dtype=np.float64))
    if cov.shape != (amap.shape[0],) or not torch.isfinite(cov).all():
        raise ValueError("need one finite coverage value per image")
    full = amap.flatten(1).amax(dim=1)
    inside = roi_max(amap, roi)
    fallback = (cov < coverage_min).to(full.device)
    return torch.where(fallback, full, inside), full, fallback


def gray_from_batch(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised RGB batch (B, 3, H, W) -> grey levels 0-255 (B, H, W), float64 (ITU-R 601)."""
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float64).view(1, 3, 1, 1)
    rgb = (x.detach().cpu().double() * std + mean) * 255.0
    w = torch.tensor((0.299, 0.587, 0.114), dtype=torch.float64).view(1, 3, 1, 1)
    return (rgb * w).sum(1)


def roi_qc(roi: np.ndarray, fit_coverages: Sequence[float], rc: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered FIT-image quality gates; raises ValueError on failure."""
    cov = np.asarray(fit_coverages, dtype=np.float64)
    pass_frac = float((cov >= rc["coverage_min"]).mean())
    _, n_regions = ndimage.label(roi)
    qc = {"area_frac": float(roi.mean()), "n_regions": int(n_regions), "n_fit": int(len(cov)),
          "fit_coverage_min": float(cov.min()), "fit_coverage_p01": float(np.quantile(cov, 0.01)),
          "fit_coverage_median": float(np.median(cov)), "fit_pass_frac": pass_frac,
          "rules": {k: rc[k] for k in ("vote_fraction", "margin_px", "coverage_min", "fit_pass_frac_min",
                                       "min_area_frac", "max_area_frac", "structure_k", "structure_far_px",
                                       "closing_px") if k in rc}}
    validate_roi(roi, roi.shape, rc["min_area_frac"], rc["max_area_frac"])
    if pass_frac < rc["fit_pass_frac_min"]:
        raise ValueError(f"ROI QC failed: only {pass_frac:.3f} of FIT images have coverage >= "
                         f"{rc['coverage_min']} (required {rc['fit_pass_frac_min']})")
    qc["status"] = "PASS"
    return qc
