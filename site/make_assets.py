"""Website images from real project outputs (run after the PCB2 and PCB3 runs exist).

    python site/make_assets.py      # -> site/static/img/*.jpg + site/assets_manifest.json

* Workflow mock-up: three PCB2 v2 official-test images (a caught defect, a false alarm, a missed defect)
  rescored end to end with the frozen PCB2 v2 model (for the heat map). The rescored value must match the
  recorded `reports/pcb2_v2/scores_test.csv` score within 0.05 (single-image vs batched GPU arithmetic) with
  the same decision, or the script stops; the page shows the RECORDED score. The ground-truth outline is drawn
  only on the missed-defect example, and labelled as such on the page.
* PCB3 board-mask rules: one FIT normal under the original colour rule and under the Addendum B1 rule,
  plus the original rule's board-area fraction over all 600 FIT normals (FIT only).
* PCB3 ROI diagnostic: the Addendum B2 figure, re-encoded for the web.

All images are derived from the VisA dataset (CC BY 4.0): registered, cropped, resized, with overlays.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = HERE / "static" / "img"
MANIFEST = HERE / "assets_manifest.json"

from ivad.align import foreground_mask  # noqa: E402
from ivad.config import load_config, resolve_path  # noqa: E402
from ivad.data import select  # noqa: E402
from ivad.features import load_rgb  # noqa: E402

# (key, image, expected label, expected decision). Chosen from the recorded PCB2 v2 test outcomes
# (reports/pcb2_v2/sample_predictions.csv): a caught defect, a false alarm, a missed small defect.
CASES = [("caught", "pcb2/Data/Images/Anomaly/006.JPG", "anomaly", "REVIEW"),
         ("false_alarm", "pcb2/Data/Images/Normal/0484.JPG", "normal", "REVIEW"),
         ("missed", "pcb2/Data/Images/Anomaly/072.JPG", "anomaly", "NO_REVIEW")]
WEB_W = 960


def _inferno(v: np.ndarray) -> np.ndarray:
    import matplotlib
    return (matplotlib.colormaps["inferno"](v)[..., :3] * 255).astype(np.float64)


def _outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    m = mask.astype(bool)
    edge = m ^ ndimage.binary_erosion(m, iterations=1)
    return ndimage.binary_dilation(edge, iterations=max(width - 1, 0)) if width > 1 else edge


def _dashed(edge: np.ndarray, period: int = 14) -> np.ndarray:
    yy, xx = np.nonzero(edge)
    keep = ((xx + yy) // (period // 2)) % 2 == 0
    out = np.zeros_like(edge)
    out[yy[keep], xx[keep]] = True
    return out


def workflow_cases() -> list[dict]:
    from ivad.pipeline import check_freeze, load_detector, read_scores
    from ivad.report import _gt_mask_view

    cfg = load_config(ROOT / "config" / "pcb2_v2.yaml")
    fz = check_freeze(cfg)
    det = load_detector(cfg)
    t = fz["thresholds"][fz["primary"]]["threshold"]
    lo, hi = fz["calib_score_summary"]["median"], max(2 * t, 1)
    rows, scores = read_scores(resolve_path(cfg, "reports") / "scores_test.csv")
    recorded = {r["image"]: (s, r["label"]) for r, s in zip(rows, scores)}
    root = resolve_path(cfg, "data_dir")
    anno = {r["image"]: r for r in csv.DictReader(open(root / "pcb2" / "image_anno.csv", encoding="utf-8"))}
    roi = det.roi.cpu().numpy()
    out = []
    for key, rel, label, decision in CASES:
        img = load_rgb(root / rel)
        s, amap, extra = det.score_one(img)
        rec, rec_label = recorded[rel]
        if abs(s - rec) > 0.05 or rec_label != label or (("REVIEW" if s > t else "NO_REVIEW") != decision):
            raise SystemExit(f"{rel}: rescored {s:.4f} / recorded {rec:.4f} / label {rec_label} - does not match")
        a = amap.cpu().numpy().astype(np.float64)
        h, w = a.shape
        base = np.asarray(det.pre.view(img).resize((w, h), Image.BILINEAR), np.float64)
        v = np.clip((a - lo) / (hi - lo), 0, 1)
        alpha = np.where(a > lo, 0.25 + 0.5 * v, 0)[..., None]
        rgb = base * (1 - alpha) + _inferno(v) * alpha
        rgb[_dashed(_outline(roi, 2))] = (124, 252, 0)
        if key == "missed":
            gt = _gt_mask_view(cfg, det, rel, anno[rel]["mask"])
            rgb[_outline(ndimage.binary_dilation(gt, iterations=6), 2)] = (0, 229, 255)
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((WEB_W, WEB_W * h // w), Image.LANCZOS)
        name = f"workflow_{key}.jpg"
        im.save(OUT / name, quality=84, optimize=True, progressive=True)
        out.append({"key": key, "image": rel, "image_name": Path(rel).name, "label": label, "decision": decision,
                    "score": f"{rec:.1f}", "score_value": rec, "rescored_for_heatmap": s, "threshold": f"{t:.1f}",
                    "coverage": f"{extra['roi_coverage']:.2f}", "file": name, "w": im.width, "h": im.height})
    return out


def pcb3_mask_rules() -> dict:
    from ivad.pipeline import load_rows, mask_kwargs, pmap

    cfg = load_config(ROOT / "config" / "pcb3_baseline.yaml")
    rows = select(load_rows(cfg), "fit", "normal")
    root = resolve_path(cfg, "data_dir")
    kw = mask_kwargs(cfg)
    areas = np.array(list(pmap(cfg)(lambda r: (float(foreground_mask(load_rgb(root / r["image"])).mean()),
                                               float(foreground_mask(load_rgb(root / r["image"]), **kw).mean())),
                                    rows)))
    rel = rows[len(rows) // 2]["image"]
    img = load_rgb(root / rel)
    w, h = img.size
    small = np.asarray(img.resize((w // 4, h // 4), Image.BILINEAR), np.float64)
    panels = []
    for m in (foreground_mask(img), foreground_mask(img, **kw)):
        p = small.copy()
        p[m] = 0.45 * p[m] + 0.55 * np.array([255, 0, 200])
        p[_outline(m, 1)] = (255, 255, 255)
        panels.append(p)
    gap = 8
    canvas = np.full((small.shape[0], 3 * small.shape[1] + 2 * gap, 3), 255.0)
    for i, p in enumerate([small] + panels):
        x = i * (small.shape[1] + gap)
        canvas[:, x:x + small.shape[1]] = p
    im = Image.fromarray(canvas.astype(np.uint8))
    im = im.resize((1200, 1200 * im.height // im.width), Image.LANCZOS)
    im.save(OUT / "pcb3_mask_rules.jpg", quality=84, optimize=True, progressive=True)
    return {"file": "pcb3_mask_rules.jpg", "w": im.width, "h": im.height, "image": rel, "n_fit": len(rows),
            "original_rule_area_median": float(np.median(areas[:, 0])),
            "original_rule_area_min": float(areas[:, 0].min()),
            "adapted_rule_area_median": float(np.median(areas[:, 1])),
            "source": "FIT normals only; align.foreground_mask without and with preprocess.board_mask.exclude_hue"}


def pcb3_roi() -> dict:
    im = Image.open(ROOT / "reports" / "pcb3_inspection" / "fit_diag_B2_roi.png").convert("RGB")
    im = im.resize((1200, 1200 * im.height // im.width), Image.LANCZOS)
    im.save(OUT / "pcb3_roi.jpg", quality=82, optimize=True, progressive=True)
    return {"file": "pcb3_roi.jpg", "w": im.width, "h": im.height,
            "source": "reports/pcb3_inspection/fit_diag_B2_roi.png (FIT only)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"note": "Generated by site/make_assets.py from recorded project outputs; VisA images CC BY 4.0, modified.",
           "workflow_cases": workflow_cases(), "pcb3_mask_rules": pcb3_mask_rules(), "pcb3_roi": pcb3_roi()}
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
