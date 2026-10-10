"""Audit of the POST-HOC localisation check (2026-10-10, after the experiment). Inference only with the frozen
best.pt of each run: no training, no threshold change, no change to any recorded score or result.

It (1) re-predicts each run's own held-out defect images with the experiment's prediction settings and checks that the
maximum box confidence reproduces the saved image score; (2) reproduces the original check exactly (top box overlaps the
ground-truth mask dilated by 16 px); (3) applies stricter definitions to the same boxes; (4) exports every
above-threshold box so the numbers can be recomputed without weights. Also covers the two unseen-type runs.

YOLO environment, development workspace (needs data/visa/pcb4 and artifacts/exploratory_yolo_pcb4/runs):

    <yolo-venv>/Scripts/python.exe exploratory/yolo_pcb4/audit/localization_audit.py

Writes exploratory/yolo_pcb4/audit/flagging_boxes.csv and localization_audit.json.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
E1 = HERE.parent
ROOT = E1.parents[1]
RES = E1 / "results"
ART = ROOT / "artifacts" / "exploratory_yolo_pcb4"
DATA = ROOT / "data" / "visa"
PRED = dict(imgsz=1024, conf=0.001, iou=0.7, verbose=False)  # train_yolo.py PRED
TOL_PX = 16  # posthoc_localization.py
RUNS = [(a, f"cv_fold{k}") for a in ("yolov8s", "yolo11s") for k in range(1, 6)] + \
       [("yolo11s", "unseen_missing"), ("yolo11s", "unseen_extra")]


def read(p: Path) -> list[dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def thr_of(rows: list[dict[str, str]]) -> float:
    cal = sorted(float(r["score"]) for r in rows if r["set"] == "calib")
    return cal[math.ceil(len(cal) * 0.95 + 0.95 - 1e-12) - 1]  # ceil((n+1)(1-0.05))-th smallest


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main() -> None:
    split = {r["image"]: r for r in read(RES / "split.csv")}
    gt: dict[str, list[tuple[float, float, float, float]]] = {}
    for r in read(HERE / "yolo_labels.csv"):
        W, H = 1358, 1104
        xc, yc, w, h = (float(r[k]) for k in ("xc", "yc", "w", "h"))
        gt.setdefault(r["image"], []).append(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    box_rows, summary, max_dev = [], {}, 0.0
    for arch, run in RUNS:
        sc = read(RES / f"scores_{arch}_{run}.csv")
        thr = thr_of(sc)
        model = YOLO(str(ART / "runs" / f"{arch}_{run}" / "weights" / "best.pt"))
        per = []
        for r in (r for r in sc if r["set"] == "test"):
            saved = float(r["score"])
            res = model.predict(source=str(ART / "images" / r["image"]), **PRED)[0]  # same file the scores came from
            conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
            xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            again = float(conf.max()) if len(conf) else 0.0
            max_dev = max(max_dev, abs(again - saved))
            flagged = saved > thr
            mraw = np.array(Image.open(DATA / split[r["image"]]["mask"]).convert("L")) > 0
            mdil = np.array(Image.fromarray((mraw * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * TOL_PX + 1))) > 0
            order = np.argsort(-conf)
            above = [int(i) for i in order if conf[i] > thr]
            area_img = mraw.shape[0] * mraw.shape[1]

            def touches(m, b, rounding):
                if rounding:
                    x0, y0, x1, y1 = (int(round(v)) for v in b)
                    return bool(m[max(y0, 0):max(y1, 0) + 1, max(x0, 0):max(x1, 0) + 1].any())
                x0, y0, x1, y1 = (int(v) for v in b)
                return bool(m[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1].any())

            def stats_for(i):
                b = xyxy[i]
                best = max((iou(b, g) for g in gt[r["image"]]), default=0.0)
                cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
                return {"touch_dilated16": touches(mdil, b, True), "touch_raw_mask": touches(mraw, b, True),
                        "best_iou_gt_box": best, "center_in_dilated_mask": bool(mdil[min(cy, mraw.shape[0] - 1), min(cx, mraw.shape[1] - 1)]),
                        "box_area_frac": float((b[2] - b[0]) * (b[3] - b[1]) / area_img)}
            for rank, i in enumerate(above):
                st = stats_for(i)
                box_rows.append({"arch": arch, "run": run, "image": r["image"], "rank": rank, "conf": f"{conf[i]:.6f}",
                                 "x0": f"{xyxy[i][0]:.1f}", "y0": f"{xyxy[i][1]:.1f}", "x1": f"{xyxy[i][2]:.1f}", "y1": f"{xyxy[i][3]:.1f}",
                                 "touch_dilated16": int(st["touch_dilated16"]), "touch_raw_mask": int(st["touch_raw_mask"]),
                                 "best_iou_gt_box": f"{st['best_iou_gt_box']:.4f}", "box_area_frac": f"{st['box_area_frac']:.5f}"})
            rec = {"image": r["image"], "flagged": flagged, "saved_score": saved, "reproduced_max_conf": again,
                   "n_boxes_above_threshold": len(above)}
            if flagged:
                top = stats_for(int(order[0]))
                rec.update({f"top_{k}": v for k, v in top.items()})
                rec["any_above_touch_dilated16_trunc"] = any(touches(mdil, xyxy[i], False) for i in above)
                rec["any_above_touch_raw_mask"] = any(stats_for(i)["touch_raw_mask"] for i in above)
                rec["any_above_iou_ge_0.5"] = any(stats_for(i)["best_iou_gt_box"] >= 0.5 for i in above)
            per.append(rec)
        fl = [x for x in per if x["flagged"]]
        cnt = lambda k, f=lambda v: bool(v): int(sum(bool(f(x[k])) for x in fl))  # noqa: E731
        summary[f"{arch}_{run}"] = {
            "threshold": thr, "n_test_images": len(per), "n_flagged": len(fl),
            "flag_changes_if_reproduced_score_used": sum((x["reproduced_max_conf"] > thr) != x["flagged"] for x in per),
            "top_touch_dilated16 (original definition)": cnt("top_touch_dilated16"),
            "any_above_touch_dilated16 (original 'any', int-truncated boxes)": cnt("any_above_touch_dilated16_trunc"),
            "top_touch_raw_mask (no tolerance)": cnt("top_touch_raw_mask"),
            "top_center_in_dilated_mask": cnt("top_center_in_dilated_mask"),
            "top_iou_ge_0.1": cnt("top_best_iou_gt_box", lambda v: v >= 0.1),
            "top_iou_ge_0.5": cnt("top_best_iou_gt_box", lambda v: v >= 0.5),
            "any_above_iou_ge_0.5": cnt("any_above_iou_ge_0.5"),
            "top_box_area_frac_max": max((x["top_box_area_frac"] for x in fl), default=None),
            "top_box_area_frac_median": float(np.median([x["top_box_area_frac"] for x in fl])) if fl else None,
            "images": per}
        print(arch, run, {k: v for k, v in summary[f"{arch}_{run}"].items() if k != "images"}, flush=True)
    agg = {}
    for arch in ("yolov8s", "yolo11s"):
        parts = [summary[f"{arch}_cv_fold{k}"] for k in range(1, 6)]
        agg[arch] = {k: sum(p[k] for p in parts) for k in parts[0] if isinstance(parts[0][k], int) and k != "n_test_images"}
        agg[arch]["n_test_images"] = sum(p["n_test_images"] for p in parts)
    out = {"note": "audit reproduction of a POST-HOC check; inference with frozen weights only; not an endpoint",
           "tolerance_px": TOL_PX, "max_abs_dev_reproduced_vs_saved_score": max_dev,
           "definitions": {
               "top_touch_dilated16": "original: the highest-confidence box (the one that set the image score) shares at least one "
                                      "pixel with the GT mask grown by 16 px (33x33 max filter); no IoU",
               "top_touch_raw_mask": "same, without the 16 px tolerance",
               "top_center_in_dilated_mask": "centre of the top box lies inside the dilated mask",
               "top_iou_ge_t": "IoU of the top box with the best-matching GT component box (the training boxes) >= t",
               "any_above_iou_ge_0.5": "at least one box above the fold threshold has IoU >= 0.5 with a GT component box"},
           "pooled_cv": agg, "runs": summary}
    (HERE / "localization_audit.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    with open(HERE / "flagging_boxes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(box_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(box_rows)
    print(json.dumps(agg, indent=1), "max dev", max_dev)


if __name__ == "__main__":
    main()
