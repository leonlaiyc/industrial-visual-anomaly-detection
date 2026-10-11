"""POST-HOC check (not a pre-registered endpoint; logged in results/deviations.md): for each out-of-fold defect
image that a YOLO fold model flags, does the box that triggered the flag (highest confidence) touch the
ground-truth defect? Guards against a shortcut (e.g. capture-session cues) behind a high image-level recall.

    <yolo-venv>/Scripts/python.exe exploratory/yolo_pcb4/posthoc_localization.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "artifacts" / "exploratory_yolo_pcb4"
OUT = HERE / "results"
DATA = ROOT / "data" / "visa"
TOL_PX = 16  # mask dilation: a box "touches" the defect if it overlaps the mask grown by this margin


def main() -> None:
    split = {r["image"]: r for r in csv.DictReader(open(OUT / "split.csv", newline="", encoding="utf-8"))}
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    out = {"note": "post hoc; not a pre-registered endpoint", "tolerance_px": TOL_PX}
    for arch in ("yolov8s", "yolo11s"):
        rows = []
        for k in range(1, 6):
            thr = res["methods"][arch].get("thresholds_by_fold", [None] * 5)[k - 1]
            sc = list(csv.DictReader(open(OUT / f"scores_{arch}_cv_fold{k}.csv", newline="", encoding="utf-8")))
            calib = sorted(float(r["score"]) for r in sc if r["set"] == "calib")
            thr = calib[143] if thr is None else thr  # 144th of 150 = P95 conformal rank (PROTOCOL §6)
            model = YOLO(str(ART / "runs" / f"{arch}_cv_fold{k}" / "weights" / "best.pt"))
            for r in (r for r in sc if r["set"] == "test"):
                if float(r["score"]) <= thr:
                    continue  # not flagged; nothing to localise
                img = DATA / r["image"]
                pred = model.predict(source=str(img), imgsz=1024, conf=0.001, iou=0.7, verbose=False)[0]
                b = pred.boxes
                top = int(b.conf.argmax())
                x0, y0, x1, y1 = (int(round(v)) for v in b.xyxy[top].tolist())
                m = Image.open(DATA / split[r["image"]]["mask"]).convert("L").point(lambda v: 255 if v > 0 else 0)
                m = np.array(m.filter(ImageFilter.MaxFilter(2 * TOL_PX + 1))) > 0
                hit_top = bool(m[max(y0, 0):max(y1, 0) + 1, max(x0, 0):max(x1, 0) + 1].any())
                above = [i for i in range(len(b.conf)) if float(b.conf[i]) > thr]
                hit_any = any(m[max(int(b.xyxy[i][1]), 0):int(b.xyxy[i][3]) + 1,
                                max(int(b.xyxy[i][0]), 0):int(b.xyxy[i][2]) + 1].any() for i in above)
                rows.append({"image": r["image"], "fold": k, "top_conf": float(b.conf[top]), "top_box_on_defect": hit_top,
                             "any_flagging_box_on_defect": hit_any})
        n = len(rows)
        out[arch] = {"n_flagged_defects": n, "top_box_on_defect": sum(x["top_box_on_defect"] for x in rows),
                     "any_flagging_box_on_defect": sum(x["any_flagging_box_on_defect"] for x in rows),
                     "off_defect": [x for x in rows if not x["top_box_on_defect"]]}
        print(arch, n, out[arch]["top_box_on_defect"], out[arch]["any_flagging_box_on_defect"])
    (OUT / "posthoc_localization.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
