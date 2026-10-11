"""Supplementary visualization after evaluation; read saved scores, never run a model.

Select the highest-scoring and closest-above-threshold unique normal image for
each of YOLOv8s, YOLO11s and M. Scores have model-specific scales and thresholds.
Requires local VisA PCB4 images. No boxes are drawn: saved box evidence covers
defective images only. Run with the project environment from any directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    split = {r["image"]: r for r in read(RES / "split.csv")}
    recorded = json.loads((HERE / "recomputed_metrics.json").read_text(encoding="utf-8"))
    selected = []
    for arch in ("yolov8s", "yolo11s", "m"):
        candidates = []
        folds = range(1, 6) if arch != "m" else [None]
        for fold in folds:
            suffix = f"{arch}_cv_fold{fold}" if fold else arch
            rows = read(RES / f"scores_{suffix}.csv")
            # Verification of the EXISTING threshold, not a calibration change.
            cal = sorted(float(r["score"]) for r in rows if r["role"] == "calib")
            assert len(cal) == 150
            threshold = cal[143]
            method = recorded["methods"][arch]
            saved = (method["per_fold"][fold - 1] if fold else method["single_model"])
            assert threshold == saved["threshold"]
            for r in rows:
                if r["role"] != "test_n" or float(r["score"]) <= threshold:
                    continue
                assert r["label"] == split[r["image"]]["label"] == "normal"
                candidates.append({"model": arch, "fold": fold, "image": r["image"],
                                   "score": float(r["score"]), "threshold": threshold,
                                   "score_csv": f"exploratory/yolo_pcb4/results/scores_{suffix}.csv"})
        # YOLO confidence is in [0,1]; M has its own score scale.
        high = max(candidates, key=lambda r: (r["score"], r["image"]))
        near = min((r for r in candidates if r["image"] != high["image"]),
                   key=lambda r: (r["score"] - r["threshold"], r["image"]))
        for row, reason in ((high, "highest score"), (near, "closest above threshold")):
            row["selection"] = reason
            src = ROOT / "data/visa" / row["image"]
            row["image_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
            assert row["image_sha256"] == split[row["image"]]["sha256"]
            selected.append(row)

    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 18)
    title = ImageFont.truetype(font_path, 29)
    canvas = Image.new("RGB", (1800, 1350), "#f6f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 15), "PCB4 normal images falsely flagged at the original frozen operating point", fill="#17212c", font=title)
    draw.text((25, 58), "Supplementary visualization produced after the original evaluation (2026-10-10). Saved scores only.", fill="#17212c", font=small)
    draw.text((25, 88), "No predicted boxes: saved normal-image box evidence is unavailable. These examples do not establish a cause.", fill="#17212c", font=small)
    for i, row in enumerate(selected):
        # One column per method, high score above / near threshold below.
        col, line = i // 2, i % 2
        x, y = 25 + col * 595, 140 + line * 555
        with Image.open(ROOT / "data/visa" / row["image"]) as original:
            image = original.convert("RGB")
        row["mean_gray_0_255"] = round(float(np.array(image.convert("L")).mean()), 2)
        image.thumbnail((565, 440))
        canvas.paste(image, (x + (565 - image.width) // 2, y))
        model = {"yolov8s": "YOLOv8s", "yolo11s": "YOLO11s", "m": "Normal-only M"}[row["model"]]
        fold = f"fold {row['fold']}" if row["fold"] else "single reference model"
        draw.text((x, y + 445), f"{model} / {fold} / {Path(row['image']).name}", fill="#17212c", font=font)
        draw.text((x, y + 479), f"score {row['score']:.6f} > frozen {row['threshold']:.6f}", fill="#17212c", font=small)
        draw.text((x, y + 508), f"{row['selection']}; mean gray {row['mean_gray_0_255']:.2f}", fill="#17212c", font=small)
    draw.text((25, 1270), "VisA, Amazon Science; Zou et al., ECCV 2022. CC BY 4.0. Images resized; captions added.", fill="#17212c", font=small)
    draw.text((25, 1300), "Source: github.com/amazon-science/spot-diff | License: creativecommons.org/licenses/by/4.0/", fill="#17212c", font=small)
    out = HERE / "fig_false_alarms_supplement.jpg"
    canvas.save(out, quality=82, optimize=True)
    assert out.stat().st_size < 500_000
    manifest = {"status": "Supplementary visualization produced after original evaluation", "date": "2026-10-10",
                "decision": "score > original frozen threshold", "predicted_boxes": "not drawn; no saved normal-image box evidence",
                "selection": "Two illustrative unique images per method, not a representative FPR sample",
                "attribution": "VisA / Amazon Science / Zou et al., ECCV 2022 / CC BY 4.0; resized and captioned",
                "figure": out.name, "figure_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "examples": selected}
    (HERE / "false_alarm_supplement.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{out.name}: {out.stat().st_size} bytes; {len(selected)} verified normal-image false alarms")


if __name__ == "__main__":
    main()
