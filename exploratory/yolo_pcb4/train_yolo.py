"""Exploratory E1 (PROTOCOL.md §5.1, E5-E7): train and score YOLOv8s / YOLO11s. Separate environment:

    <yolo-venv>/Scripts/python.exe exploratory/yolo_pcb4/train_yolo.py

Resumable: a run whose score file exists is skipped. Hyperparameters are fixed by the protocol; do not edit.
Writes exploratory/yolo_pcb4/results/scores_<arch>_<run>.csv, map50_<arch>_<run>.json, latency_<arch>.json,
train_args_<arch>_<run>.json; weights and logs stay in artifacts/exploratory_yolo_pcb4/runs/.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "artifacts" / "exploratory_yolo_pcb4"
OUT = HERE / "results"
HP = dict(imgsz=1024, epochs=100, patience=30, batch=8, seed=0, deterministic=True, workers=4)  # §5.1
PRED = dict(imgsz=1024, conf=0.001, iou=0.7, verbose=False)
RUNS = [(a, f"cv_fold{k}") for a in ("yolov8s", "yolo11s") for k in range(1, 6)] + \
       [("yolo11s", "unseen_missing"), ("yolo11s", "unseen_extra")]


def split_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(open(OUT / "split.csv", newline="", encoding="utf-8")))


def img(r: dict[str, str]) -> str:
    return str((ART / "images" / r["image"]).resolve())


def scores(model: YOLO, paths: list[str]) -> list[float]:
    out = []
    for i in range(0, len(paths), 16):
        for res in model.predict(source=paths[i:i + 16], **PRED):
            c = res.boxes.conf
            out.append(float(c.max()) if c is not None and len(c) else 0.0)
    return out


def latency(model: YOLO, paths: list[str], warmup: int = 5) -> dict:
    t = []
    for i, p in enumerate(paths[:warmup] + paths):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(source=p, **PRED)
        torch.cuda.synchronize()
        if i >= warmup:
            t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return {"n_images": len(t), "median_ms": t[len(t) // 2], "device_name": torch.cuda.get_device_name(0),
            "definition": "ultralytics predict from file path (decode + letterbox + forward + NMS), batch 1, imgsz 1024, after warm-up"}


def main() -> None:
    os.chdir(ART)  # official pretrained weights are fetched/cached here by ultralytics
    rows = split_rows()
    calib = [r for r in rows if r["role"] == "calib"]
    test_n = [r for r in rows if r["role"] == "test_n"]
    log = open(ART / "train_yolo.log", "a", encoding="utf-8")
    for arch, run in RUNS:
        dst = OUT / f"scores_{arch}_{run}.csv"
        if dst.exists():
            continue
        d = ART / "yolo" / run
        test_images = [ln.strip() for ln in (d / "test.txt").read_text().splitlines() if ln.strip()]
        by_path = {img(r): r for r in rows}
        test_rows = [by_path[p] for p in test_images]
        t0 = time.time()
        model = YOLO(f"{arch}.pt")
        model.train(data=str(d / "data.yaml"), project=str(ART / "runs"), name=f"{arch}_{run}", exist_ok=True,
                    plots=False, verbose=False, **HP)
        best = ART / "runs" / f"{arch}_{run}" / "weights" / "best.pt"
        model = YOLO(str(best))
        sets = [("test", test_rows), ("calib", calib), ("test_n", test_n)]
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["image", "label", "role", "set", "score"])
            for name, rs in sets:
                for r, s in zip(rs, scores(model, [img(r) for r in rs])):
                    w.writerow([r["image"], r["label"], r["role"], name, f"{s:.6f}"])
        m = model.val(data=str(d / "data.yaml"), split="test", imgsz=1024, batch=8, plots=False, verbose=False)
        (OUT / f"map50_{arch}_{run}.json").write_text(json.dumps(
            {"map50": float(m.box.map50), "map50_95": float(m.box.map), "n_test_images": len(test_rows)}, indent=1))
        args = json.loads(json.dumps(vars(model.ckpt.get("train_args", {})) if hasattr(model.ckpt.get("train_args", {}), "__dict__")
                                     else model.ckpt.get("train_args", {}), default=str))
        (OUT / f"train_args_{arch}_{run}.json").write_text(json.dumps(
            {"train_args": args, "best_epoch": model.ckpt.get("epoch"), "wall_s": round(time.time() - t0, 1)},
            indent=1, default=str))
        if run == "cv_fold1":
            (OUT / f"latency_{arch}.json").write_text(json.dumps(latency(model, [img(r) for r in test_n[:50]]), indent=1))
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {arch} {run} done in {time.time() - t0:.0f}s\n")
        log.flush()
        print(arch, run, "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
