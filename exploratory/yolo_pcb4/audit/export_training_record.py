"""Audit export (2026-10-10, after the experiment): small, path-free training record for every YOLO run, read from the
gitignored run directories. Changes nothing. Development workspace, project environment:

    .venv/Scripts/python.exe exploratory/yolo_pcb4/audit/export_training_record.py

Writes exploratory/yolo_pcb4/audit/train_curves/<arch>_<run>.csv (ultralytics results.csv, per-epoch internal-validation
metrics) and audit/training_record.json (checkpoint epoch, early stopping, resolved optimizer, augmentation settings,
pretrained-weight SHA-256).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = ROOT / "artifacts" / "exploratory_yolo_pcb4"
RUNS = [(a, f"cv_fold{k}") for a in ("yolov8s", "yolo11s") for k in range(1, 6)] + \
       [("yolo11s", "unseen_missing"), ("yolo11s", "unseen_extra")]
AUG_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
            "mosaic", "close_mosaic", "mixup", "cutmix", "copy_paste", "auto_augment", "erasing")
HP_KEYS = ("task", "model", "imgsz", "epochs", "patience", "batch", "seed", "deterministic", "workers", "optimizer",
           "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "cos_lr", "amp", "single_cls", "rect", "pretrained",
           "val", "fraction", "freeze", "multi_scale", "max_det", "iou")


def main() -> None:
    (HERE / "train_curves").mkdir(exist_ok=True)
    log = (ART / "train_stdout.log").read_text(encoding="utf-8", errors="replace")
    opt = re.findall(r"optimizer:\S* (\w+\(lr=[\d.]+, momentum=[\d.]+\))", log)
    rec: dict = {"note": "exported after the experiment; checkpoint = ultralytics best.pt, chosen by fitness = internal-val "
                         "mAP50-95 (ultralytics 8.4 DetMetrics.fitness weights [P, R, mAP50, mAP50-95] = [0, 0, 0, 1])",
                 "resolved_optimizer_per_run_in_order": opt,
                 "pretrained_weights_sha256": {w: hashlib.sha256((ART / w).read_bytes()).hexdigest()
                                               for w in ("yolov8s.pt", "yolo11s.pt")},
                 "runs": {}}
    for arch, run in RUNS:
        d = ART / "runs" / f"{arch}_{run}"
        rows = list(csv.DictReader(open(d / "results.csv", newline="", encoding="utf-8")))
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in rows]
        (HERE / "train_curves" / f"{arch}_{run}.csv").write_text(
            (d / "results.csv").read_text(encoding="utf-8"), encoding="utf-8")
        fit = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        best = max(range(len(fit)), key=lambda i: (fit[i], i))  # ultralytics keeps the latest epoch on a tie (>=)
        args = yaml.safe_load((d / "args.yaml").read_text(encoding="utf-8"))
        rec["runs"][f"{arch}_{run}"] = {
            "epochs_completed": len(rows), "early_stopped": len(rows) < int(args["epochs"]),
            "best_epoch": int(rows[best]["epoch"]),
            "best_internal_val": {"mAP50": float(rows[best]["metrics/mAP50(B)"]), "mAP50-95": fit[best],
                                  "precision": float(rows[best]["metrics/precision(B)"]),
                                  "recall": float(rows[best]["metrics/recall(B)"])},
            "best_pt_sha256": hashlib.sha256((d / "weights" / "best.pt").read_bytes()).hexdigest(),
            "hyperparameters": {k: args.get(k) for k in HP_KEYS},
            "augmentation": {k: args.get(k) for k in AUG_KEYS}}
        print(arch, run, len(rows), "epochs; best", rec["runs"][f"{arch}_{run}"]["best_epoch"], round(fit[best], 3))
    (HERE / "training_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
