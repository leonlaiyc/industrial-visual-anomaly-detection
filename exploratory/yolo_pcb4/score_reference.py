"""Exploratory E1 (PROTOCOL.md §5.2): fit and score the normal-only references M and M-reg on PCB4 with the
project's unchanged pipeline code. Project environment:

    .venv/Scripts/python.exe exploratory/yolo_pcb4/score_reference.py

Writes exploratory/yolo_pcb4/results/scores_<m|m_reg>.csv (every non-FIT image) and latency_<name>.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivad.config import load_config  # noqa: E402
from ivad.pipeline import _measure_latency, fit, load_detector, score_images  # noqa: E402

OUT = HERE / "results"


def main() -> None:
    rows = list(csv.DictReader(open(OUT / "split.csv", newline="", encoding="utf-8")))
    data = ROOT / "data" / "visa"
    scored = [r for r in rows if r["role"] != "fit"]
    lat_paths = [data / r["image"] for r in rows if r["role"] == "test_n"][:50]
    for name in ("m", "m_reg"):
        cfg_path = HERE / f"config_{name}.yaml"
        if not cfg_path.exists():
            print(f"{name}: not run (config absent: gate failed)")
            continue
        cfg = load_config(cfg_path)
        fit(cfg)
        det = load_detector(cfg)
        s = score_images(det, [data / r["image"] for r in scored])
        with open(OUT / f"scores_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["image", "label", "role", "score"])
            for r, v in zip(scored, s):
                w.writerow([r["image"], r["label"], r["role"], f"{v:.6f}"])
        lat = _measure_latency(det, lat_paths, warmup=5)
        (OUT / f"latency_{name}.json").write_text(json.dumps(lat, indent=1), encoding="utf-8")
        print(name, "scored", len(s), "latency median ms", round(lat["median_ms"], 1))
        del det


if __name__ == "__main__":
    main()
