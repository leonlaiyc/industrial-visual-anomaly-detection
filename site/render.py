"""Headless-Chrome rendering for QA screenshots and the executive PDF.

    python site/render.py shots  [--base http://127.0.0.1:8765] [--out site/qa]
    python site/render.py pdf    [--base http://127.0.0.1:8765]   # also copied to reports/executive_summary.pdf

Serve site/dist first (e.g. `python -m http.server 8765 --directory site/dist`).
Uses a throwaway browser profile, never the user's own.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "google-chrome", "chromium", "chromium-browser", "msedge"]
VIEWPORTS = {"desktop": (1440, 900, 1), "tablet": (820, 1180, 1), "mobile": (390, 844, 2)}


def browser() -> str:
    for c in CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("no Chrome/Edge found for headless rendering")


def run(args: list[str]) -> None:
    with tempfile.TemporaryDirectory() as prof:
        cmd = [browser(), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               "--hide-scrollbars", f"--user-data-dir={prof}", "--virtual-time-budget=4000", *args]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)


def screenshot(url: str, out: Path, w: int, h: int, scale: int = 1) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run([f"--window-size={w},{h}", f"--force-device-scale-factor={scale}", f"--screenshot={out}", url])
    return out


def pdf(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["--no-pdf-header-footer", "--print-to-pdf-no-header", f"--print-to-pdf={out}", url])
    return out


def shots(base: str, out_dir: Path, full_height: int = 0) -> list[Path]:
    made = []
    for lang, path in (("en", ""), ("zh", "zh-tw/")):
        for name, (w, h, s) in VIEWPORTS.items():
            made.append(screenshot(f"{base}/{path}", out_dir / f"{lang}_{name}_first.png", w, h, s))
            if full_height:
                made.append(screenshot(f"{base}/{path}", out_dir / f"{lang}_{name}_full.png", w, full_height, 1))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["shots", "pdf"])
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(HERE / "qa"))
    ap.add_argument("--full", type=int, default=0, help="also capture a tall window of this height")
    a = ap.parse_args()
    if a.what == "shots":
        for p in shots(a.base.rstrip("/"), Path(a.out), a.full):
            print(p)
    else:
        out = pdf(f"{a.base.rstrip('/')}/print/", HERE / "dist" / "executive-summary.pdf")
        shutil.copyfile(out, HERE.parent / "reports" / "executive_summary.pdf")  # the one executive PDF
        print(out, "-> reports/executive_summary.pdf")
