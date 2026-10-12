"""Headless-Chrome rendering for QA screenshots and the executive PDF.

    python site/render.py shots  [--base http://127.0.0.1:8765] [--out site/qa]
    python site/render.py pdf    [--base http://127.0.0.1:8765]   # one A4 sheet, also copied to reports/
    python site/render.py pdf-zh [--base http://127.0.0.1:8765]   # actual Chinese website, paginated A4

Serve site/dist first (e.g. `python -m http.server 8765 --directory site/dist`).
Uses a throwaway browser profile, never the user's own.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "google-chrome", "chromium", "chromium-browser", "msedge"]
# print page -> (file in dist/, canonical file in reports/). Retired briefs are not rendered.
PDFS = {"print/one-page/": ("executive-summary-one-page.pdf", "executive-summary-one-page.pdf")}
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


def verify_pdf(path: Path, n_pages: int, links: dict) -> None:
    from pypdf import PdfReader
    r = PdfReader(path)
    if len(r.pages) != n_pages or any(abs(float(p.mediabox.width)-595) > 2 or abs(float(p.mediabox.height)-842) > 2 for p in r.pages):
        raise ValueError(f"{path.name}: expected {n_pages} A4 pages")
    uris = {str(a.get_object().get('/A', {}).get('/URI', '')) for p in r.pages for a in p.get('/Annots', [])} - {''}
    if not uris:
        raise ValueError(f"{path.name}: no clickable URI annotations")
    for k in ('website_url', 'technical_repository_url', 'technical_evidence_url'):
        if links[k] and links[k] not in uris:
            raise ValueError(f"{path.name}: missing clickable {k}")
    import release_links
    if any(not release_links.public_url(u) for u in uris):
        raise ValueError(f"{path.name}: recipient PDF contains a non-public URI")
    if not links['website_url']:
        text = '\n'.join(p.extract_text() for p in r.pages)
        if 'Website not yet deployed.' not in text:
            raise ValueError(f"{path.name}: missing honest undeployed website notice")
    print(path.name, 'verified A4 pages:', n_pages, 'clickable URIs:', sorted(uris))


def chinese_pdf(base: str) -> Path:
    """Print the actual Chinese website; content and figures have one HTML source."""
    from pypdf import PdfReader
    import re
    import unicodedata
    out = HERE / 'dist' / 'project-notes.zh-TW.pdf'
    pdf(base.rstrip('/') + '/zh-tw/', out)
    reader = PdfReader(out)
    if not reader.pages or any(abs(float(p.mediabox.width)-595)>2 or abs(float(p.mediabox.height)-842)>2 for p in reader.pages):
        raise ValueError('Chinese website PDF must contain A4 pages')
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', ''.join(p.extract_text() for p in reader.pages)))
    for phrase in ('以 CNN 與 YOLO', '優化之後，再進行亮度測試', '除了看模型得分', 'CC BY 4.0'):
        if re.sub(r'\s+', '', unicodedata.normalize('NFKC', phrase)) not in text:
            raise ValueError('Chinese PDF missing website content: ' + phrase)
    import release_links
    uris = {str(a.get_object().get('/A', {}).get('/URI', '')) for p in reader.pages for a in p.get('/Annots', [])}
    if any(u and not release_links.public_url(u) for u in uris):
        raise ValueError('Chinese PDF contains a local or non-public URI')
    target = HERE.parent / 'reports' / out.name
    shutil.copyfile(out, target)
    print(target, 'verified A4 pages:', len(reader.pages))
    return target


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
    ap.add_argument("what", choices=["shots", "pdf", "pdf-zh"])
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(HERE / "qa"))
    ap.add_argument("--full", type=int, default=0, help="also capture a tall window of this height")
    ap.add_argument("--links", type=Path, default=HERE / "release_links.json")
    a = ap.parse_args()
    if a.what == "shots":
        for p in shots(a.base.rstrip("/"), Path(a.out), a.full):
            print(p)
    elif a.what == 'pdf-zh':
        chinese_pdf(a.base)
    else:
        links = json.loads((HERE / 'dist' / 'release-link-status.json').read_text(encoding='utf-8'))
        if not links['preview']:
            import release_links
            current = release_links.load(a.links, release=True)
            if any(current[k] != links[k] for k in ('website_url', 'technical_repository_url', 'technical_evidence_url', 'source_mode')):
                raise SystemExit('URL settings changed since release build; rebuild first')
        for page, (dist_name, rep_name) in PDFS.items():
            out = pdf(f"{a.base.rstrip('/')}/{page}", HERE / "dist" / dist_name)
            verify_pdf(out, 1 if 'one-page' in page else 2, links)
            shutil.copyfile(out, HERE.parent / "reports" / rep_name)
            print(out, "-> reports/" + rep_name)
