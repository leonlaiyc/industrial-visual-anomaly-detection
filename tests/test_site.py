"""Portfolio release contracts: narrative, evidence parity, routes, PDFs and privacy."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PDF_ONE = ROOT / "reports" / "executive-summary-one-page.pdf"
NUM = re.compile(r"[−+]?\d+(?:[.,]\d+)*%?")

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    sys.path.insert(0, str(SITE))
    import build
    dist = tmp_path_factory.mktemp("dist")
    f = build.build(dist)
    pages = {lang: (dist / p / "index.html").read_text(encoding="utf-8") for lang, p in (("en", ""), ("zh", "zh-tw"))}
    return dist, f, pages

def _text(html):
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))

def test_bilingual_numbers_and_three_equal_workflows(built):
    _, _, pages = built
    counts = []
    for html in pages.values():
        main = html.split('<main id="main">')[1].split("</main>")[0]
        counts.append(Counter(NUM.findall(_text(main))))
        flow = html.split('class="workflow-grid"')[1].split('</section>')[0]
        assert flow.count('<article class="workflow ') == 3
        assert flow.count('<ol class="steps">') == 3
        assert flow.count('<li>') == 15
    assert counts[0] == counts[1]
    assert counts[0]["13%"] == counts[0]["3%"] == counts[0]["80%"] == 1

def test_routes_assets_and_only_one_recipient_pdf(built):
    dist, _, pages = built
    for lang, html in pages.items():
        base = dist if lang == "en" else dist / "zh-tw"
        for tag in re.findall(r"<img [^>]+>", html):
            assert 'alt="' in tag and re.search(r'width="\d+"', tag) and re.search(r'height="\d+"', tag)
            src = re.search(r'src="([^"]+)"', tag)[1]
            assert (base / src).resolve().is_file()
        for href in re.findall(r'href="([^"]+)"', html):
            if href.startswith("#"):
                assert f'id="{href[1:]}"' in html
            elif not href.startswith("https://"):
                assert (base / href).resolve().is_file(), href
        assert "executive-summary-extended" not in html
        assert "localhost" not in html and "{{" not in html
    assert [p.name for p in dist.glob("*.pdf")] == ["executive-summary-one-page.pdf"]
    assert not (dist / "print" / "index.html").exists()

def test_scientific_scope_and_repair_are_visible(built):
    _, f, pages = built
    en, zh = (_text(pages[k]) for k in ("en", "zh"))
    for phrase in ("separate board datasets", "not a shared test set", "normal images only", "actually trained",
                   "one generic", "defect", "brightness confounding", "unseen defect types", "localization",
                   "does not establish architecture superiority", "candidate was not adopted", "analysis input check was repaired",
                   "No frozen score, threshold or decision changed", "overall acceptance", "AI-assisted"):
        assert phrase.lower() in en.lower(), phrase
    assert f["pcb3"]["result"]["outcome"] == "ROBUSTNESS IMPROVEMENT SUPPORTED"
    assert f["pcb3"]["detail"]["acceptance"]["base"]["verdict"] == "INCONCLUSIVE"
    assert f["pcb3"]["detail"]["acceptance"]["cand"]["verdict"] == "INCONCLUSIVE"
    for phrase in ("影像對位", "瑕疵檢出率", "誤報率", "亮度混淆", "泛化", "定位限制", "AI 輔助", "未採用", "D5"):
        assert phrase in zh, phrase

def test_narrative_is_a_learning_poc(built):
    _, _, pages = built
    for html in pages.values():
        order = [html.index(f'id="{s}"') for s in ("summary", "explorations", "findings", "lessons", "code")]
        assert order == sorted(order)
        assert "<table" not in html
    en = _text(pages["en"])
    assert "I wanted to understand" in en and "A Hands-on Computer Vision PoC" in en
    for bad in ("financial ROI", "Go / No-Go", "shadow-mode", "two days"):
        assert bad not in en

def test_private_source_is_not_claimed_anonymous(built):
    _, _, pages = built
    import release_links
    if release_links.load()["source_mode"] == "private-repository":
        assert "source repository is private" in pages["en"]
        assert "access required" in pages["en"]
        assert "來源庫目前為私人" in pages["zh"]

def test_pdf_self_contained_one_a4_page_and_metrics():
    from pypdf import PdfReader
    assert PDF_ONE.exists()
    r = PdfReader(PDF_ONE)
    assert len(r.pages) == 1
    page = r.pages[0]
    assert abs(float(page.mediabox.width)-595) < 2 and abs(float(page.mediabox.height)-842) < 2
    text = " ".join(page.extract_text().split())
    for phrase in ("CNN Baseline", "Improve & Stress-Test", "Supervised YOLO", "72/100", "13% to 3%", "80%",
                   "98%", "95%", "correction not adopted", "brightness confounding", "localization",
                   "architecture superiority", "What I learned", "VisA", "CC BY 4.0", "AI-assisted",
                   "analysis input check was repaired", "7dbd302", "2026-10-11", "Overall portfolio acceptance"):
        assert phrase in text, phrase
    uris = [str(a.get_object().get("/A", {}).get("/URI", "")) for a in page.get("/Annots", [])]
    assert any("creativecommons.org" in u for u in uris)
    assert not any(u.startswith(("http:", "file:")) for u in uris)
    assert len([v for v in page["/Resources"]["/XObject"].get_object().values()
                if v.get_object().get("/Subtype") == "/Image"]) >= 3

def test_visual_qa_evidence_when_available():
    rep = SITE / "qa" / "final-redesign" / "visual_qa.json"
    if not rep.exists():
        pytest.skip("local browser QA evidence is retained outside the source release")
    from PIL import Image, ImageStat
    d = json.loads(rep.read_text(encoding="utf-8"))
    assert {(l,v) for l in ("en","zh") for v in ("desktop","tablet","mobile")} <= {(p["page"],p["viewport"]) for p in d["pages"]}
    for p in d["pages"]:
        assert p["ok"] and not p["overflowX"] and p["loaded"] == p["visible"] == p["expected"]
        for png in [p["firstScreen"], *p["fullPageSegments"]]:
            assert ImageStat.Stat(Image.open(ROOT/png).convert("L")).stddev[0] > 1

_PRIVATE = ["Jes"+"sie", "Mic"+"ron", "Ed"+"die", "Che"+"ryl"]
def test_no_private_material_in_public_showcase(built):
    dist, _, pages = built
    texts = [*pages.values(), (dist/"print/one-page/index.html").read_text(encoding="utf-8")]
    for t in texts:
        assert all(b.lower() not in t.lower() for b in _PRIVATE)
    assert not (dist/"private_review").exists()
