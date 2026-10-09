"""Website and executive PDF: build gates, EN/zh-TW metric parity, images, wording rules, visual-QA evidence."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PDF = ROOT / "reports" / "executive_summary.pdf"
NUM = re.compile(r"[−+]?\d+(?:[.,]\d+)*%?")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    sys.path.insert(0, str(SITE))
    import build  # noqa: E402  (site/build.py)
    dist = tmp_path_factory.mktemp("dist")
    f = build.build(dist)
    pages = {lang: (dist / p / "index.html").read_text(encoding="utf-8") for lang, p in (("en", ""), ("zh", "zh-tw"))}
    return dist, f, pages


def _text(html: str) -> str:
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    html = re.sub(r"<title>.*?</title>|<desc\b.*?</desc>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _main(html: str) -> str:
    return html.split('<main id="main">', 1)[1].split("</main>", 1)[0]


def test_both_languages_show_the_same_numbers(built):
    _, _, pages = built
    en, zh = (Counter(NUM.findall(_text(_main(pages[k])))) for k in ("en", "zh"))
    assert en == zh, {"only_en": dict(en - zh), "only_zh": dict(zh - en)}
    assert sum(en.values()) > 250


def test_no_placeholders_or_template_leaks(built):
    _, _, pages = built
    for lang, html in pages.items():
        txt = _text(html)
        for bad in ("{{", "}}", "{%", "TODO", "TBD", "lorem", "undefined", "NaN", "[object"):
            assert bad not in txt, (lang, bad)


def test_images_exist_at_published_relative_paths(built):
    dist, _, pages = built
    for lang, html in pages.items():
        base = dist if lang == "en" else dist / "zh-tw"
        srcs = re.findall(r'<img [^>]*src="([^"]+)"', html)
        assert len(srcs) == 5, (lang, srcs)
        for tag in re.findall(r"<img [^>]+>", html):
            assert re.search(r'width="\d+"', tag) and re.search(r'height="\d+"', tag) and 'alt="' in tag, tag
            assert 'loading="lazy"' not in tag and 'decoding="async"' not in tag, tag
        for s in srcs:
            assert (base / s).resolve().is_file(), (lang, s)
    assert (dist / "executive-summary.pdf").is_file()


def test_first_screen_message_and_disclosure_link(built):
    _, _, pages = built
    for lang, html in pages.items():
        hero = html.split('id="summary"', 1)[1].split("</section>", 1)[0]
        assert 'href="#pcb3-repair"' in hero and 'id="pcb3-repair"' in html, lang
        assert 'href="#pcb3"' in hero
        # PCB2 controlled comparison is the headline card; PCB3 is a secondary line
        assert hero.index("metric-card") > hero.index("secondary-finding")
    assert "No manufacturing pilot recommended yet." in pages["en"]
    assert "目前不建議進行製造試產" in pages["zh"]


def test_no_cross_category_progress_arrows(built):
    _, f, pages = built
    recalls = {b["key"]: b["recall"] for b in f["boards"]}
    assert recalls["pcb1_v1"] == "72%" and recalls["pcb2_v2"] == "80%" and recalls["pcb3_base"] == "90%"
    for lang, html in pages.items():
        t = _text(html)
        assert not re.search(r"72%\s*(→|->|to)\s*80%", t), lang
        assert not re.search(r"80%\s*(→|->|to)\s*90%", t), lang
        table = html.split('class="grid-table boards-table"', 1)[1].split("</table>", 1)[0]
        assert "→" not in table and "&rarr;" not in table
        assert table.count("<tr") == 1 + 6  # header + one row per (board type, version/arm)


def test_timeline_has_dates_only_and_pending_reviews(built):
    _, _, pages = built
    for lang, html in pages.items():
        tl = html.split('class="timeline"', 1)[1].split("</ol>", 1)[0]
        assert not re.search(r"\b\d{2}:\d{2}\b", tl), lang
        assert tl.count('class="pending"') == 2
        assert len(re.findall(r"<li", tl)) == 5


def test_pcb3_two_decisions_and_acceptance_for_both(built):
    _, f, pages = built
    en = _text(pages["en"])
    assert "ROBUSTNESS IMPROVEMENT SUPPORTED" in en and "weakly informative" in en
    assert "Do not adopt the lighting correction" in en
    assert "does not approve the baseline" in en
    acc = f["pcb3"]["detail"]["acceptance"]
    assert acc["base"]["verdict"] == acc["cand"]["verdict"] == "INCONCLUSIVE"
    assert (acc["base"]["worst_pid"], acc["base"]["worst_rise"]) == ("B1", "+15.2")
    assert (acc["cand"]["worst_pid"], acc["cand"]["worst_rise"]) == ("blur", "+14.5")
    for lang, html in pages.items():
        assert html.count('class="verdict-row"') == 1, lang


def test_pdf_two_pages_and_states_the_decisions():
    if not PDF.exists():
        pytest.skip("executive PDF not rendered")
    from pypdf import PdfReader
    r = PdfReader(str(PDF))
    assert len(r.pages) == 2
    for p in r.pages:
        assert abs(float(p.mediabox.width) - 595) < 2 and abs(float(p.mediabox.height) - 842) < 2  # A4
    t = " ".join(" ".join(p.extract_text() for p in r.pages).split())
    for must in ("No manufacturing pilot recommended yet.", "do not adopt the correction", "weakly informative",
                 "not production-ready", "Not a progression", "7dbd302", "INCONCLUSIVE"):
        assert must in t, must
    assert "{{" not in t and "TODO" not in t


def test_visual_qa_evidence():
    rep = SITE / "qa" / "visual_qa.json"
    if not rep.exists():
        pytest.skip("visual QA not run here (node site/visual_qa.mjs against a served site/dist)")
    from PIL import Image, ImageStat
    d = json.loads(rep.read_text(encoding="utf-8"))
    combos = {(p["page"], p["viewport"]) for p in d["pages"]}
    assert {(l, v) for l in ("en", "zh") for v in ("desktop", "tablet", "mobile")} <= combos
    for p in d["pages"]:
        assert p["ok"] and p["expected"] == p["loaded"] == p["visible"] > 0, (p["page"], p["viewport"])
        for png in [p["firstScreen"], *p["fullPageSegments"]]:
            path = ROOT / png
            if not path.exists():
                pytest.skip("screenshots not present in this copy")
            assert ImageStat.Stat(Image.open(path).convert("L")).stddev[0] > 1, f"blank capture {png}"
