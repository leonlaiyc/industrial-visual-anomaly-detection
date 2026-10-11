"""Build the static bilingual case-study site (and its print page) from facts + content.

    python site/build.py            # -> site/dist/ (English at /, Traditional Chinese at /zh-tw/)

Inputs
  site/facts.py               the only source of numbers (reads recorded project outputs)
  site/content/{en,zh-TW}.yaml  prose; numbers only as {{ f.<path> }} references
  site/templates/*.j2         page structure, shared by both languages
  site/static/                CSS and images (copied as-is)

Gates (the build fails on any of them)
  * a content string contains a literal number that is not an allowed label (`lint_content`);
  * the two languages do not define the same content keys;
  * a template references a missing content key or fact (StrictUndefined).
No JavaScript is required to read or navigate the site; the language switch is a plain link.
"""
from __future__ import annotations

import re
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import facts as facts_mod  # noqa: E402
import release_links  # noqa: E402

DIST = HERE / "dist"
LANGS = {"en": {"dir": "", "html_lang": "en", "label": "EN"},
         "zh-TW": {"dir": "zh-tw/", "html_lang": "zh-Hant-TW", "label": "繁體中文"}}

# Tokens with digits that are names, not measurements (board types, versions, criteria IDs, units).
ALLOWED_TOKENS = {"PCB1", "PCB2", "PCB3", "PCB4", "v1", "v2", "v3", "P90", "P95", "P99", "P1", "P2", "P3",
                  "P4", "P5", "A4", "A0", "A1", "B1", "G1", "N1", "P0", "ResNet-50-2", "Wide-ResNet-50-2",
                  "WideResNet-50-2", "layer2", "layer3", "B2", "B3", "B4", "A2", "A3",
                  "4.0",  # CC BY licence version
                  "2022",  # dataset citation year (Zou et al., ECCV 2022)
                  "YOLOv8", "YOLOv8s", "YOLO11", "YOLO11s",  # model names (exploratory study)
                  "AGPL-3.0", "mAP50", "D5"}  # licence/model names and deviation identifier
_EXPR = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.S)
_TOKEN = re.compile(r"[A-Za-z\-]*\d[\w.\-]*")


def lint_content(content: dict[str, Any], lang: str) -> list[str]:
    """Literal numbers in prose are forbidden: they must come from facts."""
    bad = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k not in ("image", "w", "h"):
                    walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            text = re.sub(r"<[^>]+>", " ", _EXPR.sub(" ", node))
            for tok in _TOKEN.findall(text):
                if tok.strip(".-") not in ALLOWED_TOKENS:
                    bad.append(f"{lang}{path}: literal '{tok}'")
    walk(content, "")
    return bad


def keys(node, prefix="") -> set[str]:
    if isinstance(node, dict):
        return set().union(*[keys(v, f"{prefix}.{k}") for k, v in node.items()]) | {prefix}
    if isinstance(node, list):
        return {prefix}
    return {prefix}


def render_content(node, env: Environment, ctx: dict[str, Any]):
    """Render every content string as a template against the facts; mark it safe (authored HTML)."""
    if isinstance(node, dict):
        return {k: render_content(v, env, ctx) for k, v in node.items()}
    if isinstance(node, list):
        return [render_content(v, env, ctx) for v in node]
    if isinstance(node, str):
        return Markup(env.from_string(node).render(**ctx).strip())
    return node


# --------------------------------------------------------------------------- #
# Historical PCB2 unit-chart helper (the portfolio uses parallel workflow cards)
# --------------------------------------------------------------------------- #
def unit_chart(f: dict[str, Any], t: dict[str, Any]) -> Markup:
    """100 normal + 100 defective test images per version; each square is one image.

    Counts come from the recorded confusion matrices; nothing is typed in.
    """
    cell, gap, cols = 9, 2, 20
    rows = 5
    gw = cols * (cell + gap) - gap
    gh = rows * (cell + gap) - gap
    label_w, pad, col_gap, head = 34, 0, 28, 30
    W = label_w + 2 * gw + col_gap
    H = head + 2 * gh + 22
    parts = []

    def grid(x0, y0, n_hi, n, hi_cls, lo_cls):
        out = []
        for i in range(n):
            r, c = divmod(i, cols)
            cls = hi_cls if i < n_hi else lo_cls
            x, y = x0 + c * (cell + gap), y0 + r * (cell + gap)
            if cls == "u-miss":
                out.append(f'<rect class="{cls}" x="{x + .75}" y="{y + .75}" width="{cell - 1.5}" '
                           f'height="{cell - 1.5}" rx="1.5"/>')
            else:
                out.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="1.5"/>')
        return "".join(out)

    x_norm, x_def = label_w, label_w + gw + col_gap
    parts.append(f'<text class="u-head" x="{x_norm}" y="12">{t["normal_head"]}</text>')
    parts.append(f'<text class="u-head" x="{x_def}" y="12">{t["defect_head"]}</text>')
    for k, v in enumerate(("v1", "v2")):
        c = f["pcb2"][v]["confusion"]
        y0 = head + k * (gh + 22) - 10
        parts.append(f'<text class="u-row" x="0" y="{y0 + gh / 2 + 4}">{v}</text>')
        parts.append(grid(x_norm, y0, c["FP"], c["FP"] + c["TN"], "u-fa", "u-ok"))
        parts.append(grid(x_def, y0, c["TP"], c["TP"] + c["FN"], "u-hit", "u-miss"))
    svg = (f'<svg class="unit-chart" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-labelledby="uc-title uc-desc" preserveAspectRatio="xMinYMin meet">'
           f'<title id="uc-title">{t["title"]}</title><desc id="uc-desc">{t["desc"]}</desc>'
           + "".join(parts) + "</svg>")
    return Markup(svg)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def load_content() -> dict[str, dict[str, Any]]:
    out = {}
    for lang in LANGS:
        out[lang] = yaml.safe_load((HERE / "content" / f"{lang}.yaml").read_text(encoding="utf-8"))
    return out


def build(dist: Path = DIST, *, release: bool = False, links_path: Path = release_links.DEFAULT) -> dict[str, Any]:
    links = release_links.load(links_path, release=release)
    f = facts_mod.write()
    content = load_content()
    errors = []
    for lang, c in content.items():
        errors += lint_content(c, lang)
    ken, kzh = keys(content["en"]), keys(content["zh-TW"])
    if ken != kzh:
        errors.append(f"content keys differ: only en {sorted(ken - kzh)[:8]}, only zh {sorted(kzh - ken)[:8]}")
    if errors:
        raise SystemExit("content lint failed:\n  " + "\n  ".join(errors))

    env = Environment(loader=FileSystemLoader(HERE / "templates"), undefined=StrictUndefined,
                      autoescape=True, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    if dist.resolve() in (HERE.resolve(), HERE.parent.resolve()) or HERE.resolve() not in dist.resolve().parents:
        # Test outputs may live outside site/, but never delete a source/project root.
        if dist.resolve() in (HERE.resolve(), HERE.parent.resolve()) or (dist.exists() and any(dist.iterdir())):
            raise ValueError("output must be a build directory, not a source tree")
    if dist.exists():
        shutil.rmtree(dist)
    (dist / "assets" / "img").mkdir(parents=True)
    for filename in ("site.css", "favicon.svg"):
        shutil.copyfile(HERE / "static" / filename, dist / "assets" / filename)
    images = {"workflow_caught.jpg"} | {item["image"] for item in content["en"]["findings"]["items"]}
    for filename in sorted(images):
        shutil.copyfile(HERE / "static" / "img" / filename, dist / "assets" / "img" / filename)
    pages = [("index.html.j2", "index.html")]
    for lang, meta in LANGS.items():
        t = render_content(content[lang], env, {"f": f, "links": links})
        root = "../" if meta["dir"] else ""
        alt = {l: ("../" if meta["dir"] else "") + m["dir"] for l, m in LANGS.items()}
        for tpl, out_name in pages:
            html = env.get_template(tpl).render(
                t=t, f=f, lang=lang, meta=meta, langs=LANGS, root=root, alt=alt, page=out_name,
                links=links)
            out = dist / meta["dir"] / out_name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8", newline="\n")
    # One recipient-facing A4 project sheet. The historical extended brief is source evidence only.
    t_en = render_content(content["en"], env, {"f": f, "links": links})
    for tpl, src, out_dir in (("print_one_page.html.j2", "print_one_page.en.yaml", dist / "print" / "one-page"),):
        p = yaml.safe_load((HERE / "content" / src).read_text(encoding="utf-8"))
        bad = lint_content(p, src)
        if bad:
            raise SystemExit("content lint failed:\n  " + "\n  ".join(bad))
        html = env.get_template(tpl).render(p=render_content(p, env, {"f": f, "links": links}), t=t_en, f=f, links=links,
                                            )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(html, encoding="utf-8", newline="\n")
    (dist / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8", newline="\n")
    (dist / "release-link-status.json").write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")
    (dist / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(HERE.parent / "NOTICE.md", dist / "NOTICE.md")
    # Rebuilding wipes dist/, so carry over the canonical sheet until re-rendered.
    for rep_name, dist_name in (("executive-summary-one-page.pdf", "executive-summary-one-page.pdf"),):
        pdf = HERE.parent / "reports" / rep_name
        if pdf.exists():
            shutil.copyfile(pdf, dist / dist_name)
        else:
            print(f"warning: reports/{rep_name} missing; run `python site/render.py pdf`")
    print(f"site built -> {dist}")
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true", help="require actual anonymously accessible public URLs")
    ap.add_argument("--links", type=Path, default=release_links.DEFAULT)
    ap.add_argument("--out", type=Path, default=DIST)
    args = ap.parse_args()
    try:
        build(args.out, release=args.release, links_path=args.links)
    except (ValueError, OSError) as error:
        raise SystemExit(f"release link gate failed: {error}") from error
