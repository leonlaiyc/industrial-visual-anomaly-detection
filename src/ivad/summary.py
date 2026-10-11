"""README results block and the one-page executive PDF, generated from recorded outputs only.

Runs shown (each with its own config and reports directory):

    PCB1 v1   config/default.yaml   original v1 result (PCB1 is the discovery data for v2)
    PCB2 v1   config/pcb2_v1.yaml   baseline on the prospective validation category
    PCB2 v2   config/pcb2_v2.yaml   board-ROI aggregation, same data and model

Every number printed here is read from metrics.json, robustness.json,
frozen_operating_point.json or the paired comparison.json; `ivad check` traces
each numeric token in the PDF back to those files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

from .acceptance import acceptance_status, load_criteria
from .compare import comparison_dir
from .config import REPO_ROOT, load_config, resolve_path

RUNS = {"PCB1 v1": "config/default.yaml", "PCB2 v1": "config/pcb2_v1.yaml", "PCB2 v2": "config/pcb2_v2.yaml"}
BASE, NEW = "PCB2 v1", "PCB2 v2"
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
# The current one-page project sheet is printed from the website and checked by `ivad check` C5/C8.
PDF_ONE_PAGE = REPO_ROOT / "reports" / "executive-summary-one-page.pdf"
EXECUTIVE_PDFS = {PDF_ONE_PAGE: 1}  # the extended brief is retired historical evidence
# The earlier ReportLab one-page PCB2 PDF (`build_summary(legacy_pdf=True)`), for comparison only.
PDF_PATH = REPO_ROOT / "reports" / "executive_summary_legacy_pcb2.pdf"

# The one conclusion every summary document states verbatim (`ivad check` C8 verifies it).
RECOMMENDATION = ("Although v2 improved false-positive performance, it did not meet the proposed "
                  "detection-sensitivity and robustness requirements. It is therefore not recommended for a "
                  "manufacturing pilot without further work.")
# Non-measurement numbers printed in the PDF: chart axis ticks and the project period.
PDF_FIXED_LABELS = {"0", "20", "30", "40", "60", "80", "100", "2026"}
PROJECT_PERIOD = "October 2026"
# Left blank on purpose until the project owner has reviewed and accepted the deliverables.
DELIVERY_TIME = "pending owner review"


def _j(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def facts() -> dict[str, Any]:
    acc = load_criteria()
    runs = {}
    for name, path in RUNS.items():
        cfg = load_config(REPO_ROOT / path)
        rep = resolve_path(cfg, "reports")
        m, fz = _j(rep / "metrics.json"), _j(rep / "frozen_operating_point.json")
        rb = _j(rep / "robustness.json") if (rep / "robustness.json").exists() else None
        pr = fz["primary"]
        runs[name] = {"cfg": cfg, "m": m, "fz": fz, "rb": rb, "pr": pr,
                      "op": m["official_test"]["operating_points"][pr],
                      "hold": m["supplementary_holdout"]["operating_points"][pr]["fpr"],
                      "auroc": m["official_test"]["auroc"], "lat": m["latency"],
                      "status": acceptance_status(m, rb, pr, acc)}
    cmp_ = _j(comparison_dir(runs[NEW]["cfg"]) / "comparison.json")
    return {"runs": runs, "cmp": cmp_, "acc": acc}


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def rate(r: dict, nd: int = 0) -> str:
    return f"{pct(r['value'], nd)} ({r['x']}/{r['n']})"


def ci(r: dict, nd: int = 0) -> str:
    return f"[{pct(r['ci_low'], nd)}, {pct(r['ci_high'], nd)}]"


def perturbation_label(spec: dict) -> str:
    if spec["kind"] == "photometric":
        return f"{'darker' if spec['brightness'] < 1 else 'brighter'} image (brightness and contrast x{spec['brightness']})"
    if spec["kind"] == "blur":
        return f"blur (sigma {spec['sigma_px']} px)"
    if spec["kind"] == "shift":
        return f"camera shift ({spec['dx_px']}, {spec['dy_px']}) px"
    return spec["name"]


def verdict_text(st: dict) -> str:
    unmet = [lab for k, lab in (("recall", "recall"), ("fpr_test", "test FPR"), ("fpr_holdout", "holdout FPR"),
                                ("auroc", "AUROC"), ("latency", "latency"), ("robustness", "robustness"))
             if st[k] is False]
    if st["verdict"] == "NO-GO":
        return "NO-GO (" + ", ".join(st["no_go_triggers"]).replace("_", " ") + ")"
    if st["verdict"] == "GO":
        return "all proposed criteria met"
    return "not met: " + ", ".join(unmet) if unmet else st["verdict"]


# --------------------------------------------------------------------------- #
# README block
# --------------------------------------------------------------------------- #
def readme_block(f: dict | None = None) -> str:
    f = f or facts()
    R, tg = f["runs"], f["acc"]["performance"]
    names = list(RUNS)
    hdr = "| Official VisA test split (100 normal + 100 defective) | Proposed target | " + " | ".join(
        f"{n}{' (original)' if n == 'PCB1 v1' else ''}" for n in names) + " |"
    rows = [
        ("Recall (defects sent to review)", f">= {pct(tg['recall_min'])}",
         [f"{rate(R[n]['op']['recall'])} {ci(R[n]['op']['recall'])}" for n in names]),
        ("False-alarm rate, official test normals", f"<= {pct(tg['fpr_max'])}",
         [f"{rate(R[n]['op']['fpr'])} {ci(R[n]['op']['fpr'])}" for n in names]),
        ("False-alarm rate, supplementary normal holdout", f"<= {pct(tg['fpr_max'])}",
         [f"{rate(R[n]['hold'], 1)} {ci(R[n]['hold'], 1)}" for n in names]),
        ("Image AUROC", f">= {tg['auroc_min']:.2f}",
         [f"{R[n]['auroc']['value']:.3f} [{R[n]['auroc']['ci_low']:.3f}, {R[n]['auroc']['ci_high']:.3f}]" for n in names]),
        ("Median latency per image (GPU)", f"<= {tg['latency_ms_max']:.0f} ms",
         [f"{R[n]['lat']['median_ms']:.0f} ms" for n in names]),
        ("TP / FN / FP / TN at the frozen P95 threshold", "",
         [" / ".join(str(R[n]['op']['confusion'][k]) for k in ("TP", "FN", "FP", "TN")) for n in names]),
        ("Status against the proposed criteria", "", [verdict_text(R[n]["status"]) for n in names]),
    ]
    lines = [README_START, hdr, "|---|---|" + "---|" * len(names)]
    lines += [f"| {a} | {b} | " + " | ".join(c) + " |" for a, b, c in rows]
    p = f["cmp"]["paired"]
    r, ft, fh, au = p["recall_test_anomalies"], p["fpr_test_normals"], p["fpr_holdout_normals"], p["auroc"]
    lines += ["", f"Paired PCB2 comparison (v2 − v1, same images, each at its own frozen threshold): "
              f"recall {100 * r['delta']:+.0f} pts [{100 * r['ci_low']:+.0f}, {100 * r['ci_high']:+.0f}] "
              f"(McNemar p = {r['p_value']:.2f}); test false alarms {100 * ft['delta']:+.0f} pts "
              f"[{100 * ft['ci_low']:+.0f}, {100 * ft['ci_high']:+.0f}] (p = {ft['p_value']:.3f}); "
              f"holdout false alarms {100 * fh['delta']:+.1f} pts (p = {fh['p_value']:.2f}); AUROC "
              f"{au['delta']:+.3f} [{au['ci_low']:+.3f}, {au['ci_high']:+.3f}]. "
              f"Pre-registered outcome on the primary endpoint (recall): **{f['cmp']['decision']['outcome'].lower()}**.",
              README_END]
    return "\n".join(lines)


def sync_readme(f: dict | None = None) -> None:
    p = REPO_ROOT / "README.md"
    s = p.read_text(encoding="utf-8")
    if README_START not in s:
        return
    a, b = s.index(README_START), s.index(README_END) + len(README_END)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s[:a] + readme_block(f) + s[b:])


# --------------------------------------------------------------------------- #
# Executive PDF (one A4 page for a non-specialist reader)
# --------------------------------------------------------------------------- #
INK, MUTED, ACCENT, RULE = "#1b1f24", "#5b6470", "#1f3a5f", "#cfd5dc"
BASE_C, MET_C, UNMET_C = "#98a2ad", "#2f6b4f", "#8a5300"
MARGIN_MM = 19


def _comparison_chart(R: dict, tg: dict, width: float):
    """Vector dot-and-range chart: recall and test false-alarm rate, v1 vs v2, with target zones."""
    from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    C = colors.HexColor
    panels = [("Defects flagged for review (recall)", "recall", 100, (0, 20, 40, 60, 80, 100),
               (100 * tg["recall_min"], 100)),
              ("False alarms, official-test normal images", "fpr", 30, (0, 10, 20, 30),
               (0, 100 * tg["fpr_max"]))]
    gap, lab_w, h = 9 * mm, 7 * mm, 24.5 * mm
    pw = (width - gap) / 2
    d = Drawing(width, h)
    for i, (title, key, xmax, ticks, zone) in enumerate(panels):
        left = i * (pw + gap)
        x0, aw = left + lab_w, pw - lab_w - 3 * mm

        def X(v: float) -> float:
            return x0 + aw * v / xmax

        top, ax_y = h - 3 * mm, 5.5 * mm
        d.add(String(left, top, title, fontName="Helvetica-Bold", fontSize=7.6, fillColor=C(INK)))
        d.add(Rect(X(zone[0]), ax_y, X(min(zone[1], xmax)) - X(zone[0]), top - 3.5 * mm - ax_y,
                   fillColor=C("#e9eef4"), strokeColor=None))
        d.add(Line(x0, ax_y, x0 + aw, ax_y, strokeColor=C(RULE), strokeWidth=0.6))
        for t in ticks:
            d.add(Line(X(t), ax_y, X(t), ax_y - 1 * mm, strokeColor=C(RULE), strokeWidth=0.6))
            d.add(String(X(t), ax_y - 4 * mm, f"{t}%", fontName="Helvetica", fontSize=6.4, fillColor=C(MUTED),
                         textAnchor="middle"))
        for j, (name, lab, col) in enumerate(((BASE, "v1", BASE_C), (NEW, "v2", ACCENT))):
            r = R[name]["op"][key]
            y = top - 8 * mm - j * 6.2 * mm
            d.add(String(left, y - 2.4, lab, fontName="Helvetica-Bold", fontSize=7.6, fillColor=C(col)))
            d.add(Line(X(100 * r["ci_low"]), y, X(100 * r["ci_high"]), y, strokeColor=C(col), strokeWidth=1.6))
            d.add(Circle(X(100 * r["value"]), y, 2.2, fillColor=C(col), strokeColor=colors.white, strokeWidth=0.6))
            d.add(String(X(100 * r["ci_high"]) + 1.6 * mm, y - 2.4, pct(r["value"]), fontName="Helvetica-Bold",
                         fontSize=7.4, fillColor=C(col)))
    return d


def build_pdf(f: dict | None = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    f = f or facts()
    R, c, tg = f["runs"], f["cmp"], f["acc"]["performance"]
    b, n = R[BASE], R[NEW]
    if c["decision"]["outcome"] == "IMPROVEMENT SUPPORTED":
        raise ValueError("the executive text states that recall did not improve; revise it before rebuilding")
    # Helvetica throughout; DejaVu Sans (bundled with Matplotlib) only for the arrow glyph.
    pdfmetrics.registerFont(TTFont("Sym", str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")))
    arrow = f'<font name="Sym" size="11" color="{MUTED}">&#8594;</font>'
    C = colors.HexColor

    M = MARGIN_MM * mm
    W = A4[0] - 2 * M
    LAB = 36 * mm  # left label column
    CW = W - LAB
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=12.8, textColor=C(INK))
    lead = ParagraphStyle("lead", parent=body, fontSize=10.2, leading=14.2)
    small = ParagraphStyle("small", parent=body, fontSize=7.7, leading=10.2, textColor=C(MUTED))
    label = ParagraphStyle("label", parent=body, fontName="Helvetica-Bold", fontSize=8.6, leading=11.5,
                           textColor=C(ACCENT))
    over = ParagraphStyle("over", parent=small, fontName="Helvetica-Bold", fontSize=7.2)
    title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=21, leading=25)
    sub = ParagraphStyle("sub", parent=body, fontSize=12.5, leading=16, textColor=C(ACCENT))
    tl = ParagraphStyle("tl", parent=small, fontSize=7.5, leading=9.8)
    tv = ParagraphStyle("tv", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=19)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11.6)
    pad0 = [("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP")]

    def section(name: str, flow: list, space: float = 4.6 * mm) -> list:
        t = Table([[Paragraph(name, label), flow]], colWidths=[LAB, CW])
        t.setStyle(TableStyle(pad0 + [("RIGHTPADDING", (0, 0), (0, -1), 4 * mm)]))
        return [KeepTogether([t]), Spacer(1, space)]

    def status(ok: bool) -> str:
        return f'<font color="{MET_C}"><b>Met</b></font>' if ok else f'<font color="{UNMET_C}"><b>Not met</b></font>'

    rec_b, rec_n = b["op"]["recall"], n["op"]["recall"]
    fp_b, fp_n = b["op"]["fpr"], n["op"]["fpr"]
    hb, hn = b["hold"], n["hold"]
    rb = n["rb"]
    dark = next(v for v in rb["perturbations"].values()
                if v["spec"]["kind"] == "photometric" and v["spec"]["brightness"] < 1)
    dark_pct = round(100 * (1 - dark["spec"]["brightness"]))
    st = n["status"]

    # A. Header
    rule = Table([[""]], colWidths=[W], rowHeights=[0.1])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.8, C(ACCENT))]))
    S: list = [
        Paragraph("INDEPENDENT PROJECT &nbsp;&middot;&nbsp; PUBLIC BENCHMARK DATA", over), Spacer(1, 1.5 * mm),
        Paragraph("Industrial Visual Anomaly Detection", title),
        Paragraph("From Requirements to a Tested Prototype", sub), Spacer(1, 1 * mm),
        Paragraph("Flags printed-circuit-board images for manual review. Built on the public VisA benchmark "
                  "(Amazon Science, CC BY 4.0) with AI coding assistance; no factory data was used.", small),
        Spacer(1, 3 * mm), rule, Spacer(1, 5 * mm)]

    # B. Executive summary
    S += section("Executive summary", [Paragraph(
        "Developed and evaluated an AI-assisted visual inspection prototype using public printed-circuit-board images. "
        f"A targeted refinement reduced false alarms from {pct(fp_b['value'])} to {pct(fp_n['value'])} on a board type "
        f"that was held back and evaluated independently, while defect recall remained {pct(rec_n['value'])}. "
        f"The prototype is below the proposed {pct(tg['recall_min'])} recall target and remains sensitive to darker "
        "images. Further improvement is required before a manufacturing pilot.", lead)])

    # C. Results at a glance
    def tile(name: str, v1: str, v2: str, cap: str) -> list:
        return [Paragraph(name, tl), Paragraph(f'<font color="{BASE_C}">{v1}</font> {arrow} {v2}', tv),
                Paragraph(cap, tl)]

    tiles = Table([[
        tile("Defects flagged (recall)", pct(rec_b["value"]), pct(rec_n["value"]),
             f"of {rec_n['n']} defective images<br/>target {pct(tg['recall_min'])}: {status(st['recall'])}"),
        tile("False-alarm rate", pct(fp_b["value"]), pct(fp_n["value"]),
             f"of {fp_n['n']} normal images<br/>target {pct(tg['fpr_max'])} or less: {status(st['fpr_test'])}"),
        tile("Ranking quality (AUROC)", f"{b['auroc']['value']:.3f}", f"{n['auroc']['value']:.3f}",
             f"1.0 = perfect separation<br/>target {tg['auroc_min']:.2f}: {status(st['auroc'])}"),
        tile("Median time per image", f"{b['lat']['median_ms']:.0f}", f"{n['lat']['median_ms']:.0f}",
             f"milliseconds, one GPU<br/>target {tg['latency_ms_max'] / 1000:.0f} s or less: {status(st['latency'])}"),
    ]], colWidths=[CW / 4] * 4)
    tiles.setStyle(TableStyle(pad0 + [("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                      ("LINEABOVE", (0, 0), (-1, 0), 0.6, C(RULE))]))
    S += section("Results at a glance", [
        Paragraph(f"Board type PCB2, official test set: {rec_n['n']} defective and {fp_n['n']} normal images. "
                  "v1 = first version, v2 = refined version; review thresholds fixed before testing.", small),
        Spacer(1, 2 * mm), tiles, Spacer(1, 3.5 * mm), _comparison_chart(R, tg, CW), Spacer(1, 1.5 * mm),
        Paragraph(f"Shaded: proposed target. Lines: 95% confidence ranges ({pct(rec_n['value'])} recall on "
                  f"{rec_n['n']} defective images is compatible with roughly {pct(rec_n['ci_low'])} to "
                  f"{pct(rec_n['ci_high'])}). A separate check on {hn['n']} further normal images, outside the official "
                  f"test, gave false-alarm rates of {pct(hb['value'], 1)} (v1) and {pct(hn['value'], 1)} (v2).", small)])

    # D. What changed and why
    S += section("What changed and why", [Paragraph(
        "The first version sometimes treated background texture, such as dust on the cloth under the board, as a "
        "defect. The second version focuses anomaly scoring on the board region, reducing false alarms without "
        "improving missed-defect detection; most remaining misses are very small defects. The change was fixed in "
        "advance and tested once on a held-back board type.", body)])

    # E. Reliability and go / no-go
    ge, le = '<font name="Sym">&#8805;</font>', '<font name="Sym">&#8804;</font>'
    rows = [[Paragraph(f"Recall {ge} {pct(tg['recall_min'])}", cell),
             Paragraph(f"{status(st['recall'])} &nbsp;&middot;&nbsp; {pct(rec_n['value'])} of defects flagged", cell)],
            [Paragraph(f"False alarms {le} {pct(tg['fpr_max'])}", cell),
             Paragraph(f"{status(st['fpr_test'] and st['fpr_holdout'])} &nbsp;&middot;&nbsp; {pct(fp_n['value'])} "
                       f"official test; {pct(hn['value'], 1)} separate check", cell)],
            [Paragraph("Lighting robustness", cell),
             Paragraph(f"{status(st['robustness'])} &nbsp;&middot;&nbsp; {dark_pct}% darker: false alarms "
                       f"{pct(rb['base']['fpr']['value'])} {arrow} {pct(dark['fpr']['value'])} "
                       f"({rb['base']['fpr']['n']} normal images)", cell)]]
    crit = Table(rows, colWidths=[CW * 0.27, CW * 0.73])
    crit.setStyle(TableStyle(pad0 + [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                     ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
                                     ("LINEABOVE", (0, 0), (-1, -1), 0.5, C(RULE)),
                                     ("LINEBELOW", (0, -1), (-1, -1), 0.5, C(RULE))]))
    rec = Table([[Paragraph(f"<b>Recommendation: not ready for a manufacturing pilot.</b> {RECOMMENDATION}", body)]],
                colWidths=[CW])
    rec.setStyle(TableStyle(pad0 + [("LINEBEFORE", (0, 0), (0, -1), 2.2, C(ACCENT)), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                    ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    S += section("Reliability and go / no-go", [
        crit, Spacer(1, 2.2 * mm),
        Paragraph(f"<b>Formal evaluation</b> (rules fixed before testing): the expected recall gain was not "
                  f"demonstrated. Rule-based verdict: &ldquo;{st['verdict'].capitalize()}&rdquo;. Targets were missed, "
                  "but no automatic stop rule fired, because recall sits exactly at the "
                  f"{pct(f['acc']['no_go']['recall_floor'])} minimum; this does not mean the criteria were met.", body),
        Spacer(1, 2.2 * mm), rec])

    # F. Engineering readiness
    S += section("Engineering readiness", [
        Paragraph("<b>Reproducible evaluation:</b> every number on this page is regenerated from saved per-image results "
                  "and checked automatically. <b>Frozen thresholds</b> were recorded before any test image was scored. "
                  "<b>Automated tests</b>, <b>documented assumptions</b> and <b>engineering handoff</b> notes are part "
                  "of the repository.", body),
        Spacer(1, 2 * mm),
        Paragraph("<b>Limitations and next steps.</b> Two board types from one public benchmark, laboratory images, "
                  f"{rec_n['n']} defective images per board type; no transfer to other products, cameras or factories is "
                  "implied. Next: lighting normalisation, better small-defect sensitivity, and targets confirmed by a "
                  "real manufacturing user, each followed by a fresh, independent evaluation.", body)], space=0)

    def footer(canvas, doc):
        canvas.saveState()
        y = 13 * mm
        canvas.setStrokeColor(C(RULE))
        canvas.setLineWidth(0.6)
        canvas.line(M, y + 4.5 * mm, A4[0] - M, y + 4.5 * mm)
        canvas.setFont("Helvetica", 6.9)
        canvas.setFillColor(C(MUTED))
        canvas.drawString(M, y + 1 * mm, "Data: VisA (Amazon Science), CC BY 4.0. Pretrained network: torchvision "
                          "Wide-ResNet-50-2 (ImageNet weights). Methods, statistics and reproduction:")
        canvas.drawString(M, y - 2.4 * mm, "project repository (README, docs/v2_decision_record.md). No endorsement by "
                          "the dataset authors or any company is implied.")
        canvas.drawString(M, y - 5.8 * mm, f"Project period: {PROJECT_PERIOD}.  Delivery time: {DELIVERY_TIME}.")
        canvas.restoreState()

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=13 * mm,
                            bottomMargin=20 * mm, title="Industrial Visual Anomaly Detection - Executive Summary",
                            subject="PCB image review-support prototype on the public VisA benchmark", author="")
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print(f"pdf: {PDF_PATH}")
    return PDF_PATH


def build_summary(legacy_pdf: bool = False) -> None:
    """README results block. Since the PCB3 experiment the executive PDF is printed from the website
    (`python site/build.py` then `python site/render.py pdf`, which writes the current one-page project sheet);
    `legacy_pdf=True` rebuilds the earlier one-page PCB2 PDF instead, for comparison only."""
    f = facts()
    sync_readme(f)
    if legacy_pdf:
        build_pdf(f)
