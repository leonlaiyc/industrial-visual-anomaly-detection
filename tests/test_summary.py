"""Executive PDF built from the committed outputs: one page, honest wording, every number traceable."""
import json

import yaml
from pypdf import PdfReader

from ivad import summary
from ivad.config import REPO_ROOT, load_config, resolve_path
from ivad.consistency import PDF_LITERALS, pdf_tokens, traceable_tokens


def _build(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "PDF_PATH", tmp_path / "executive_summary.pdf")
    f = summary.facts()
    reader = PdfReader(str(summary.build_pdf(f)))
    return f, reader, " ".join("\n".join(p.extract_text() for p in reader.pages).split())


def test_pdf_is_one_page_and_states_the_unchanged_recall_and_recommendation(tmp_path, monkeypatch):
    f, reader, text = _build(tmp_path, monkeypatch)
    assert len(reader.pages) == 1
    assert summary.RECOMMENDATION in text
    assert "recall remained 80%" in text and "not demonstrated" in text
    assert "Inconclusive" in text and "does not mean the criteria were met" in text
    for banned in ("minutes", "McNemar", "pre-registered", "conformal", "whitening", "prepared by"):
        assert banned.lower() not in text.lower()
    assert f"Delivery time: {summary.DELIVERY_TIME}" in text


def test_every_pdf_number_traces_to_recorded_outputs(tmp_path, monkeypatch):
    f, _, text = _build(tmp_path, monkeypatch)
    srcs = [f["cmp"], yaml.safe_load((REPO_ROOT / "config" / "acceptance_criteria.yaml").read_text(encoding="utf-8"))]
    for path in summary.RUNS.values():
        cfg = load_config(REPO_ROOT / path)
        rep = resolve_path(cfg, "reports")
        for fn in ("metrics.json", "frozen_operating_point.json", "robustness.json"):
            if (rep / fn).exists():
                srcs.append(json.loads((rep / fn).read_text(encoding="utf-8")))
        srcs.append(cfg["robustness"])
    allowed = traceable_tokens(srcs) | PDF_LITERALS | summary.PDF_FIXED_LABELS
    assert [t for t in pdf_tokens(text) if t not in allowed] == []
