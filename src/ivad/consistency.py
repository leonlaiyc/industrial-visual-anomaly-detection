"""`ivad check`: automated consistency gates across all runs, the comparison, PDF and README.

C1  per run: metrics.json recomputed independently from the per-image score CSVs
C2  per run: freeze matches config/model/split and precedes evaluation; for runs that require a
    committed freeze, the freeze file was committed before the evaluation timestamp and the
    recorded evaluation HEAD contains the same freeze file. Reported as SKIP (not PASS) when the
    fitted model is absent (fresh clone before `fit`) or the evaluation commit is not in this
    repository's history (curated snapshot).
C3  split manifests pass the leakage audit
C4  paired comparison recomputed from the score CSVs; pairing checks all true
C5  executive PDFs (A4, built from site/ since the PCB3 experiment): the one-page summary has exactly one
    page and the extended brief two; in each, every numeric token traces to a recorded output and the
    headline numbers are present
C6  README results block equals the generated block
C7  repository hygiene: no dataset, weights, large files or secret-like strings tracked (allowed images: the
    website's resized, overlaid VisA derivatives under site/static/img/, and the exploratory study's named
    illustrative figures in EXPLORATORY_FIGS, each < 500 kB and each listed in NOTICE.md)
C8  current summaries distinguish production deployment from authorized shadow validation;
    the frozen decision record and handoff retain their original recommendation
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from . import metrics
from .config import REPO_ROOT, load_config, resolve_path
from .data import audit_split, expected_counts, read_split
from .pipeline import check_freeze, freeze_path, read_scores

# Literal numbers allowed in PDF prose that are not measurements (section numbers, board names).
PDF_LITERALS = {"1", "2", "3", "4", "5", "6", "7", "100", "200", "600", "150", "50", "10", "95",
                "2022", "4.0"}  # dataset citation year and CC BY licence version
# Website images derived from VisA (CC BY 4.0, resized + overlays): the only images allowed in git.
SITE_IMG_DIR, SITE_IMG_MAX_BYTES = "site/static/img/", 500_000
# Downscaled, overlaid PCB4 contact sheets of the exploratory study (VisA derivatives, CC BY 4.0, named in NOTICE.md).
EXPLORATORY_FIGS = ("exploratory/yolo_pcb4/results/fig_failures.jpg", "exploratory/yolo_pcb4/audit/fig_annotation_check.jpg",
                    "exploratory/yolo_pcb4/audit/fig_false_alarms_supplement.jpg")


def _numbers(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)


def traceable_tokens(sources: list[Any]) -> set[str]:
    """All renderings of recorded numbers that a report may legitimately print."""
    out: set[str] = set()
    for v in (x for s in sources for x in _numbers(s)):
        for w in (v, 100 * v, 1000 * v, v / 1000, abs(v), 100 * abs(v)):
            for nd in (0, 1, 2, 3, 4):
                out.add(f"{w:.{nd}f}")
        if v == int(v):
            out.add(str(int(v)))
    return out


def pdf_tokens(text: str) -> list[str]:
    # numbers not glued to letters (skips P95, PCB1, v2 ...)
    return re.findall(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?(?![A-Za-z\d])", text)


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def run_checks() -> bool:
    from .compare import comparison_dir, mcnemar_exact
    from .summary import (BASE, EXECUTIVE_PDFS, NEW, PDF_FIXED_LABELS, RECOMMENDATION, RUNS, README_END, README_START,
                          facts, readme_block)

    results: list[tuple[str, bool | None, str]] = []  # ok None = not verifiable here (reported as SKIP)

    def record(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    def skip(name, detail):
        results.append((name, None, detail))

    cfgs = {n: load_config(REPO_ROOT / p) for n, p in RUNS.items()}
    for name, cfg in cfgs.items():
        rep = resolve_path(cfg, "reports")
        m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
        fz = json.loads((rep / "frozen_operating_point.json").read_text(encoding="utf-8"))
        # C1
        trows, ts = read_scores(rep / "scores_test.csv")
        _, hs = read_scores(rep / "scores_holdout.csv")
        y = np.array([r["label"] == "anomaly" for r in trows])
        bad = []
        if abs(metrics.auroc(y, ts) - m["official_test"]["auroc"]["value"]) > 1e-12:
            bad.append("auroc")
        for tn, t in fz["thresholds"].items():
            if metrics.confusion(y, ts > t["threshold"]) != m["official_test"]["operating_points"][tn]["confusion"]:
                bad.append(f"confusion {tn}")
            if int((hs > t["threshold"]).sum()) != m["supplementary_holdout"]["operating_points"][tn]["fpr"]["x"]:
                bad.append(f"holdout {tn}")
        record(f"C1 {name}: metrics recomputed from per-image scores", not bad, ", ".join(bad))
        # C2
        c2 = f"C2 {name}: freeze intact and precedes evaluation"
        if not (resolve_path(cfg, "artifacts") / "model.pt").exists():
            # Fitted models are not committed; a fresh clone must run `fit` before this can be checked.
            skip(c2, "fitted model not present; run `ivad --config <config> fit` first")
            continue
        try:
            check_freeze(cfg)
            ok = datetime.fromisoformat(fz["frozen_at_utc"]) < datetime.fromisoformat(m["evaluated_at_utc"])
            detail = f"frozen {fz['frozen_at_utc']} < evaluated {m['evaluated_at_utc']}"
            head = m.get("git_head_at_evaluation") or ""
            if cfg["evaluation"].get("require_committed_freeze") and _git("cat-file", "-e", f"{head}^{{commit}}").returncode:
                # Curated snapshot without the development history: the git-ordering evidence lives elsewhere.
                record(c2, ok, detail)
                skip(f"C2 {name}: freeze committed before evaluation (git history)",
                     f"evaluation commit {head[:7]} is not in this repository's history")
                continue
            if cfg["evaluation"].get("require_committed_freeze"):
                rel = freeze_path(cfg).resolve().relative_to(REPO_ROOT).as_posix()
                first = _git("log", "--diff-filter=A", "--format=%cI", "--", rel).stdout.split()
                committed = datetime.fromisoformat(first[-1]).astimezone(timezone.utc) if first else None
                at_eval = _git("show", f"{m['git_head_at_evaluation']}:{rel}").stdout
                same = json.loads(at_eval) == fz if at_eval else False
                ok = ok and committed is not None and committed < datetime.fromisoformat(m["evaluated_at_utc"]) and same
                detail += f"; freeze committed {committed}; identical at evaluation HEAD {m['git_head_at_evaluation'][:7]}: {same}"
            record(f"C2 {name}: freeze intact and precedes evaluation", ok, detail)
        except Exception as e:  # noqa: BLE001
            record(f"C2 {name}: freeze intact and precedes evaluation", False, str(e))

    # C3
    for obj, cfg in (("pcb1", cfgs["PCB1 v1"]), ("pcb2", cfgs[NEW])):
        try:
            audit_split(read_split(resolve_path(cfg, "split_manifest")), expected_counts(cfg))
            record(f"C3 {obj} split leakage audit", True)
        except ValueError as e:
            record(f"C3 {obj} split leakage audit", False, str(e))

    # C4
    cdir = comparison_dir(cfgs[NEW])
    cmp_ = json.loads((cdir / "comparison.json").read_text(encoding="utf-8"))
    rb_, rn_ = (resolve_path(cfgs[k], "reports") for k in (BASE, NEW))
    fzb = json.loads((rb_ / "frozen_operating_point.json").read_text(encoding="utf-8"))
    fzn = json.loads((rn_ / "frozen_operating_point.json").read_text(encoding="utf-8"))
    tb, tn = fzb["thresholds"][fzb["primary"]]["threshold"], fzn["thresholds"][fzn["primary"]]["threshold"]
    rows, sb = read_scores(rb_ / "scores_test.csv")
    _, sn = read_scores(rn_ / "scores_test.csv")
    _, hb = read_scores(rb_ / "scores_holdout.csv")
    _, hn = read_scores(rn_ / "scores_holdout.csv")
    y = np.array([r["label"] == "anomaly" for r in rows])
    bad = []
    for key, (a, b) in {"recall_test_anomalies": (sb[y] > tb, sn[y] > tn),
                        "fpr_test_normals": (sb[~y] > tb, sn[~y] > tn),
                        "fpr_holdout_normals": (hb > tb, hn > tn)}.items():
        mc, rec = mcnemar_exact(a, b), cmp_["paired"][key]
        if any(mc[k] != rec[k] for k in ("flagged_by_new_only", "flagged_by_base_only", "p_value")):
            bad.append(key)
    if abs(metrics.auroc(y, sn) - metrics.auroc(y, sb) - cmp_["paired"]["auroc"]["delta"]) > 1e-12:
        bad.append("auroc delta")
    failed_pairing = [k for k, v in cmp_["pairing_checks"].items() if v is not True]
    record("C4 paired comparison recomputed; pairing checks true", not bad and not failed_pairing,
           f"{bad} {failed_pairing}")

    # C5
    from pypdf import PdfReader
    srcs: list[Any] = [cmp_, yaml.safe_load((REPO_ROOT / "config" / "acceptance_criteria.yaml").read_text(encoding="utf-8"))]
    for cfg in cfgs.values():
        rep = resolve_path(cfg, "reports")
        for fn in ("metrics.json", "frozen_operating_point.json", "robustness.json"):
            if (rep / fn).exists():
                srcs.append(json.loads((rep / fn).read_text(encoding="utf-8")))
        srcs.append(cfg["robustness"])
    # Since the PCB3 experiment the executive PDFs are printed from the website, whose numbers all come
    # from site/facts.py -> site/data/facts.json (itself read only from recorded outputs).
    for extra in ("reports/pcb3_illum_comparison/comparison.json", "reports/pcb3_baseline/robustness.json",
                  "reports/pcb3_illum/robustness.json", "site/data/facts.json", "site/assets_manifest.json"):
        if (REPO_ROOT / extra).exists():
            srcs.append(json.loads((REPO_ROOT / extra).read_text(encoding="utf-8")))
    allowed = traceable_tokens(srcs) | PDF_LITERALS | PDF_FIXED_LABELS
    f = facts()
    must = []
    for n_ in (BASE, NEW):  # every executive PDF reports the confirmatory PCB2 comparison
        r_ = f["runs"][n_]
        must += [f"{r_['auroc']['value']:.3f}", f"{100 * r_['op']['recall']['value']:.0f}%",
                 f"{100 * r_['op']['fpr']['value']:.0f}%", f"{r_['lat']['median_ms']:.0f}",
                 f"{100 * r_['hold']['value']:.1f}%"]
    pdf_texts: dict[str, str] = {}
    for pdf_path, n_pages in EXECUTIVE_PDFS.items():
        name = pdf_path.name
        if not pdf_path.exists():
            pdf_texts[name] = ""
            record(f"C5 {name} exists", False, str(pdf_path))
            continue
        reader = PdfReader(str(pdf_path))
        text = "\n".join(p.extract_text() for p in reader.pages)
        a4 = all(abs(float(p.mediabox.width) - 595) < 2 and abs(float(p.mediabox.height) - 842) < 2
                 for p in reader.pages)
        untraced = sorted({t for t in pdf_tokens(text) if t not in allowed})
        record(f"C5 {name}: exactly {n_pages} A4 page(s)", len(reader.pages) == n_pages and a4,
               f"{len(reader.pages)} page(s), A4={a4}")
        record(f"C5 {name}: every number traces to recorded outputs", not untraced, f"untraced: {untraced}")
        # PDF extraction may put a line-break space after a printed word hyphen.
        flat = re.sub(r"(?<=\w)-\s+(?=\w)", "-", " ".join(text.split()))
        missing = [s_ for s_ in must if s_ not in flat]
        record(f"C5 {name}: headline numbers present", not missing, f"missing: {missing}")
        pdf_texts[name] = flat

    # C8 presentation wording can clarify deployment scope without rewriting frozen verdicts.
    holders = dict(pdf_texts)
    for rel in ("README.md", "docs/v2_decision_record.md", "docs/handoff.md"):
        p_ = REPO_ROOT / rel
        holders[rel] = " ".join(p_.read_text(encoding="utf-8").split()) if p_.exists() else ""
    current = {name: holders[name] for name in (*pdf_texts, "README.md")}
    required = ("Not ready for production-affecting deployment", "bounded, non-disruptive",
                "shadow-mode validation", "authorization", "entry checks",
                "Factory access has not been obtained")
    lacking = {name: [phrase for phrase in required if phrase.lower() not in text.lower()]
               for name, text in current.items()}
    lacking = {name: phrases for name, phrases in lacking.items() if phrases}
    frozen_lacking = [name for name in ("docs/v2_decision_record.md", "docs/handoff.md")
                      if RECOMMENDATION not in holders[name]]
    record("C8 deployment/shadow-validation scope clear; frozen recommendations retained",
           not lacking and not frozen_lacking,
           f"current missing: {lacking}; frozen missing: {frozen_lacking}")

    # C6
    readme = REPO_ROOT / "README.md"
    s = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if README_START in s:
        block = s[s.index(README_START): s.index(README_END) + len(README_END)]
        record("C6 README results block equals generated block", block == readme_block())
    else:
        record("C6 README results block present", False)

    # C7
    try:
        tracked = _git("ls-files").stdout.split()
        notice = (REPO_ROOT / "NOTICE.md").read_text(encoding="utf-8") if (REPO_ROOT / "NOTICE.md").exists() else ""
        site_img = [t for t in tracked if t.startswith(SITE_IMG_DIR) or (t in EXPLORATORY_FIGS and t in notice)]
        bad_paths = [t for t in tracked if t.startswith(("data/", "artifacts/")) or
                     (t.lower().endswith((".pt", ".pth", ".jpg", ".tar", ".npz", ".npy", ".env")) and t not in site_img)]
        bad_paths += [t for t in site_img if (REPO_ROOT / t).stat().st_size > SITE_IMG_MAX_BYTES]
        big = [t for t in tracked if (REPO_ROOT / t).exists() and (REPO_ROOT / t).stat().st_size > 5_000_000]
        secret_re = re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}|"
                               r"github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,})")
        secrets = [t for t in tracked if (REPO_ROOT / t).exists() and (REPO_ROOT / t).stat().st_size < 5_000_000
                   and secret_re.search((REPO_ROOT / t).read_bytes().decode("utf-8", "ignore"))]
        record("C7 no dataset, weights or large files tracked", not bad_paths and not big, f"{bad_paths} {big}")
        record("C7 no secret-like strings in tracked files", not secrets, f"{secrets}")
    except Exception as e:  # noqa: BLE001
        record("C7 repository hygiene", False, str(e))

    width = max(len(n) for n, _, _ in results)
    for n_, ok, d in results:
        tag = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        print(f"{tag}  {n_:<{width}}  {d if (not ok or d.startswith('frozen')) else ''}")
    failed = [r for r in results if r[1] is False]
    skipped = [r for r in results if r[1] is None]
    if failed:
        print("SOME CHECKS FAILED")
    elif skipped:
        print(f"ALL RUN CHECKS PASS; {len(skipped)} NOT VERIFIABLE HERE (see SKIP lines)")
    else:
        print("ALL CHECKS PASS")
    return not failed
