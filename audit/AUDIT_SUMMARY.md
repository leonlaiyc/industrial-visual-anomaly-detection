# Final audit snapshot — PCB3 results, analysis-repair disclosure, website and PDF

**Status: READY FOR INDEPENDENT FINAL AUDIT.** Not ready to send. The project owner's review is pending and
this snapshot has not been independently inspected. Branch `review/pcb3-final-20261009` of the private
repository; `main` is unchanged; nothing is deployed or published.

This branch is a review snapshot. `audit/` holds evidence for the auditor and is not part of an eventual
release. The earlier branch `review/pcb3-site-20261009` (pre-test FIT-stage snapshot) is left unchanged.

## A. Provenance of this snapshot

- Parent: `main` = `1fa93e2` (curated private release).
- Content: every tracked file of the development repository at commit `e397094` except the owner's private working-notes folder
  (203 files). Each blob was verified byte-identical to the development commit before staging.
- Added for review only: `audit/` (this file, test/QA logs, visual-QA screenshots, built site
  `audit/site_dist/`, changed-file list).
- Commit hashes cited in `docs/` and `reports/*.json` refer to the private development history, which is not
  in this repository (README → *Development provenance*). Tests that need that history are SKIPPED here.

## B. Freeze and evaluation chronology (2026-10-09, +08:00)

| Time | Event | Development commit |
|---|---|---|
| 14:01:57 | PCB3 protocol pre-registered (before PCB3 data access) | `d7f5e81` |
| 16:56:24 | Addendum B2: analysis input validation added (contains the faulty label check) | `d8b5f1e` |
| 16:58:10 | **Freeze** (thresholds, configs, split, `src/`) | `eb4e7a9` |
| 16:58:51–17:07:55 | **Single planned evaluation** (evaluate + robustness, both variants) | outputs committed `115c54b` 17:11:40 |
| 17:09:13, 17:10:20 | Paired analysis refused twice, unchanged code (second refusal = `attempt1_refused.log`) | — |
| **17:14:22** | **Post-evaluation analysis-code repair** | `7dbd302` |
| 17:15:38 | Paired-analysis outputs committed | `9c9f7c5` |

## C. Post-evaluation analysis repair (summary; full record `docs/pcb3_post_evaluation_repair.md`)

- **What:** one condition in `illum.defect_descriptors` required VisA `image_anno.csv` `label == "anomaly"`;
  VisA stores the defect type (`bent`, `melt`, …). Patch: `reports/pcb3_illum_comparison/repair_7dbd302.diff`.
- **Effect:** the function feeds only the descriptive small/low-contrast subgroup table (protocol §8 C). The
  primary statistic and decision rule run before it. No change to fitting, preprocessing, thresholds,
  predictions, scores, the primary statistic or the rule. No re-scoring; the test was not repeated.
- **Disclosed explicitly:** the analysis program that produced the PCB3 result is not the frozen one; the
  repair was made after summary test metrics had been seen; there were two refused invocations, not one.
  Whether §9's "serious implementation defect → invalidate" rule applies is left to the auditor (D5).
- **Independent recomputation** (`verification/recompute_pcb3.py`, imports nothing from `src/ivad`): 90/90
  checks agree in the development repository (dataset present); 89/90 agree + 1 SKIPPED here (subgroup
  table needs the dataset).

## D. Verified PCB3 metrics (each variant at its own frozen P95 threshold; recomputed from raw scores)

| Quantity | Baseline (v2 pipeline) | Candidate (v2 + illumination correction) |
|---|---|---|
| Recall, official test | 90/100 = 90.0% [82.4%, 95.1%] | 91/100 = 91.0% [83.6%, 95.8%] |
| FPR, official-test normals | 3/101 = 3.0% [0.6%, 8.4%] | 3/101 = 3.0% [0.6%, 8.4%] |
| FPR, supplementary holdout | 8/155 = 5.2% [2.3%, 9.9%] | 6/155 = 3.9% [1.4%, 8.2%] |
| FPR, pooled 256 unseen normals (robustness reference) | 11/256 = 4.3% | 9/256 = 3.5% |
| AUROC | 0.982 | 0.983 |
| G1 darkening: absolute FPR / change | 22/256 = 8.6% / +4.3 pts | 8/256 = 3.1% / −0.4 pts |
| B1 brightening | 50/256 = 19.5% / +15.2 pts | 16/256 = 6.2% / +2.7 pts |
| N1 non-uniform | 16/256 = 6.2% / +2.0 pts | 15/256 = 5.9% / +2.3 pts |
| Mild blur | 12/256 = 4.7% / +0.4 pts | 46/256 = 18.0% / +14.5 pts |
| Shift | 11/256 = 4.3% / +0.0 pts | 11/256 = 4.3% / +0.8 pts |

Intervals: Clopper–Pearson exact 95%. Primary endpoint: DiD −4.7 pts, paired bootstrap 95% CI [−8.6, −0.8]
(B = 10000, seed 2026; upper bound −1.2 to −0.8 under ten other seeds).

## E. Two decisions

- **A. Pre-registered outcome: ROBUSTNESS IMPROVEMENT SUPPORTED — weakly informative** (baseline G1 rise
  +4.3 < 5 pts). Narrow meaning only: the correction undoes a synthetic global darkening.
- **B. Engineering adoption: DO NOT ADOPT the candidate** (blur +14.5 pts > the original 10-pt P5 limit).
- **The baseline is not production-ready either** (brightening +15.2 pts; no end-user validation, factory
  pilot or operational ROI). Original criteria: both variants **INCONCLUSIVE** (P1–P4 met, P5 not met).
- Website/README/PDF: PCB1/PCB2/PCB3 shown as separate datasets, no cross-category arrows; the PCB2 v1→v2
  controlled comparison is the headline; "No manufacturing pilot recommended yet."

## F. Commands and results

| Check | Where | Command | Result |
|---|---|---|---|
| Full test suite | development repo `e397094` | `python -m pytest -rs` | **151 passed**, 0 skipped (`logs/pytest_dev.log`) |
| Full test suite | this export, own `src/` | `PYTHONPATH=src python -m pytest -rs` | **144 passed, 7 SKIPPED**: dataset absent (2), dev history absent (4), visual QA not run here (1) (`logs/pytest_export.log`) |
| Consistency gates C1–C8 (PCB1/PCB2, PDF, README, hygiene) | development repo | `python site/build.py && python -m ivad check` | ALL CHECKS PASS, exit 0 (`logs/ivad_check_dev.log`) |
| Same | this export | same with `PYTHONPATH=src` | ALL RUN CHECKS PASS; C2 ×3 SKIP (no fitted models), exit 0 (`logs/ivad_check_export.log`) |
| PCB3 independent recomputation | development repo | `python verification/recompute_pcb3.py` | 90/90 agree (`logs/recompute_dev.log`, `reports/pcb3_verification/`) |
| Same | this export | `python verification/recompute_pcb3.py --check` | 89/90 agree, 1 SKIPPED (`logs/recompute_export.log`) |
| Website build, literal-number lint, EN/zh key parity | both | `python site/build.py` | pass; export `site/dist` byte-identical to development build |
| EN/zh metric parity, images, placeholders, arrows, timeline, PDF content | both | `tests/test_site.py` | pass |
| Visual QA (real browser) | development repo | `node site/visual_qa.mjs` against `python -m http.server --directory site/dist` | 7/7 PASS (`visual_qa/visual_qa.json`) |
| PDF | development repo | `python site/render.py pdf` | 2 A4 pages; rasterised and inspected (`visual_qa/pdf_page_*.jpg`) |
| Privacy/secret scan of every staged file | this export | see `audit/SCAN.md` | no findings (`audit/SCAN.md`) |

## G. Visual QA

Headless Chrome over the DevTools protocol, for EN and zh-TW at desktop 1440×900, tablet 820×1180 and
mobile 375×812 (DPR 2), plus the print page: every page waited for fonts and for each image to load **and**
decode, then scrolled. Expected = loaded = visible images on every page (5 on each case-study page, 3 on
the print page, resolved at the published relative paths `assets/img/…` and `../assets/img/…`); 0 failed
requests, 0 console errors, 0 broken anchors, 0 bad same-site links, no horizontal overflow, no placeholders.
Full pages are captured in ≤ 4000 px segments. Every capture was checked for blank output and inspected.

Findings fixed during QA: (1) `executive-summary.pdf` returned 404 after a rebuild (`build.py` wiped
`dist/`); (2) the print page requested a missing `/favicon.ico`; (3) the first QA run returned blank
**mobile first-screen captures** — caused by capture order in the QA tool (a viewport capture taken right
after a beyond-viewport capture under mobile emulation), not by the page: the full-page mobile segments
from the same run had content. The tool now captures the first screen first and waits for two animation
frames; `tests/test_site.py` fails on any blank capture. (4) `decoding="async"` removed from images.

## H. Remaining limitations and open issues

- Independent auditor to confirm D5 (repair not treated as run-invalidating) and the narrow reading of the
  pre-registered outcome.
- Session-log timestamps in the repair record (17:09:13, 17:10:20) come from the development session's
  tool-execution log, which is private and not in this snapshot; commit times are verifiable only against
  the private history.
- Synthetic perturbations only; 100 defects per board type; no real user, factory image, pilot or ROI.
- Visual QA used one browser engine (Chromium); no screen-reader or cross-browser test.
- The project owner's review is pending. Not ready to send.
