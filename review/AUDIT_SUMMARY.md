# Review snapshot audit summary — `review/pcb3-site-20261009`

**Purpose.** Independent technical audit of the latest project state before any PCB3 calibration or
test evaluation. This branch is a **sanitized snapshot**, not the development history. It was built in a
fresh clone of `origin/main` (`1fa93e235f0f57ee0125d8e6d63ea2530d8c2536`) by copying an explicit file
list from the development repository's committed `HEAD`. **This branch does not, by itself, prove the
development chronology**: its single snapshot commit has `main` as parent and today's commit time. The
chronology below is a transcription of the local development history, which is kept unchanged locally
for separate inspection on request.

Prepared 2026-10-09 (Asia/Taipei) with AI assistance (Claude Code). Not yet reviewed by the project owner.

## 1. Development repository state at snapshot time

| Item | Value |
|---|---|
| Development branch | `v2-dev` (local only; no remote) |
| Development HEAD | `83e5e0f271e3e77ea2050159ed6c467d01d75947` |
| Uncommitted changes | none (`git status --porcelain` empty; only ignored local outputs exist: data, model weights, built site) |
| PCB3 protocol commit | `d7f5e813619fa2da219398f60f4938c70b1d7e42` (2026-10-09 14:01:57 +08:00), before PCB3 extraction |
| PCB3 code commit (no PCB3 data yet) | `caca753249097dbfb9373ffed5dac02e19fec62d` (14:08:07 +08:00) |
| Addendum B1 commit (FIT inspection, adaptation, reference values, FIT-only ROI QC) | `5ba6aca88e30c74b8686210a6e942d835676ec97` (14:24:03 +08:00) |
| `main` content equals development commit | `d79431e` minus `private_review/` (all 127 files byte-identical) |

### PCB3 stage: **FIT only**

| Question | Answer | Evidence (`review/verification/00_pcb3_fit_only.txt`) |
|---|---|---|
| CALIB images scored? | **No** | no `scores_calib.csv`, no `frozen_operating_point.json` for either PCB3 run |
| HOLDOUT images scored? | **No** | no `scores_holdout.csv`, no `metrics.json` |
| Official TEST images scored? | **No** | no `scores_test.csv`, no robustness or comparison output |
| Models/configs changed since FIT? | **No** | each `model.pt` stores the config hash it was fitted with; both equal the hash of the committed configs |
| Which images were fitted? | 600 distinct images, all `role=fit` in `manifests/split_pcb3.csv` | same file |

**Disclosed data handling that is not scoring.** `ivad prepare` extracted all 1,207 PCB3 archive files
(images and masks). `ivad split` read the bytes of each of the 1,106 manifest images to compute SHA-256
for the leakage audit, including the 201 test images. No CALIB, HOLDOUT or TEST image was decoded for
inspection, passed to a model or scored. Inspection, photometric statistics, the background-hue band and
both fits used FIT images only.

**Chronology inside commit `5ba6aca`** (file modification times in the development repo): photometric
statistics 14:11, background-hue band 14:12, re-inspection with the band 14:13–14:14, baseline fit and
ROI QC 14:19, candidate fit 14:22 and ROI QC 14:23, commit 14:24. The first inspection (before the band;
it revealed the 96% board-mask failure) ran in the same command as, and just before, the photometric statistics (about 14:10–14:11); its outputs were overwritten by the
re-inspection, and its key numbers are recorded in Addendum B1.

**Other scoring disclosed.** As a regression check of the board-mask code change, three already
evaluated PCB2 v2 test images were re-scored with `ivad predict`, before and after the change, with
identical results. No PCB3 data was involved.

## 2. Original commit chronology (development repository, linear history)

Private working-notes files (under `private_review/`, and three files kept elsewhere before commit `4028646` moved them there) are counted but not named. Early versions of some public documents also contained internal working notes that were removed later; this is one more reason the history is not pushed. Subjects of commits that only touch private notes
are withheld. SHAs, parents and timestamps are complete.

| Commit | Parent | Author date | Commit date | Files changed | Subject |
|---|---|---|---|---|---|
| `1c5b121` | `—` | 2026-10-09T00:03:55+08:00 | 2026-10-09T00:03:55+08:00 | 37 (0 private, not listed) | Initial PoC: requirements baseline, data prep, frozen split, registration, Mahalanobis model, tests |
| `fdd1f35` | `1c5b121` | 2026-10-09T00:04:22+08:00 | 2026-10-09T00:04:22+08:00 | 2 (0 private, not listed) | Freeze operating point (P95 primary) from calibration normals only |
| `f643526` | `fdd1f35` | 2026-10-09T00:06:22+08:00 | 2026-10-09T00:06:22+08:00 | 8 (0 private, not listed) | Fix CRLF-sensitive manifest hash; re-freeze (identical thresholds) |
| `5a66747` | `f643526` | 2026-10-09T00:21:57+08:00 | 2026-10-09T00:21:57+08:00 | 26 (1 private, not listed) | Final evaluation, robustness, diagnostics, reports, PDF and docs (WIP) |
| `32dbe6b` | `5a66747` | 2026-10-09T00:29:57+08:00 | 2026-10-09T00:29:57+08:00 | 13 (3 private, not listed) | Review package: reproducibility record, consistency gates, docs, final PDF |
| `ad8a386` | `32dbe6b` | 2026-10-09T11:06:27+08:00 | 2026-10-09T11:06:27+08:00 | 2 (1 private, not listed) | Pre-register v2 hypothesis and PCB2 protocol (before any PCB2 data access) |
| `fd1caf8` | `ad8a386` | 2026-10-09T11:18:51+08:00 | 2026-10-09T11:18:51+08:00 | 14 (0 private, not listed) | v2: board-ROI score aggregation, paired comparison, PCB2 configs, tests |
| `9b69850` | `fd1caf8` | 2026-10-09T11:29:49+08:00 | 2026-10-09T11:29:49+08:00 | 18 (0 private, not listed) | PCB2 data, split audit, FIT inspection; protocol Addendum A1 (ROI rule amendment) |
| `862c28f` | `9b69850` | 2026-10-09T11:34:33+08:00 | 2026-10-09T11:34:33+08:00 | 10 (3 private, not listed) | FREEZE PCB2 v1 and v2 before any test/holdout scoring |
| `2b775be` | `862c28f` | 2026-10-09T11:37:22+08:00 | 2026-10-09T11:37:22+08:00 | 6 (0 private, not listed) | PCB2 confirmatory evaluation (single run), v1 and v2 — raw outputs |
| `20639bc` | `2b775be` | 2026-10-09T11:41:10+08:00 | 2026-10-09T11:41:10+08:00 | 4 (0 private, not listed) | PCB2 robustness at frozen thresholds (v1, v2) — raw outputs |
| `736dcc7` | `20639bc` | 2026-10-09T11:44:11+08:00 | 2026-10-09T11:44:11+08:00 | 16 (0 private, not listed) | PCB2 paired comparison, per-run reports and post-hoc diagnostics |
| `0f00897` | `736dcc7` | 2026-10-09T11:44:23+08:00 | 2026-10-09T11:44:23+08:00 | 4 (0 private, not listed) | Add PCB1 v2 retrospective config (post hoc, discovery data); summary and check modules |
| `87ba3ee` | `0f00897` | 2026-10-09T11:46:10+08:00 | 2026-10-09T11:46:10+08:00 | 4 (0 private, not listed) | Freeze PCB1 v2 retrospective (post hoc, discovery data) |
| `b0ba490` | `87ba3ee` | 2026-10-09T11:48:36+08:00 | 2026-10-09T11:48:36+08:00 | 8 (0 private, not listed) | PCB1 v2 retrospective (post hoc) and threshold-mechanism analysis |
| `4028646` | `b0ba490` | 2026-10-09T11:54:48+08:00 | 2026-10-09T11:54:48+08:00 | 17 (4 private, not listed) | v2 documentation, executive summary, README; private files moved out of docs/ |
| `38efd70` | `4028646` | 2026-10-09T11:55:44+08:00 | 2026-10-09T11:55:44+08:00 | 1 (0 private, not listed) | README: provenance note for the curated public snapshot |
| `eec7438` | `38efd70` | 2026-10-09T11:58:08+08:00 | 2026-10-09T11:58:08+08:00 | 2 (2 private, not listed) | [subject withheld: private working notes only] |
| `a093e62` | `eec7438` | 2026-10-09T13:19:15+08:00 | 2026-10-09T13:19:15+08:00 | 11 (3 private, not listed) | Final delivery refinement: executive PDF redesign, docs, review package |
| `d79431e` | `a093e62` | 2026-10-09T13:23:49+08:00 | 2026-10-09T13:23:49+08:00 | 1 (1 private, not listed) | [subject withheld: private working notes only] |
| `d7f5e81` | `d79431e` | 2026-10-09T14:01:57+08:00 | 2026-10-09T14:01:57+08:00 | 1 (0 private, not listed) | Pre-register PCB3 illumination-correction experiment (before any PCB3 data access) |
| `caca753` | `d7f5e81` | 2026-10-09T14:08:07+08:00 | 2026-10-09T14:08:07+08:00 | 9 (0 private, not listed) | PCB3 illumination experiment: correction, N1 perturbation, paired analysis, configs, tests (no PCB3 data yet) |
| `5ba6aca` | `caca753` | 2026-10-09T14:24:03+08:00 | 2026-10-09T14:24:03+08:00 | 26 (0 private, not listed) | PCB3: data, split audit, FIT inspection, Addendum B1 (background-hue board mask), reference values, FIT-only ROI QC |
| `7c7b5f1` | `5ba6aca` | 2026-10-09T14:33:08+08:00 | 2026-10-09T14:33:08+08:00 | 9 (0 private, not listed) | Website prototype: facts layer, bilingual content with literal-number lint, first screen, case-study structure, render helper |
| `83e5e0f` | `7c7b5f1` | 2026-10-09T14:34:01+08:00 | 2026-10-09T14:34:01+08:00 | 1 (0 private, not listed) | Site: mobile header fixes |

### Files changed in the five commits after the published snapshot (none private)

**`d7f5e813619fa2da219398f60f4938c70b1d7e42`** (parent `d79431e21d4f27395bff8b528a2cfaa6f519475b`), 2026-10-09T14:01:57+08:00 — Pre-register PCB3 illumination-correction experiment (before any PCB3 data access)

- `A` `docs/illumination_experiment_protocol.md`

**`caca753249097dbfb9373ffed5dac02e19fec62d`** (parent `d7f5e813619fa2da219398f60f4938c70b1d7e42`), 2026-10-09T14:08:07+08:00 — PCB3 illumination experiment: correction, N1 perturbation, paired analysis, configs, tests (no PCB3 data yet)

- `A` `config/pcb3_baseline.yaml`
- `A` `config/pcb3_illum.yaml`
- `M` `src/ivad/acceptance.py`
- `M` `src/ivad/cli.py`
- `A` `src/ivad/illum.py`
- `A` `src/ivad/photometric.py`
- `M` `src/ivad/pipeline.py`
- `M` `src/ivad/robustness.py`
- `A` `tests/test_illum.py`

**`5ba6aca88e30c74b8686210a6e942d835676ec97`** (parent `caca753249097dbfb9373ffed5dac02e19fec62d`), 2026-10-09T14:24:03+08:00 — PCB3: data, split audit, FIT inspection, Addendum B1 (background-hue board mask), reference values, FIT-only ROI QC

- `M` `config/pcb3_baseline.yaml`
- `M` `config/pcb3_illum.yaml`
- `M` `docs/illumination_experiment_protocol.md`
- `A` `manifests/split_pcb3.csv`
- `A` `manifests/split_pcb3_audit.json`
- `A` `manifests/visa_1cls_pcb3.csv`
- `A` `manifests/visa_1cls_pcb3.provenance.json`
- `A` `reports/pcb3_baseline/roi_qc.json`
- `A` `reports/pcb3_baseline/roi_qc.png`
- `A` `reports/pcb3_illum/roi_qc.json`
- `A` `reports/pcb3_illum/roi_qc.png`
- `A` `reports/pcb3_inspection/background_hue_fit.json`
- `A` `reports/pcb3_inspection/contact_sheet_fit_normals.png`
- `A` `reports/pcb3_inspection/inspection_stats.json`
- `A` `reports/pcb3_inspection/mean_std_fit_normals.png`
- `A` `reports/pcb3_inspection/photometric_fit_stats.json`
- `A` `reports/pcb3_inspection/registered_crops.png`
- `A` `reports/pcb3_inspection/registered_mean_std_fit_normals.png`
- `A` `reports/pcb3_inspection/resize_loss_crops.png`
- `M` `src/ivad/align.py`
- `M` `src/ivad/cli.py`
- `M` `src/ivad/inspect_data.py`
- `M` `src/ivad/pipeline.py`
- `M` `src/ivad/report.py`
- `M` `src/ivad/roi.py`
- `M` `tests/test_illum.py`

**`7c7b5f14b98364d5ec1575023c8b7af0c8e28f8f`** (parent `5ba6aca88e30c74b8686210a6e942d835676ec97`), 2026-10-09T14:33:08+08:00 — Website prototype: facts layer, bilingual content with literal-number lint, first screen, case-study structure, render helper

- `M` `.gitignore`
- `A` `site/build.py`
- `A` `site/content/en.yaml`
- `A` `site/content/zh-TW.yaml`
- `A` `site/facts.py`
- `A` `site/render.py`
- `A` `site/static/favicon.svg`
- `A` `site/static/site.css`
- `A` `site/templates/index.html.j2`

**`83e5e0f271e3e77ea2050159ed6c467d01d75947`** (parent `7c7b5f14b98364d5ec1575023c8b7af0c8e28f8f`), 2026-10-09T14:34:01+08:00 — Site: mobile header fixes

- `M` `site/static/site.css`

## 3. What this snapshot contains

Relative to `main`: **9 modified** and **30 added** project files, exactly
`git diff --name-only d79431e 83e5e0f` in the development repository (no `private_review/` entries),
copied from committed `HEAD` content with `git show`; plus the `review/` folder.

| Path | Content |
|---|---|
| `src/ivad/photometric.py` (new) | global luminance affine standardisation (candidate correction) |
| `src/ivad/illum.py` (new) | FIT-only reference statistics; the planned paired analysis (DiD bootstrap, outcome rule, subgroups, runtime) |
| `src/ivad/align.py`, `roi.py`, `pipeline.py`, `inspect_data.py`, `report.py`, `cli.py` | optional `exclude_hue` board-mask band (Addendum B1) passed through registration, ROI masks, inspection and reports; photometric step in `Preprocessor`; CLI commands `photostats`, `illum-compare` |
| `src/ivad/robustness.py`, `acceptance.py` | N1 illumination ramp; P5 acceptance ignores conditions marked `acceptance: false` |
| `config/pcb3_baseline.yaml`, `config/pcb3_illum.yaml` | the two PCB3 variants |
| `docs/illumination_experiment_protocol.md` | pre-registration and Addendum B1 |
| `manifests/*pcb3*` | official PCB3 manifest and provenance, split, split audit |
| `reports/pcb3_inspection/`, `reports/pcb3_*/roi_qc.*` | FIT-only inspection evidence and ROI QC (figures contain downscaled VisA FIT images, CC BY 4.0, as for PCB1/PCB2) |
| `tests/test_illum.py` | 18 tests: correction, N1, DiD, outcome rule, terciles, acceptance filter, config isolation, hue band |
| `site/` | static site source: `facts.py` (only number source), `content/en.yaml`, `content/zh-TW.yaml`, `templates/`, `static/`, `build.py` (literal-number lint, key parity), `render.py` (headless Chrome) |
| `.gitignore` | ignores local tooling, `site/dist/`, `site/qa/` |
| `review/generated/` | build output at this snapshot: `site_dist/` and `facts.json` (normally not committed; included for visual review) |
| `review/screenshots/` | desktop 1440×900, tablet 820 and mobile 375 (EN and 中文), one full-page desktop capture |
| `review/verification/` | logs of every check in §5 (local paths redacted) |

**PCB1/PCB2 artifacts unchanged.** All 90 files under `reports/`, `config/`, `manifests/` and `docs/` on
`main` are byte-identical (git blob hash) in the development HEAD and in this snapshot. The only files
changed from `main` are `.gitignore` and the eight `src/ivad/` modules above. `ivad check` recomputes every
PCB1/PCB2 metric from per-image scores and passes (§5).

## 4. Intentionally excluded

| Excluded | Why |
|---|---|
| Development Git history (25 commits, including every commit listed above) | contains private working notes; kept locally |
| `private_review/` (11 files at HEAD) | private review and personal working materials |
| `data/` (VisA archive and images), `artifacts/` (`model.pt`, about 406 MB each) | dataset is downloaded by `ivad prepare`; weights are regenerated by `ivad fit` |
| Local tooling (`.claude/`, `.venv/`, caches) and scratch QA files | not project content |

## 5. Verification actually run (logs in `review/verification/`)

| # | Command (where) | Result |
|---|---|---|
| 00 | PCB3 FIT-only checks (dev repo) | no CALIB/HOLDOUT/TEST/freeze outputs; config hashes match fitted models; 600 FIT images |
| 01 | `python -m pytest` (dev repo) | **98 passed** |
| 02 | `python -m ivad check` (dev repo) | **ALL CHECKS PASS** (C1–C8, including C2 freeze-before-evaluation via git) |
| 03 | `python -m compileall -q src site tests`; import of all 20 `ivad` modules (dev) | exit 0; 20 modules imported |
| 04 | `python site/build.py` (dev) and two negative tests (literal number injected; key removed from zh-TW) | build exit 0; each negative test **fails the build** (exit 1); files restored |
| 05 | facts vs README headline rows; independent break-even k recomputation | ALL PASS (0.0375 < k < 4.85) |
| 06 | `PYTHONPATH=src python -m pytest`; `python -m ivad check` (**in this snapshot**) | **98 passed**; all run checks pass; C2 ×3 **SKIP** (needs `model.pt`, excluded) |
| 07 | `python site/build.py` in this snapshot; `diff -r` against the dev build | exit 0; output and `facts.json` byte-identical |
| 08 | staged-file privacy and secret scan before commit | see the snapshot commit message |

**Not run / limitations.** No linter or formatter is installed in the project environment (no ruff,
flake8, pyflakes, black, mypy or pylint); only `compileall` and import checks were run. `ivad check` does
not cover PCB3 yet (there is no PCB3 evaluation to check; the PCB3 split audit is in
`manifests/split_pcb3_audit.json`, not in C3). `illum_compare` is unit-tested in its parts but has never
run end to end, because it needs frozen PCB3 outputs. No accessibility audit tool was run. Mobile
screenshots come from the desktop app's browser-pane emulation, because headless Chrome enforces a
minimum window width of about 500 px. Snapshot tests used the development virtual environment with
`PYTHONPATH=src` (the printed `ivad.__file__` confirms the snapshot source was imported); a fresh
environment install was not repeated.

## 6. Known issues and open technical risks

1. **Post-protocol code change (Addendum B1).** The PCB3 background is saturated green; the
   pre-registered pipeline's colour mask marked about 96% of every frame as board, so registration did
   nothing. A FIT-derived hue band was added to the code and to both configs before any fit. Auditors
   should judge whether this is an allowed protocol §10 adaptation.
2. **G1 is inside the correction's model class.** Global darkening is affine with one gain; the
   correction inverts it up to rounding. A favourable G1 result is a mechanism check, not real-world
   robustness. N1 is the out-of-class test (protocol §3, risk 1).
3. **Weakly informative primary endpoint possible.** On PCB3 FIT images G1 maps the mean grey level to
   about the bottom of the natural range (about 83 vs P5 89.3); the baseline may not reproduce the PCB2
   false-alarm rise.
4. **Pre-existing batch-size float drift.** Single-image `predict` differs from batch-8 evaluation
   scores by about 0.01 (e.g. 50.627 vs 50.637); identical before and after this snapshot's changes.
5. **Website is a prototype.** Case-study sections are placeholders; the "Executive summary (PDF)" link
   points to a file not yet generated; `hreflang` alternates are relative; no image mock-up yet.
6. **`site/data/` is ignored** by the repository-wide `data/` pattern, so `facts.json` is generated, not
   tracked (a copy is in `review/generated/`).
7. Open owner decisions (not technical defects): public display of downscaled VisA images in a mock-up;
   the "Project owner" name in the site footer.

## 7. Reproduce

```bash
# tests (Python 3.11 environment from README "Reproduce")
python -m pytest
python -m ivad check

# website (no network; reads reports/ only)
python site/build.py                                  # -> site/dist/; fails on literal numbers or missing keys
python -m http.server 8765 --directory site/dist      # open http://127.0.0.1:8765/ and /zh-tw/
python site/render.py shots --out site/qa             # optional: headless Chrome/Edge screenshots
```

The PCB3 FIT stage can be regenerated with `ivad --config config/pcb3_baseline.yaml prepare`, `split`,
`inspect`, `photostats`, then `fit` and `roi` for both configs. **Do not run `calibrate`, `evaluate`,
`robustness` or `illum-compare` for PCB3**: the protocol requires the freeze commit first, and the owner
has paused the experiment pending this audit.
