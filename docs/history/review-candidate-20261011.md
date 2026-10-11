# IVAD private final technical review

Prepared 2026-10-11 (Asia/Taipei) for branch `review/ivad-final-20261011` in `leonlaiyc/industrial-visual-anomaly-detection`. Repository visibility was verified PRIVATE before preparation. This is a private review candidate, not an approved public release. No push has been performed during preparation.

## Reviewer entry point

Read this branch through your authenticated GitHub connector or a clone with repository access. Ordinary source files do not require owner uploads. The repository must remain private. Request access only if the reviewer’s connector lacks permission. Check the branch HEAD from GitHub; the owner handoff supplies the exact prepared SHA.

1. [Release manifest](review/release_manifest.json): 321 candidate payload files, per-file sizes and SHA-256, plus all changes since the latest reviewed October 10 ZIP.
2. [Recorded final QA](review/qa_summary.json): completed preparation checks and explicit limitations. These results are from the validated finalization run; engineering tests were not rerun for this handoff-only task.
3. [D5 full incident and owner decision](docs/pcb3_post_evaluation_repair.md), [deviation chronology](docs/illumination_experiment_deviations.md). D5 ACCEPTED BY PROJECT OWNER on 2026-10-11; overall Owner Acceptance PENDING. Original pending statements remain dated historical entries and are superseded only for D5. No third-party human audit or formal signature is claimed.
4. [Technical README](README.md), [one-page PDF](reports/executive_summary_one_page.pdf), [extended PDF](reports/executive_summary_extended.pdf), [PCB4 protocol](exploratory/yolo_pcb4/PROTOCOL.md), [PCB4 audit](exploratory/yolo_pcb4/audit/AUDIT.md). Site sources and the exact built preview are under `site/` and `site/dist/`.
5. [Licensing open questions](docs/licensing_review.md), [release URLs and hosting gates](docs/release_preparation.md).

## Integrity and history

All 321 payloads are byte-identical to the verified ZIP; the only additional tracked files are this document and the two `review/` JSON files. The manifest is copied unchanged and covers the candidate payload, not these three review-only attachments. No datasets, weights, development history, personal correspondence or account-specific filesystem paths are included. Standard browser installation paths in the existing renderer are platform defaults, not owner-specific paths. Preview PDFs contain explicitly local HTTP links, not filesystem links.

History consists of the pre-existing single curated `main` commit `1fa93e235f0f57ee0125d8e6d63ea2530d8c2536` and one new sanitized review commit. Both use GitHub no-reply author metadata. No other development or review history is imported; all existing branches and `main` remain untouched.

Validate manifest hashes against Git blob bytes (`git cat-file blob HEAD:path`), because the existing `.gitattributes` can normalize line endings on some checkouts. Git blobs intentionally preserve the exact candidate bytes. Review metadata is a separate allowlist of three files.

Verified release ZIP filename: `ivad-release-candidate-20261011.zip`; size: 28,577,165 bytes; SHA-256: `813866494ee85b5b750eb033d9faa73bbb0d8c121cdcdd0737de3938f72f017e`. It is kept privately with the owner; the technical audit can be completed directly from this branch and manifest without uploading the ZIP.

## Changes since the latest reviewed ZIP

Baseline ZIP SHA-256: `13a63f7166fcba4a6094d4f3e88579580a93c23639996a196c1ebd7215e08663`. There are 20 source/deliverable changes plus 8 built-site changes, mostly D5 governance, canonical links, licensing/hosting documents and PDF navigation. No frozen score or evaluation change.

- `README.md` (modified)
- `docs/illumination_experiment_deviations.md` (modified)
- `docs/licensing_review.md` (added)
- `docs/pcb3_post_evaluation_repair.md` (modified)
- `docs/release_preparation.md` (added)
- `pyproject.toml` (modified)
- `reports/executive_summary_extended.pdf` (modified)
- `reports/executive_summary_one_page.pdf` (modified)
- `site/build.py` (modified)
- `site/content/en.yaml` (modified)
- `site/content/print.en.yaml` (modified)
- `site/content/zh-TW.yaml` (modified)
- `site/release_links.json` (added)
- `site/release_links.py` (added)
- `site/render.py` (modified)
- `site/templates/index.html.j2` (modified)
- `site/templates/print.html.j2` (modified)
- `site/templates/print_one_page.html.j2` (modified)
- `site/visual_qa.mjs` (modified)
- `tests/test_release_links.py` (added)
- `site/dist/.nojekyll` (added)
- `site/dist/executive-summary-extended.pdf` (modified)
- `site/dist/executive-summary-one-page.pdf` (modified)
- `site/dist/index.html` (modified)
- `site/dist/print/index.html` (modified)
- `site/dist/print/one-page/index.html` (modified)
- `site/dist/release-link-status.json` (added)
- `site/dist/zh-tw/index.html` (modified)

## Previous GitHub snapshot comparison

Previous GitHub review branch: `review/pcb4-yolo-20261010`, HEAD `f77160f16885d71ad5c83a31481903798d573aba`. This predates the reviewed final-corrections ZIP, so the following list also includes that already-validated correction round. Compare file content with this exact SHA; newline-only differences are distinguished from technical changes.

- `NOTICE.md` (modified)
- `README.md` (modified)
- `docs/illumination_experiment_deviations.md` (modified)
- `docs/licensing_review.md` (added)
- `docs/pcb3_post_evaluation_repair.md` (modified)
- `docs/release_preparation.md` (added)
- `exploratory/yolo_pcb4/audit/AUDIT.md` (modified)
- `exploratory/yolo_pcb4/audit/FALSE_ALARM_SUPPLEMENT.md` (added)
- `exploratory/yolo_pcb4/audit/false_alarm_supplement.json` (added)
- `exploratory/yolo_pcb4/audit/fig_false_alarms_supplement.jpg` (added)
- `exploratory/yolo_pcb4/audit/make_false_alarm_supplement.py` (added)
- `pyproject.toml` (modified)
- `reports/executive_summary_extended.pdf` (modified)
- `reports/executive_summary_one_page.pdf` (modified)
- `site/build.py` (modified)
- `site/content/en.yaml` (modified)
- `site/content/print.en.yaml` (modified)
- `site/content/print_one_page.en.yaml` (modified)
- `site/content/zh-TW.yaml` (modified)
- `site/release_links.json` (added)
- `site/release_links.py` (added)
- `site/render.py` (modified)
- `site/templates/index.html.j2` (modified)
- `site/templates/print.html.j2` (modified)
- `site/templates/print_one_page.html.j2` (modified)
- `site/visual_qa.mjs` (modified)
- `src/ivad/consistency.py` (modified)
- `tests/test_release_links.py` (added)
- `tests/test_site.py` (modified)
- `site/dist/.nojekyll` (added)
- `site/dist/assets/favicon.svg` (added)
- `site/dist/assets/img/pcb3_mask_rules.jpg` (added)
- `site/dist/assets/img/pcb3_roi.jpg` (added)
- `site/dist/assets/img/workflow_caught.jpg` (added)
- `site/dist/assets/img/workflow_false_alarm.jpg` (added)
- `site/dist/assets/img/workflow_missed.jpg` (added)
- `site/dist/assets/site.css` (added)
- `site/dist/executive-summary-extended.pdf` (added)
- `site/dist/executive-summary-one-page.pdf` (added)
- `site/dist/index.html` (added)
- `site/dist/print/index.html` (added)
- `site/dist/print/one-page/index.html` (added)
- `site/dist/release-link-status.json` (added)
- `site/dist/robots.txt` (added)
- `site/dist/zh-tw/index.html` (added)

Removed paths relative to that GitHub snapshot: []. Newline-only byte differences: 42 files; numerical content is unchanged for those files. The exact ZIP manifest remains the authoritative byte baseline.

## Completed QA and remaining gates

- Full pytest: 174 passed, 0 failed, 0 skipped; final targeted checks: 23 passed.
- `ivad check`: 19/19; PCB3 independent verifier: 90/90; PCB4 saved-score checks: 50/50; fold/leakage: 70/70.
- 2,562 protected files unchanged; clean build: 16 identical outputs.
- Browser QA: 8/8, English/Traditional Chinese desktop/tablet/mobile and print; PDFs: exactly one and two A4 pages, clickable local-preview annotations, rasterized visual inspection completed.
- Candidate privacy scan: no issues; frozen PCB1–PCB4 outputs and complete D5 disclosure preserved.
- No new training, scoring or full GPU re-evaluation. A clean snapshot has no dataset/weights or private development history, so those dependent tests/checks may skip when rerun here. Do not report those skips as executed GPU verification.

Remaining: overall owner review, source-code licence choice and YOLO scope, pretrained-weight/commercial-use and existing PDF font review, actual public URL assignment/access checks. The release build correctly rejects missing public URLs; local-preview PDF links are not usable by a remote reviewer until they run the documented local preview. Static GitHub PDF viewing still provides the technical brief.

Public deployment, publication, visibility changes, merge, push and correspondence delivery remain unauthorized. Approval to push this new private review branch would authorize only that operation; no public release or project-wide acceptance follows from it.
