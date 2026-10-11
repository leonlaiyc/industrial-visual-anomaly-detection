# Portfolio implementation and release record

Release authorization: project owner approved source synchronization to main and GitHub Pages publication
on October 11, 2026 (Asia/Taipei). Repository visibility must remain PRIVATE. No new source licence is applied.
Overall acceptance of the outward-facing portfolio remains separate from D5, which was already accepted.

## Final presentation

The website and the only active recipient-facing PDF, `reports/executive-summary-one-page.pdf`, present:
A: PCB1 normal-only CNN baseline; B: PCB2 ROI scoring and a separate PCB3 lighting/robustness experiment;
C: actually trained PCB4 supervised YOLOv8s / YOLO11s using one generic defect class.
Representative frozen results: PCB1 recall 72%; PCB2 false alarms 13% to 3% at unchanged 80% recall;
PCB4 supervised image recall 98% / 95%. These are different datasets and supervision regimes, not an
architecture ranking. PCB3 correction was not adopted. Brightness confounding, unseen-type limitations,
localization limitations and the post-evaluation analysis repair remain disclosed.

The former one/two-page briefs are explicitly historical evidence in `docs/history/`, not active downloads.
The original technical protocols, score CSVs, thresholds, splits, model/config evidence, repair and
deviation chronology are retained. No model was trained or rescored during this presentation release.

## Executed QA

- Development checkout: full pytest suite, 174 passed.
- Sanitized source checkout: full pytest suite, 167 passed, 7 skipped. The skips require original
  development Git history, VisA masks, or local visual-QA screenshots not distributed with the source.
- `ivad check`: all development checks passed; source export gates explicitly skip unavailable fitted
  models and development commits instead of claiming verification.
- Recorded PCB3 and PCB4 integrity tests recompute from existing score records; no inference was rerun.
- 2,529 existing development experiment/model/verification files unchanged by SHA-256.
- 178 frozen source-evidence files byte-identical to reviewed Git blobs at
  `9f9a3a001933c09349a05792d57014b4d431041b`.
- Seven browser cases: English and Traditional Chinese at desktop/tablet/mobile, plus print layout.
  All images loaded; no horizontal overflow, missing local routes, broken anchors or console errors.
- One A4 PDF, visually inspected after raster rendering; three genuine technical figures and clickable
  HTTPS attribution/source annotations. Website PDF bytes match the canonical project sheet.
- Sanitized source/static privacy and credential scan passed. Dataset, weights, private correspondence,
  personal host paths and generated caches are excluded from both branch payloads.
- Asset provenance is in `site/portfolio_assets_manifest.json`. VisA derivatives carry CC BY 4.0
  attribution; annotation examples are explicitly labelled as ground truth rather than predictions.

## Hosting blocker observed

GitHub Pages creation API returned HTTP 422:
"Your current plan does not support GitHub Pages for this repository."
The Pages GET API returns HTTP 404; no site or assigned `html_url` exists. The expected project URL is
`https://leonlaiyc.github.io/industrial-visual-anomaly-detection/`, but is not a verified deployed URL.
Anonymous probes of English, Traditional Chinese and PDF download returned 404.
`site/release_links.json` deliberately keeps `website_url` empty, and the PDF states "Website not yet
deployed." The strict `--release` gate must continue to reject an undeployed website.

User-side action: upgrade the owner's account to a plan supporting Pages for private repositories
(GitHub Pro or another eligible plan), then enable deployment from `gh-pages`, folder `/`.
[GitHub Pages availability](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
documents plan eligibility. Do not make the repository public to work around this blocker.

After hosting is enabled: capture the Pages API's actual `html_url`, verify English/Chinese/PDF anonymously,
put that actual URL in the canonical link settings, run the strict release build, regenerate the single PDF,
verify it visually and update both branch payloads plus their manifests. These URL-dependent checks could
not be completed because no deployed site exists.

## Source access and licensing scope

Only the owner is listed as a repository collaborator. External readers cannot anonymously access this
private source. The website and PDF explicitly say source access requires permission. No access was granted.
The owner can separately grant a reviewer access; no emails or messages were sent.

The existing source licence status (no applied IVAD licence) and Ultralytics AGPL / alternative-permission
scope review remain in `docs/licensing_review.md`. No blanket MIT licence is asserted. This release keeps
source private; the static branch contains authored HTML/CSS, result figures and one PDF, with no
Ultralytics package, YOLO service, executable training code, model weights or raw dataset.
Public source redistribution remains a separate licensing/visibility decision.

## Repository structure

`main` is advanced from the reviewed sanitized source, preserving its ancestry and technical evidence.
`gh-pages` is a separate static tree with root `index.html`, `zh-tw/`, `assets/`, one PDF, print source,
notices and static metadata. Source-tree `site/dist/` is generated/ignored rather than tracked on main.
Branch updates use fast-forward pushes. No force push, visibility change, dataset upload or development
history import is performed. Earlier review/reconciliation manifests and QA are historical records;
the current manifest is `review/portfolio_release_manifest.json`.

The privately retained workflow package outside this repository is untouched and excluded from release.
