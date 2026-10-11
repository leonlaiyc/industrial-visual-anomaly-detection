# Canonical artifact reconciliation - 2026-10-11 (Asia/Taipei)

D5 ACCEPTED BY PROJECT OWNER. Overall Owner Acceptance PENDING. Publication NOT AUTHORIZED.

The private review branch `review/ivad-final-20261011` at
`9f9a3a001933c09349a05792d57014b4d431041b` contains the earlier PDFs generated at
18:01:34/35 Asia/Taipei. A later independent static packaging pass generated the received PDFs at
19:50:11/12 without updating that source/review/release snapshot. Thus the branch QA correctly records
localhost URI annotations for the earlier version; it does not describe the received version.

This is a content and annotation discrepancy, not just metadata. The later pass replaces the local
preview link with "Website link pending first deployment.", corrects private-source/website-download
wording, adds clickable VisA and CC BY attribution links, and discloses figure transformations in the
extended footer. All technical body content, numerical tokens and scientific conclusions remain unchanged.
The still later URL-fixture PDFs are private test data and are not selected for delivery.

## Canonical local preview

The exact received bytes are selected and preserved, rather than regenerated merely to change metadata:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/executive_summary_one_page.pdf` | 129111 | `7738141a3fcbd35b4bb36236262260359e1e5205b437223756646e686cca7b99` |
| `reports/executive_summary_extended.pdf` | 542054 | `41ecc743bcb5a70971568e6c6cdc4645db88914da4010b2ed138ffbf1412a557` |

The matching copies in `site/dist/` use hyphenated download filenames. Both PDFs have only the dataset
source and CC BY licence URI annotations, no localhost/file/http/fixture links, and exactly one/two A4
pages. Print content and templates now reproduce the same pages. Creation timestamps may differ on a
future render; source/PDF parity is verified by HTML, text, annotations and page raster comparison.

Earlier one-page hash: `1b33e3050b79c838543d935a1a1f6aeed6bb060467a658f572680e89ed16187a` (128331 bytes).
Earlier extended hash: `9c89e482cf62b24378776425570e8eda2aaa4f86f76b2459debf18cbb8dc4c5a` (541096 bytes).
Earlier dated ZIPs, manifests and QA remain historical evidence; they are superseded for delivery by
the explicitly labelled reconciled package and its new manifest. A local uncommitted review correction
does not update the remote branch. No push or merge is performed.

## Release gates

The website, technical-repository and evidence URL fields remain empty. The owner still chooses
[A or B and the licensing boundaries](licensing_review.md#publication-scope-clarification-2026-10-11-reconciliation).
The current `source_mode` default is not owner approval. A verified final public release requires separate
authorization, actual reachable canonical HTTPS endpoints, anonymous inspection of the website and
selected source/evidence content, a release build, PDF re-rendering, and new manifests/package hashes.
Any first or staging deployment is itself publication and requires authorization.

Only the static `site/dist/` package is a website upload candidate; the sanitized source candidate and
evidence subset have distinct approval scopes. None is authorized for upload in this task. D5 acceptance,
the complete post-test deviation chronology, original licensing review and NOTICE remain intact.

## Later portfolio release status (October 11, 2026)

The owner subsequently authorized synchronization to main and publication through GitHub Pages while
preserving PRIVATE repository visibility. The earlier pending/not-authorized statements above describe
those earlier snapshots. Current scope and the actual HTTP 422 account-plan hosting blocker are recorded
in [portfolio_release.md](portfolio_release.md). No source licence or Ultralytics scope decision is silently
applied; public source redistribution remains separate. D5 acceptance and overall portfolio acceptance
remain distinct. Private correspondence and model/data distribution are excluded.
