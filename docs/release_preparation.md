# Pre-publication release and hosting steps

Prepared 2026-10-11 (Asia/Taipei). D5 ACCEPTED BY PROJECT OWNER; overall project-owner review PENDING.
Website deployment and GitHub push, merge or visibility changes are NOT AUTHORIZED.

## Source access choices

| Choice | Reviewer experience | Release requirement |
|---|---|---|
| A: public website + sanitized public GitHub repository | Lowest friction: browse scripts, protocols, raw scores and tests directly | Separate authorization for public source, licence decision and YOLO scope review; publish only the curated snapshot, with no development history or weights |
| B: public website + private repository + public evidence package | No login for results/protocols/raw scores; source inspection requires separate access | A public evidence URL, with protocols, deviations, splits, thresholds, per-image scores, verification results, notices and limitations; do not label the private GitHub URL as public source |

The owner must choose A or B; neither is selected by this reconciliation. Evidence publication still
needs review and authorization. The local sanitized
candidate includes the complete technical evidence and source for private review, not permission to publish.

## Canonical links and local preview

`site/release_links.json` is the only URL setting used by website navigation and both PDF templates:

- `website_url`: actual canonical HTTPS website URL, including any project subpath and trailing slash.
- `technical_repository_url`: actual anonymously readable sanitized repository URL; currently empty.
- `technical_evidence_url`: actual public evidence landing page or downloadable package; currently empty.
- `source_mode`: `public-repository` for A or `public-evidence` for B.

The known GitHub repository is private. Its address is not automatically copied into public navigation.
No website URL is assigned or invented. An empty setting builds a preview with an explicit source-access
notice. Recipient PDF previews display "Website link pending first deployment." and have only public
attribution URI annotations. Localhost links are rejected in every PDF mode. The renderer's local
server is a transport for generating the PDF, not a recipient-facing link.

From the project root with the recorded Python environment (or `pip install -e ".[site,dev]"`):

```powershell
python site/build.py
python -m http.server 8765 --bind 127.0.0.1 --directory site/dist
```

Keep that terminal running; in another terminal:

```powershell
python site/render.py pdf
node site/visual_qa.mjs
python -m pytest
ivad check
```

Local routes: `/`, `/zh-tw/`, `/print/one-page/`, `/print/`. Keep all assets and PDFs together.
After assigning approved actual URLs, run `python site/build.py --release`, then re-render both PDFs,
verify their URI annotations and exact 1/2-page A4 layouts, and rerun link/privacy/manifest QA. The release
gate performs anonymous HTTPS GETs, rejects missing URLs/login redirects/errors and verifies GitHub public
repository metadata through the anonymous API. Generic evidence endpoints also need a logged-out browser
check for login interstitials; HTTP 200 alone cannot establish useful content. A transient network or API
rate-limit failure stops the release build. Every configured external navigation target is checked.
Release mode requires the website plus either source or evidence. For B the private repository setting
must remain empty.

The gate runs before clearing output. A build copies canonical PDFs from `reports/`; they must match
the print sources. The reconciled preview uses the exact received and verified PDF bytes documented in
[artifact reconciliation](artifact_reconciliation.md). Preserve those bytes while URLs remain pending.
After authorized URL changes, re-render, verify source/PDF parity, and regenerate all package manifests.
This preview package is not a verified final public release.

## Hosting comparison

| Item | Cloudflare Pages | GitHub Pages |
|---|---|---|
| Private source | Direct Upload needs no repository access; optional Git integration | Private repository Pages requires an eligible paid GitHub plan |
| Account/cost | Cloudflare account, Free plan adequate for these assets; Git builds limited to 500/month | GitHub Free for public repository; Pro/Team/Enterprise for private repository Pages |
| Build | Upload locally verified `site/dist`; no cloud build or backend | Publish verified static files in a dedicated approved publishing branch/folder or Pages artifact |
| Routing/assets | Directory index routes and relative assets | Same layout; `.nojekyll` included; project subpath must be in canonical URL |
| HTTPS/public URL | Provider assigns actual `pages.dev` address; optional custom domain | Provider displays actual Pages URL; enable Enforce HTTPS |
| Maintenance | Rebuild, verify, upload complete static folder; keep previous version for rollback | Rebuild, verify, approved push/deploy; preserve prior artifact for rollback |

Recommended minimum: Cloudflare Pages Direct Upload. It avoids granting access to a private repository and
avoids installing GPU/PyTorch/Chrome in a hosted builder. It cannot later switch to Git integration in the
same project. [Official Direct Upload guide](https://developers.cloudflare.com/pages/get-started/direct-upload/)
accepts a folder or ZIP through the dashboard (1,000 files, 25 MiB per asset); our static bundle fits.
[Cloudflare limits](https://developers.cloudflare.com/pages/platform/limits/) describe the Free build limits.
[GitHub Pages availability](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
documents eligible plans. A public website does not grant access to private source.

## Exact account actions, after separate authorization

1. Leon approves final content, licence/source scope, public evidence or repository, and hosting separately.
2. Leon signs in to Cloudflare, opens Workers & Pages, selects Pages and Direct Upload, and chooses a name.
   Record only the actual provider-assigned address. No custom domain is required. Do not create this
   project before deployment authorization.
3. Resolve the first-deployment accessibility dependency explicitly: with authorization, deploy a staging
   preview containing the reviewed website and evidence (or make the approved sanitized repository
   readable). Preview deployment is itself publication and requires authorization. Record its actual URLs.
4. Set canonical links, run the strict release build against reachable endpoints, re-render PDFs and verify
   that PDF -> website -> source/evidence works anonymously. Review actual target content, not just HTTP status.
5. Upload the complete final `site/dist` folder, not the source candidate ZIP. Dashboard build preset:
   none; build command: none (prebuilt); output directory: the uploaded folder root. No functions,
   worker, API, environment secrets or backend. Ensure HTTPS, English/Chinese routes and PDF downloads work.
6. Rerun anonymous route/download checks after upload; save deployed file hashes and URL/version in the
   release record. Roll back to the prior uploaded version if those checks fail.

For GitHub Pages instead: Leon confirms plan, authorizes the specific sanitized repository/branch push,
then Settings -> Pages -> Deploy from a branch -> approved publishing branch -> `/ (root)` containing the
verified dist contents and `.nojekyll`. Save, record the displayed address and enforce HTTPS. Use the same
staging/strict-link/re-render procedure. Never push the development checkout or its historical private notes.

No external account operation, repository mutation or deployment has been performed in this task.

## Later portfolio release status (October 11, 2026)

The owner subsequently authorized synchronization to main and publication through GitHub Pages while
preserving PRIVATE repository visibility. The earlier pending/not-authorized statements above describe
those earlier snapshots. Current scope and the actual HTTP 422 account-plan hosting blocker are recorded
in [portfolio_release.md](portfolio_release.md). No source licence or Ultralytics scope decision is silently
applied; public source redistribution remains separate. D5 acceptance and overall portfolio acceptance
remain distinct. Private correspondence and model/data distribution are excluded.
