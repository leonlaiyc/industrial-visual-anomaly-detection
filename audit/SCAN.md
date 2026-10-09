# Pre-push scan of every staged file (2026-10-09)

Run over `git ls-files` of this branch (265 files, 44.3 MB) immediately before the commit.

| Check | Pattern / method | Result |
|---|---|---|
| Local machine paths | drive-letter paths, user-profile and temp directories, sibling workspace names | none (one hit: this audit's own statement of what was excluded, since reworded) |
| Private material | owner-review folder, correspondence and interview-context names, personal mail domains | none |
| E-mail addresses | generic address regex | none |
| Secret-like strings | AWS / GitHub / OpenAI / Slack token formats, private-key headers, `api_key=`, `password=`, `token=` | one hit: the detector regex itself in `src/ivad/consistency.py` |
| Dataset, weights, archives | `ivad check` C7 and file listing | none tracked; no `data/`, `artifacts/`, `*.pt` |
| Large files | > 1.5 MB | six pre-existing report figures already on `main` (≤ 2.0 MB); largest new file 1.0 MB |
| Embedded metadata | PDF info dictionary; PNG/JPEG text chunks and EXIF | PDF: title, HeadlessChrome/Skia producer, creation time; figures: Matplotlib version; no EXIF, no names, no paths |
| Copied content equals development outputs | blob hashes of all 203 copied files vs development commit `e397094` | identical |
| Built site | `audit/site_dist` vs a fresh `python site/build.py` in this export and in the development repository | identical |
