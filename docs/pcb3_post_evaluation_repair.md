# PCB3: post-evaluation analysis-code repair — incident record

**Status: disclosed deviation. Written 2026-10-09 (+08:00), after the evaluation, from the repository
history and the development session's tool-execution log. Nothing here was backdated, no commit was
rewritten, and the repaired analysis program is not presented as unchanged since pre-registration.**

Summary in one paragraph: the pre-registered paired-analysis command (`ivad illum-compare`) was run on the
recorded PCB3 scores after the single planned evaluation and refused to run, because an input-validation
check added before the freeze wrongly assumed how VisA labels anomalous images. The check was corrected in
the analysis code (commit `7dbd302`) **after** the test and holdout scores had been generated and after
summary test metrics had been printed to the console. The correction changes only that check. It does not
touch model fitting, preprocessing, thresholds, prediction, the primary statistic or the decision rule; no
score was regenerated and no test prediction was re-run. An independent recomputation from the immutable
score files reproduces every recorded number (90/90 checks). The repaired function feeds only the
descriptive small/low-contrast subgroup table (protocol §8 C), which an independent script also reproduces
from the original ground-truth masks.

## 1. Freeze commit

`eb4e7a92c417e2c76750e8acb530e3bddbd04b2d` — committed 2026-10-09 16:58:10 +08:00, "FREEZE PCB3 baseline
and candidate before any test/holdout scoring". The calibration it records ran at `d8b5f1e` on the 150
CALIB normals only (`frozen_at_utc` 08:57:06 Z = 16:57:06 +08:00). The freeze froze the configs, split
manifest, thresholds, calibration scores and the whole of `src/`, including the analysis program
`src/ivad/illum.py` (git blob `70fb265`).

## 2. Evaluation chronology (all times 2026-10-09, +08:00)

| Time | Event | Evidence |
|---|---|---|
| 14:01:57 | Protocol pre-registered, before PCB3 data access | `d7f5e81` |
| 14:08:07 | Original analysis implementation (`defect_descriptors` without any label check) | `caca753` |
| 16:56:24 | Addendum B2 adds input validation to the analysis, including the faulty label check | `d8b5f1e` |
| 16:56:28–16:57:06 | Calibration on CALIB normals (both variants) | `frozen_operating_point.json` |
| 16:58:10 | **Freeze** | `eb4e7a9` |
| **16:58:51** | **First evaluation starts** (`evaluate` baseline, candidate; then `robustness` both) at HEAD `eb4e7a9` | `reports/pcb3_eval_run.log` line 1 |
| 17:00:17 / 17:02:39 | `metrics.json` written (baseline / candidate) | `evaluated_at_utc` fields |
| 16:59–17:07 | Console summary lines (AUROC, recall, FPR per condition) inspected while the run progressed | session log |
| 17:07:55 | Evaluation and robustness scoring end. **No test or holdout image was scored after this time.** | `pcb3_eval_run.log` last line |
| 17:09:13–17:09:56 | **Analysis invocation 1**: `python -m ivad illum-compare` (defaults `--base config/pcb3_baseline.yaml --new config/pcb3_illum.yaml`) at HEAD `eb4e7a9` → refused with `ValueError`; output shown on the terminal only | session log |
| 17:10:02 | Diagnosis: the `label` column of the PCB3 and PCB2 `image_anno.csv` files was tabulated (metadata only; no scoring) | session log |
| 17:10:20 | **Analysis invocation 2**: the same command, unchanged code, re-run to capture the refusal to a file → identical refusal | `reports/pcb3_illum_comparison/attempt1_refused.log` |
| 17:11:16–17:11:30 | Log sanitised (local absolute paths and library warning lines removed; see §3) | — |
| 17:11:40 | Raw evaluation outputs **and** the refusal log committed, before any code change | `115c54b` |
| 17:11:46–17:14:17 | Check corrected; synthetic fixture corrected; regression test added; full suite run (128 passed) | — |
| **17:14:22** | **Repair committed** | `7dbd302` |
| 17:14:2x–17:14:44 | Analysis invocation 3 at HEAD `7dbd302` → completed: ROBUSTNESS IMPROVEMENT SUPPORTED | session log |
| 17:15:17 | Post-hoc blur mechanism check on 50 **FIT** images (labelled post hoc; changes nothing) | `docs/pcb3_illumination_results.md` §2 |
| 17:15:38 | Paired-analysis outputs committed | `9c9f7c5` |

**Correction to earlier wording.** The file `attempt1_refused.log` and the commit message of `115c54b`
describe a single "attempt 1". There were in fact **two** refused invocations of the unchanged command
(17:09 and 17:10); the file holds the second. Both used the same code and the same inputs and refused at
the same line before writing anything. The file is left unchanged as evidence; this record supersedes its
"attempt 1" label.

## 3. Failed command and sanitised error log

Command (both invocations): `python -m ivad illum-compare` from the repository root, with the default
configs. Log as committed in `115c54b` (`reports/pcb3_illum_comparison/attempt1_refused.log`, SHA-256
`8dbb9cf21448dbf4327f59b339da3c58abd0505a091a27a1318dc6ab97ec8f64`):

```text
# ivad illum-compare, attempt 1 (2026-10-09 17:10:20 +0800), at commit eb4e7a9 — refused, no output produced
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "src\ivad\__main__.py", line 5, in <module>
    sys.exit(main())
  File "src\ivad\cli.py", line 125, in main
    illum_compare(load_config(args.base), load_config(args.new))
  File "src\ivad\illum.py", line 362, in illum_compare
    desc = defect_descriptors(cfg_cand, [r["image"] for r in an_rows])
  File "src\ivad\illum.py", line 140, in defect_descriptors
    raise ValueError(f"invalid mask metadata for {rel}: no anomaly row with a mask path in {anno_path.name}")
ValueError: invalid mask metadata for pcb3/Data/Images/Anomaly/000.JPG: no anomaly row with a mask path in image_anno.csv
```

Sanitisation: the absolute repository prefix (a local drive path) and the Python installation path were
removed from the traceback, and `Warning` lines emitted by third-party libraries were dropped. No line of
the traceback or of the evaluation summary was otherwise edited.

Root cause: VisA's `image_anno.csv` stores the **defect type** in `label` (`bent`, `melt`, `scratch`,
`missing`, and combinations such as `melt,scratch`) and `normal` for good boards; it never contains the
string `anomaly`. The check required `label == "anomaly"`, so it refused every real anomaly. The synthetic
end-to-end fixture written for Addendum B2 encoded the same wrong premise, so the test suite passed.

## 4. Original analysis code

- `caca753` (14:08:07, before PCB3 data access): `defect_descriptors` read the mask path with no label check.
- `d8b5f1e` (16:56:24, before calibration): Addendum B2 added refusals for missing rows, wrong labels,
  missing/empty/mis-sized masks and masks without a surrounding ring. **This is the version frozen at
  `eb4e7a9`** (`src/ivad/illum.py` blob `70fb265`, SHA-256 `e828fd7b…63ce7c`).

## 5. Repair commit

`7dbd302940bc7ceca3aee5f139adaca927459923`, committed 2026-10-09 17:14:22 +08:00 — 16 min after
evaluation started, 6 min 27 s after the last test/holdout image was scored. `src/ivad/illum.py` blob
`82d60b5` (SHA-256 `cf2949ad…c45b3`). It is the only change to `src/` between the freeze and the paired
analysis: `git diff --stat eb4e7a9 9c9f7c5 -- src config manifests` lists `src/ivad/illum.py | 5 +++--`
and nothing else.

## 6. Exact patch (source part; tests in §7)

Full sanitised diff: [`reports/pcb3_illum_comparison/repair_7dbd302.diff`](../reports/pcb3_illum_comparison/repair_7dbd302.diff)
(`git diff eb4e7a9 7dbd302 -- src tests`, SHA-256 `87a2de7e…2b0b82`).

```diff
@@ -136,8 +136,9 @@ def defect_descriptors(cfg: dict[str, Any], images: list[str]) -> dict[str, np.n
     area, contrast = [], []
     for rel in images:
         a = anno.get(rel)
-        if a is None or a.get("label") != "anomaly" or not a.get("mask"):
-            raise ValueError(f"invalid mask metadata for {rel}: no anomaly row with a mask path in {anno_path.name}")
+        # VisA image_anno.csv stores the defect TYPE in `label` (bent, melt, ...) and "normal" for good boards.
+        if a is None or a.get("label") in (None, "", "normal") or not a.get("mask"):
+            raise ValueError(f"invalid mask metadata for {rel}: no defect row with a mask path in {anno_path.name}")
         if not (root / a["mask"]).is_file():
             raise ValueError(f"invalid mask metadata for {rel}: mask file {a['mask']} not found")
```

## 7. Functions changed

| File | Function / test | Change |
|---|---|---|
| `src/ivad/illum.py` | `defect_descriptors` | the label condition of one refusal check (and its message); the mask-area and contrast computation below it is untouched |
| `tests/test_illum_e2e.py` | `_rows` (fixture) | anomaly rows now carry defect-type labels, as in VisA |
| `tests/test_illum_e2e.py` | `test_real_visa_annotation_format` (new) | regression: reads the real PCB3 `image_anno.csv` when present, else SKIPPED |
| `tests/test_illum_e2e.py` | `test_refuses_invalid_gt_mask_metadata` | new case `labelled_normal` (a row labelled `normal` is still refused) |

## 8. What the repair could and could not affect

| Stage | Affected? | Why |
|---|---|---|
| Model fitting | No | `illum.py` is imported only by the `photostats` and `illum-compare` commands, never by `fit`; fitted models predate the freeze (model hashes in `frozen_operating_point.json`) |
| Preprocessing (registration, correction, ROI) | No | `align.py`, `photometric.py`, `roi.py` unchanged since the freeze (`git diff eb4e7a9 HEAD` on these files is empty) |
| Thresholds | No | written at calibration, committed at `eb4e7a9`, never modified (one commit each) |
| Prediction / test scores | No | written by `evaluate`/`robustness` at `eb4e7a9`, committed at `115c54b`, never modified; not re-run |
| Primary statistic and decision rule | No | in the frozen `illum_compare`, the G1 difference-in-differences and `classify_illum` run at line 347, **before** `defect_descriptors` at line 362 (the line in the traceback); neither the rule nor its constants changed |
| Descriptive subgroup table (protocol §8 C) | Enabled, not altered | `defect_descriptors` produces the mask area and contrast used only for the size/contrast terciles. Before the repair this table could not be produced at all; the repair removed a refusal and changed no arithmetic |

During invocations 1 and 2 the primary statistics were computed in memory before the refusal, but nothing
was printed or written: the command prints its result only at the end. The first time the analysis result
was seen was invocation 3, after the repair was committed. Summary **test** metrics (per-variant AUROC,
recall and FPR, and the per-condition FPR) *had* been seen on the evaluation console before the repair, so
the repair was made with knowledge of those summaries. Because the repaired check cannot change any count,
rate, interval or outcome, this knowledge could not have been used to steer the result through the repair;
it is disclosed regardless.

## 9. Artifact hashes (SHA-256 of the committed files)

Each file below has been touched by exactly one commit (`git log -- <file>`), and the working-tree bytes
equal the committed blob.

| File | Commit | SHA-256 |
|---|---|---|
| `reports/pcb3_baseline/scores_calib.csv` | `eb4e7a9` | `36fe1eb8de6055c0cb2f2c07d43e80f13dcc2ff25d3d73143000deeb2d03e1da` |
| `reports/pcb3_baseline/frozen_operating_point.json` | `eb4e7a9` | `c0f9bd816f8a49a5d7777050e65984d6634ee89b7992dcdfc656e00e03170f89` |
| `reports/pcb3_baseline/scores_test.csv` | `115c54b` | `0a433c64a7806bc3567e22c6abcf61530a99f5fe8c45c44b7f99f681287a794d` |
| `reports/pcb3_baseline/scores_holdout.csv` | `115c54b` | `1c72c726a2b1dbaefdaf26cbdb5b1f73d04cfe745a982eea1f5bb28708780a29` |
| `reports/pcb3_baseline/robustness_scores.csv` | `115c54b` | `789de27782b8cb39d5ff4acb543f10e799dbe5cf22fc03ce24b6d3d4014a00fc` |
| `reports/pcb3_baseline/metrics.json` | `115c54b` | `1811ff9dc5897f9dd99ed65337618a4276ff5c875cd4b0c78d4f346ef2534643` |
| `reports/pcb3_baseline/robustness.json` | `115c54b` | `036d333aaba52a6ffdb54a54d6862b0a468ba32cbe78067b219c7ad9e6995d3e` |
| `reports/pcb3_illum/scores_calib.csv` | `eb4e7a9` | `f7a593bf285ff93cadddade8976b049b651714d41a06a4c3f1fb55cd3af76394` |
| `reports/pcb3_illum/frozen_operating_point.json` | `eb4e7a9` | `8803df04e3249405d84cc1b4ce9373c32eea613ee3eff1d77f79f329311ee86e` |
| `reports/pcb3_illum/scores_test.csv` | `115c54b` | `b42668d8753de3b6e16973b5589290b5946dc2fe4e378fee7dd161c63b53d1bf` |
| `reports/pcb3_illum/scores_holdout.csv` | `115c54b` | `9c40bbf9e7018ca6681b6e9822852b1397d0a9b1d49024abe420ab47c82236b6` |
| `reports/pcb3_illum/robustness_scores.csv` | `115c54b` | `9e5e659c09de5647e69e10b7b7b84f32fab08014d63c7c7ae6eeff7a78f6d2db` |
| `reports/pcb3_illum/metrics.json` | `115c54b` | `d1f7191f1d4b348314379966c9fd07ba2c44cfa73add47f5dbf0e9e1a2111aa9` |
| `reports/pcb3_illum/robustness.json` | `115c54b` | `67db67f484624296bff0145c7f36b0fec90dd7ecf4fac7f8d2aca4ae705e7e07` |
| `reports/pcb3_illum_comparison/comparison.json` | `9c9f7c5` | `6ea6092b169bdf2f4a01f081c7a959c88f8d19ab62eafbab1ea0cf34e7c2f35d` |
| `reports/pcb3_eval_run.log` | `115c54b` | `cbeda8e8f479e73d2ba32502498d7150fee1897848e3ac7e27ab805d522df2bc` |

The same table is regenerated by `python verification/recompute_pcb3.py` and checked by
`tests/test_pcb3_verification.py`.

## 10. Were scores regenerated or test predictions re-run?

No. `evaluate` and `robustness` were run once per variant (16:58:51–17:07:55). No command that scores a
CALIB, holdout or test image was run on PCB3 after 17:07:55. The three `illum-compare` invocations read the
recorded CSV files only. The post-hoc blur check (17:15) used 50 FIT images. Website images were produced
from PCB2 test images and PCB3 FIT images only.

## 11. Independent recomputation from the immutable score files

`python verification/recompute_pcb3.py` imports nothing from `src/ivad`. From the raw CSVs and the frozen
thresholds it re-derives: each threshold as the order statistic k = 144 of 150 calibration scores;
confusion counts, recall and FPR with exact intervals for the official test, the holdout and the 256 pooled
normals; AUROC; the absolute FPR and recall under every condition; the change from clean; the paired
difference-in-differences and its bootstrap interval; exact McNemar tests; the pre-registered outcome; the
original P1–P5 acceptance verdicts; and — from the official VisA masks, when the dataset is present — the
size and contrast terciles of question C.

Result (2026-10-09, dataset present): **90/90 checks agree with the recorded outputs, 0 disagree, 0
skipped** ([`reports/pcb3_verification/recompute.md`](../reports/pcb3_verification/recompute.md)). The G1
interval's upper bound stays below zero under ten other bootstrap seeds (−1.2 to −0.8 pts), so the rule
outcome does not depend on the random stream. In a copy without the dataset the subgroup check is reported
as SKIPPED.

## 12. Deviations from the freeze and post-test rules

| # | Deviation | Rule | Assessment |
|---|---|---|---|
| D1 | Addendum B1 committed ~5 min after fitting (pre-test) | §10 order | disclosed in Addendum B2.1; no CALIB/test image scored at the time |
| D2 | Addendum B2 added outside §10's reasons (pre-test) | §10 | disclosed in B2 itself; diagnostics on FIT only |
| D3 | **Analysis code changed after the freeze and after test scoring** (`7dbd302`) | freeze of `src/`; §9 bullet 1 | §9 forbids *score-affecting* changes; this one is not score-affecting (§8 above), but the analysis program that produced the result is **not** the frozen one. Disclosed here, in the protocol's post-test record, in the results document, README and website |
| D4 | Repair made after summary test metrics had been seen | §9 spirit | cannot change any number (§8, §11); disclosed |
| D5 | §9 "serious implementation defect → run invalidated" | §9 bullet 5 | judged **not applicable**: the defect was in the analysis program's input validation, not in the evaluated pipeline, and no score depended on it. The run was therefore not declared invalidated and not repeated. This is a judgment for the independent auditor to confirm or overrule |
| D6 | The refused analysis was invoked twice; the log labelled "attempt 1" is the second invocation | — | corrected in §2; both invocations used identical code and inputs |
| D7 | Reporting-code changes after test scoring: `ivad check` gates C5/C7 (`ef8480d`) and site/PDF builders | — | reporting only; disclosed in the release QA record and README |

No configuration, threshold, correction parameter, perturbation definition, split, fitted model or
score-affecting code changed after the freeze. No PCB4 data was used. The test was not repeated.
