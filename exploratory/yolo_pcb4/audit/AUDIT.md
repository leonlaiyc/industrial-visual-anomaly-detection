# Exploratory E1 (PCB4): audit record for independent review

**Status: prepared for independent review. This is an AI-assisted audit. No human has audited this study yet, and
owner review is still pending.**

The audit was written on 2026-10-10 (Taiwan time), after the experiment had finished. It retrains nothing, changes
no hyperparameter or threshold, and does not rescore any image to change a result. Every recorded file in
`../results/` is left as it was. Where the audit found a problem, the problem and its consequences are recorded here
and in `../results/deviations.md` (an appended section). The historical experiment has not been corrected.

Each script below states what it needs. The scripts marked **clean** run in a fresh checkout without the dataset or
the weights.

| File | What it shows | Needs |
|---|---|---|
| `recompute_metrics.py` → `recomputed_metrics.json`, `.md` | Every metric recomputed from the saved per-image scores, with no `ivad` import, and compared with `results.json` (50/50 match) | **clean** |
| `check_leakage.py` → `leakage_audit.json` | Fold membership, counts, label use and 70 construction and leakage checks (all pass) | **clean** |
| `fold_manifest.csv` | Every image's partition in all 12 YOLO runs, read from the list files actually used for training | exported |
| `yolo_labels.csv` | All 488 training boxes actually used, re-derived from the masks (0 mismatches) | exported |
| `dataset_identity.json` | Geometry, exact and near-duplicate hashes, and the official VisA splits compared with the E1 roles | dataset |
| `fig_annotation_check.jpg` | Training boxes (red) over the mask outlines (cyan) for 12 images, including multi-type images | dataset |
| `localization_audit.py` → `localization_audit.json`, `flagging_boxes.csv` | Reproduction of the post-hoc check with frozen weights (inference only), plus stricter definitions | weights + dataset |
| `acquisition_confound.py` → `acquisition_confound.json` | Defect-blind image cues, and whether the saved scores follow them | dataset |
| `export_training_record.py` → `training_record.json`, `train_curves/` | Checkpoint epoch, early stopping, resolved optimizer, augmentation, weight hashes | run directories |
| `build_manifests.py` | Builds the exported files listed above | dataset + run directories |

`tests/test_exploratory_yolo.py` reruns the two **clean** scripts into a temporary directory. It checks that their
output equals the committed JSON, and that the localisation audit reproduces the original post-hoc counts.

## 1. Chronology (from git, file timestamps and the development session log)

| Time | Event |
|---|---|
| 13:36 | The archive member list for `pcb4/` was read (names only). |
| 13:37 | `pcb4/image_anno.csv` was read from the archive (labels and defect types, no pixels). `data/visa/` held only pcb1–pcb3. |
| **13:38:37** | **Protocol committed (`d5c00ab`).** It is unchanged since. |
| 13:39:37–53 | `prepare.py` was written and edited twice, all before its first run. |
| 13:39:58–59 | PCB4 was extracted (all 1,206 files carry these mtimes), and the split, the audit and the configurations were written. This was the only run. |
| 13:40:52 | `score_reference.py` started (fit and score of M, then M-reg; done 13:42:51). Its script and configurations date from 13:40 and are unchanged in the commit. |
| 13:41:52 | Split and scripts committed (`809b190`). The code of `prepare.py`, `train_yolo.py` and `score_reference.py`, both configurations and the split files are unchanged since. The only later edit is the redaction, during the audit, of a local virtual-environment path in one docstring usage line of `train_yolo.py` (and of `posthoc_localization.py`); it changes no code. The normal-only fit had started 60 s earlier. |
| 13:43:44 | YOLO training started; the 12 runs ended between 13:52 and 15:14 (`train_yolo.log`). |
| 13:53 | During training, the session displayed YOLOv8s fold-1 hyperparameters, its test-fold mAP50 (0.547) and its latency. No code, setting or protocol changed afterwards. |
| 13:57 | `analyze.py` written (committed 14:02 in `bfe7dd3`). It could not run until all 12 runs existed. |
| 15:15 | First analysis run. Its output is identical to the committed `results.md`. |
| 15:16 | Post-hoc localisation check written, after the results were seen, and logged as post hoc. |
| 15:41 | `analyze.py` edited to change only the format of the CI strings, then rerun. The numbers are unchanged. |
| 15:43 | Local paths redacted from `train_args_*.json` and `env.txt` (text replacement only). |
| 15:44:25 | Results committed (`b536796`). |

No PCB4 pixel was read before the protocol commit. The labels CSV and the member list were read before it, as the
protocol itself states.

## 2. Dataset and split

- 1,105 images (1,005 normal, 100 defective), all 1358×1104. All masks match their image size. The SHA-256 values
  recorded in `split.csv` still match the files. There is no exact duplicate anywhere. Near duplicates were checked
  with a dHash (16×16, 256 bits): no cross-role pair lies within 10 bits, and the nearest cross-role distance is 19.
- Roles: FIT 600, CALIB 150, TEST-N 255, folds 1–5 with 20 defects each.
- Defect types are multi-label, and 56 of the 100 images carry two or more types. Labels are one class (`defect`),
  with one box per 8-connected mask component of at least 4 px. That gives 488 boxes in total, from 1 to 63 per
  image (26 images have one box). Types therefore play no part in the labels. They are used only for fold
  stratification (by the first listed type) and for the unseen-type runs.
- The official VisA `1cls` split is not used (it puts all defects in test). E1 TEST-N holds 223 official-train and
  32 official-test normals; FIT holds 53 official-test normals. The supervised VisA splits (`2cls_highshot`,
  `2cls_fewshot`) are not used either. The cross-tabulations are in `dataset_identity.json`.

## 3. Five-fold construction and leakage (`leakage_audit.json`, 70/70 checks pass)

| Run | Train defects / normals | Internal-val defects / normals | Test defects | CALIB | TEST-N | Boxes train / val / test | Multi-type train / val / test |
|---|---|---|---|---|---|---|---|
| cv_fold1 | 64 / 128 | 16 / 32 | 20 | 150 | 255 | 389 / 44 / 55 | 36 / 9 / 11 |
| cv_fold2 | 64 / 128 | 16 / 32 | 20 | 150 | 255 | 327 / 71 / 90 | 39 / 7 / 10 |
| cv_fold3 | 64 / 128 | 16 / 32 | 20 | 150 | 255 | 282 / 76 / 130 | 35 / 9 / 12 |
| cv_fold4 | 64 / 128 | 16 / 32 | 20 | 150 | 255 | 301 / 69 / 118 | 37 / 7 / 12 |
| cv_fold5 | 64 / 128 | 16 / 32 | 20 | 150 | 255 | 336 / 57 / 95 | 35 / 10 / 11 |
| unseen_missing | 54 / 128 | 13 / 32 | 19 | 150 | 255 | 297 / 74 / 44 | 32 / 10 / 0 |
| unseen_extra | 56 / 128 | 14 / 32 | 12 | 150 | 255 | 345 / 41 / 14 | 32 / 6 / 0 |

- Within each run, train, internal validation and test are disjoint. CALIB and TEST-N never enter training or
  validation. Every run uses the same 160 FIT background normals.
- Each of the 100 defects receives exactly one out-of-fold prediction per architecture, and the score files match
  the list files.
- **Reuse:** the same 150 CALIB normals set every threshold, and every model scores the same 255 TEST-N normals.
  Pooled YOLO FPR is therefore 1,275 evaluations of 255 unique images, not 1,275 independent observations. The
  intervals in §5 resample unique images.
- **Label use:** box labels of each run's training and internal-validation defects are permitted supervision. Each
  run's test-defect labels and masks were used only for recall, mAP50 and the post-hoc check. Thresholds use CALIB
  scores only. The checkpoint (`best.pt`) is chosen on internal-validation fitness, which is mAP50-95.

## 4. YOLO implementation (`training_record.json`, `../results/train_args_*.json`, `../results/env.txt`)

- Task: **detection** (`task: detect`), not segmentation. Weights are the official COCO-pretrained `yolov8s.pt`
  (SHA-256 `1f47a78b…`) and `yolo11s.pt` (`85a76fe8…`). Software: ultralytics 8.4.175, torch 2.14.1+cu126, on an
  RTX 4070 Ti (driver 591.86).
- Settings: `imgsz=1024`, `epochs=100`, `patience=30`, `batch=8`, `seed=0`, `deterministic=True`
  (`warn_only`, with AMP on, so the runs are not guaranteed bit-reproducible) and `workers=4`. `optimizer=auto`
  resolved to **AdamW (lr 0.002, momentum 0.9)** in all 12 runs, so the `lr0=0.01` setting was ignored.
- Default augmentation: mosaic 1.0 (closed for the last 10 epochs), HSV (h 0.015, s 0.7, v 0.4), translate 0.1,
  scale 0.5, fliplr 0.5. The `auto_augment` and `erasing` arguments apply only to classification and are inactive
  for detection.
- Checkpoint epochs (best of 100): v8s 93/95/83/84/95, 11s 87/99/94/96/80, unseen_missing 94, and unseen_extra
  48. unseen_extra stopped early at epoch 78.
- Image score: the maximum box confidence from `predict(conf=0.001, iou=0.7, imgsz=1024)`, or **0 when no box
  survives**. Many normals score 0; for YOLOv8s that is 13–65 of the 150 CALIB images per fold. Thresholds (the
  144th of 150 CALIB scores) are therefore low box confidences: v8s 0.021–0.078 and 11s 0.035–0.231. They were set
  by a fixed rule on normal images only, with no test-label feedback.
- All runs followed the protocol's hyperparameters, and no run crashed or was repeated.

## 5. Recomputed metrics (`recomputed_metrics.md`)

| Method | TP / FN | Recall (Clopper–Pearson 95%) | FP / TN | FPR | FPR denominator | Cluster bootstrap 95%, thresholds fixed (recall; FPR) |
|---|---|---|---|---|---|---|
| M, no registration | 60 / 40 | 60% (50–70%) | 11 / 244 | 4.3% | 255 images | 50–69%; 2.0–7.1% |
| M-reg | 50 / 50 | 50% (40–60%) | 19 / 236 | 7.5% | 255 images | 41–60%; 4.3–10.6% |
| YOLOv8s | 98 / 2 | 98% (93–100%) | 51 / 1,224 | 4.0% (folds 2.0–5.5%) | 1,275 evaluations of 255 images | 95–100%; 2.5–5.7% |
| YOLO11s | 95 / 5 | 95% (89–98%) | 39 / 1,236 | 3.1% (folds 1.6–3.5%) | 1,275 evaluations of 255 images | 90–99%; 1.9–4.3% |

- When CALIB is also resampled and the thresholds recomputed, the intervals widen considerably: M recall 37–74%,
  YOLOv8s FPR 1.2–7.7%. This is because the 144th-of-150 rank is noisy; per threshold, the marginal FPR follows
  Beta(7, 144), with a 95% range of 1.9–8.5%.
- Across the five fold models, 33 (YOLOv8s) and 27 (YOLO11s) of the 255 unique normals were flagged at least once.
- **YOLOv8s against YOLO11s:** the two disagree on 5 of the 100 defects (4 caught only by YOLOv8s, 1 only by
  YOLO11s; exact McNemar p = 0.375). The paired Δrecall interval is −1 to +8 pts with thresholds fixed and −3 to +7
  recalibrated. The two architectures cannot be ranked on this sample.
- YOLO compared with M: +38 and +35 pts recall (38/0 and 38/3 discordant pairs), and ΔFPR −0.3 and −1.3 pts with
  intervals spanning 0.

## 6. Fairness of the comparison

The detectors were trained on box labels for about 64 defects of the same board and the same defect types they
were tested on. M saw no defect. M was also run without any PCB4 adaptation, by protocol. This study compares
**learning regimes with different label budgets**, not architectures. It supports one statement: when
representative labelled defects of the expected types exist, a small supervised detector can be a strong
candidate for this review task on this board. It does **not** support "YOLO is generally 38 points better than
normal-only anomaly detection". The PCB4 results are separate from the confirmatory PCB1–PCB3 findings and change
none of them.

## 7. Post-hoc localisation check (`localization_audit.json`)

- **Original definition** (`../posthoc_localization.py`): for each out-of-fold defect that was flagged, the
  **highest-confidence box** is the box that set the image score. It counts as on-defect if it shares at least
  one pixel with the ground-truth mask **dilated by 16 px**. **No IoU is required.** "Any" means any box above the
  fold threshold. Images that were not flagged are excluded. Numerator and denominator: flagged images with an
  on-defect top box, divided by flagged images.
- The audit reproduced this check with the frozen weights (inference only). Results: v8s **92/98** top, 94/98 any;
  11s **89/95** top, 93/95 any, which exactly matches the original. The re-predicted maximum confidence differs
  from the saved score by at most 0.00064, and no flag changes.
- Stricter definitions applied to the same boxes:

  | Definition | YOLOv8s | YOLO11s |
  |---|---|---|
  | top box touches the undilated mask | 92/98 | 89/95 |
  | centre of the top box inside the dilated mask | 89/98 | 83/95 |
  | top box IoU ≥ 0.1 with a ground-truth component box | 90/98 | 85/95 |
  | top box IoU ≥ 0.5 | 79/98 | 74/95 |
  | any above-threshold box IoU ≥ 0.5 | 86/98 | 85/95 |

- In 6 flagged images per architecture, the flag came from a box off the defect: a background or other box set
  the score. The "touch" rule is lenient, and top boxes cover a median 0.4–0.5% of the frame (at most 4.6%), so it
  is not a degenerate whole-image box. The check was added after the results were seen. It is **not** a
  pre-registered endpoint and **not** a localisation-accuracy metric.

## 8. Acquisition cue found by the audit (a genuine methodological issue)

`acquisition_confound.json` measures cues that do not depend on the defect:

- The PCB4 defect images are **darker** than the TEST-N normals (median grey level 46.8 against 51.5).
  Brightness alone ranks defect above normal with AUROC **0.79**, and board centroid x reaches 0.69. Defects cover
  too little of the frame to explain this, so it reflects how the images were acquired.
- The YOLO scores follow the cue **where no defect exists**. Over the 255 normals, the Spearman correlation between
  grey level and score is −0.69 for YOLOv8s and −0.74 for YOLO11s (darker images score higher). For M it is +0.19.
- Consequence, computed from the saved scores and the thresholds actually used, with TEST-N reweighted to the
  defect images' brightness: FPR is **9.0%** for YOLOv8s, **6.9%** for YOLO11s, 2.1% for M and 6.0% for M-reg. The
  reported YOLO FPR of 3–4% therefore understates the false-alarm rate at defect-like brightness.
- With thresholds calibrated on brightness-matched CALIB normals (weighted P95; the effective sample is small and
  noisy), recall is **96%** for YOLOv8s, **91%** for YOLO11s and 63% for M, at a brightness-matched FPR of 3.0%,
  2.1% and 4.4%. The large recall gap therefore does not appear to be explained by brightness. Session cues other
  than brightness are not excluded.
- Together with §7, the evidence is consistent with mostly defect-based detection plus a measurable sensitivity
  to acquisition. This was not designed into the protocol and is recorded here, not corrected.

## 9. Unseen defect type (E7)

- Held out: `missing` (19 single-label test images) and `extra` (12). Every training or validation image carrying
  the type was excluded. Multi-label images with the type (14 for `missing`, 18 for `extra`) are in **neither**
  training nor test. Checkpoint selection never saw the withheld type. The same images are compared across arms.
- `missing`: 19/19 when the type was seen (5-fold models), 11/19 when unseen (Clopper–Pearson 33–80%), 10/19 for
  M. `extra`: 10/12, 9/12 and 7/12.
- Two confounds: the seen and unseen arms are different models, and the unseen runs trained on fewer defects (67
  and 70 against 80).
- **Audit finding:** the unseen-`missing` model flagged 11 images. Only 6 of their top boxes touch the defect, and
  **none** of their boxes reaches IoU ≥ 0.5; mAP50 on that run is 0.0. Its image-level hits are therefore not
  detections of the missing part. For `extra`, 9 were flagged, 3 have a top box at IoU ≥ 0.5 and 6 have any box at
  IoU ≥ 0.5. This is a small exploratory sensitivity analysis and makes no claim of zero-shot or novel-defect
  performance.

## 10. Latency comparability

| Pipeline | Median | Boundary | Input | Registration | Device / batch | Warm-up / n | Sync |
|---|---|---|---|---|---|---|---|
| YOLOv8s (fold-1 model) | 14.8 ms | file path → ultralytics `predict` (decode, letterbox, forward, NMS) → max conf | 1024 letterbox | no | GPU, 1 | 5 / 50 | `cuda.synchronize` before and after |
| YOLO11s (fold-1 model) | 14.0 ms | same | 1024 letterbox | no | GPU, 1 | 5 / 50 | same |
| M | 31.6 ms | file path → decode → resize, normalise → CNN → Mahalanobis map → smoothing → max | 768×640 | **no** | GPU, 1 | 5 / 50 | same |
| M-reg | 135.0 ms | same, **plus registration** | 768×640 | **yes** | GPU, 1 | 5 / 50 | same |

The boundaries are broadly similar (file to score, batch 1, synchronised), but the input size, the
postprocessing and the presence of registration differ. Each figure is one run of 50 images, and YOLO was timed
with one fold model only, so **no speed ranking is supported**. The public text no longer claims one; the
per-pipeline figures stay, with their limitations. Record error: the `definition` string in `latency_m.json` says
"register + crop", but M ran with `register: false` and `crop_wh: null` (the string is a fixed template in
`ivad.pipeline._measure_latency`). The measured time excludes registration.

## 11. Other discrepancies found

1. The protocol §7 asks for a contact sheet of misses **and of false alarms**. Only misses were produced
   (`fig_failures.jpg`), so this deliverable is missing.
2. `train_args_*.json` records `best_epoch: -1` because the stripped checkpoint has no epoch. The actual epochs are
   in `training_record.json`.
3. `posthoc_localization.py` looks up a `thresholds_by_fold` key that `results.json` does not contain, and falls
   back to the 144th CALIB score. That is the protocol threshold, so there is no effect.
4. The site text claimed "so the recall is not a shortcut such as capture-session cues" and "it was also faster".
   Both were removed or qualified (EN and zh-TW), with the audit numbers above added.
5. `deviations.md` says "before any model was trained" at 13:41. The normal-only reference fit had started at
   13:40:52; YOLO training started at 13:43:44.

## 12. Still to be reviewed independently

- A human or independent review of the code paths in `prepare.py`, `train_yolo.py`, `analyze.py` and these audit
  scripts. Everything above is AI-assisted.
- Retraining from scratch with the official weights, to see how much the 12 runs vary under nondeterminism. Not
  done here, by instruction.
- Whether the brightness cue reflects a VisA acquisition-session artefact on PCB4, and whether a
  brightness-matched or session-matched design changes the YOLO false-alarm rate. That would need new data or a
  new protocol.
- A visual review of `flagging_boxes.csv` for the off-defect flags, and of the unseen-type flags.
