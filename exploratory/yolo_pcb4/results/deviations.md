# Exploratory E1: deviations log

Any departure from PROTOCOL.md is recorded here with a timestamp (Taiwan time).

(no deviation before training; written 2026-10-10 13:41, before any model was trained)

## 2026-10-10 15:16 — post-hoc localisation check added (not an endpoint)

After seeing the E1 results (YOLO recall far above the normal-only reference), a post-hoc check was added:
for each out-of-fold defect a YOLO fold model flags, does the box that triggered the flag touch the
ground-truth defect (mask grown by 16 px)? Purpose: rule out a shortcut such as capture-session cues behind
the image-level recall (cf. the PCB1 orientation confound). It changes no endpoint, threshold or score and is
reported separately as post hoc (`posthoc_localization.py`, `posthoc_localization.json`).

Note on the optimizer: the protocol left it at the ultralytics default `auto`; ultralytics resolved `auto` to
AdamW for these runs (recorded in `train_args_*.json`). Not a deviation; recorded for transparency.

## 2026-10-10 (evening) — audit addendum (appended; nothing above or in results/ was changed)

An AI-assisted audit after the experiment (`../audit/AUDIT.md`) retrained nothing and changed no threshold or score.
It recomputed every recorded number from the saved scores (50/50 match) and passed 70/70 fold and leakage checks.
It recorded these issues instead of correcting them:

- **Acquisition cue.** The PCB4 defect images are darker than the test normals (brightness alone gives AUROC 0.79),
  and the YOLO scores on normal images rise with darkness (Spearman −0.69 and −0.74). At defect-like brightness,
  the YOLO false-alarm rate is 9.0% and 6.9% instead of the reported 4.0% and 3.1%. Recall with brightness-matched
  thresholds stays at 96% and 91%. This was not anticipated by the protocol.
- **Post-hoc localisation** uses "top box touches the mask dilated by 16 px", with no IoU. At IoU ≥ 0.5 the counts
  are 79/98 and 74/95. The unseen-`missing` hits (11/19) include no box at IoU ≥ 0.5.
- **Normal-image reuse.** The same 255 test normals and 150 calibration normals serve every model. The pooled YOLO
  FPR covers 1,275 evaluations of 255 images. The intervals in the audit resample unique images.
- **Missing deliverable.** Protocol §7 asks for a false-alarm contact sheet; it was not produced.
- **Record errors.** The latency `definition` string for M names registration and crop that did not run.
  `best_epoch` was recorded as −1; the actual epochs are in `../audit/training_record.json`.
- **Latency.** The figures use different input sizes and boundaries and do not support a speed ranking. The
  public claim was removed.
- **Display during training.** At 13:53, during training, one test-fold metric (YOLOv8s fold-1 mAP50) was
  displayed. No protocol, code or setting changed afterwards.
