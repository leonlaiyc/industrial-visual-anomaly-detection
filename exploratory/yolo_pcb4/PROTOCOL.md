# Exploratory study E1: supervised detectors (YOLOv8s, YOLO11s) vs. the normal-only approach on VisA PCB4

**Status: EXPLORATORY. Written and committed before any PCB4 image was extracted or viewed.**
Only the PCB4 annotation file (`pcb4/image_anno.csv`, image names and defect-type labels) and the archive
member list were read to write this protocol.

This study is separate from the frozen v2 work. It changes no PCB1, PCB2 or PCB3 artefact, threshold, score,
acceptance criterion, verdict or conclusion, and its results cannot change any of them. It has its own data
(PCB4), split, code (`exploratory/yolo_pcb4/`), software environment and artefact directory
(`artifacts/exploratory_yolo_pcb4/`, not in git).

## 1. Question

With a small budget of box-labelled defects (about 64 per training fold), how do off-the-shelf supervised
detectors compare with the project's normal-only approach **at the project's task**: flag an image for human
review, at a threshold set on separate normal images to target about 5% false alarms, on a board type never
used before?

A secondary question: what happens when a supervised detector meets a defect type it was not trained on? The
project's model-selection table currently states this as a risk of object detectors without having tested it.

This is a **suitability probe, not a benchmark**. The methods use different supervision (see §8), so no
result here can show that one approach is generally better.

## 2. Why PCB4

The official test sets of PCB1, PCB2 and PCB3 are spent. Training a detector on their defects, or choosing
anything by looking at them again, would not be independent. PCB4 has never been used in this project.

## 3. Data

VisA `pcb4`, from the official archive already on disk (`data/raw/VisA_20220922.tar`, the same file
`ivad prepare` verifies), extracted to `data/visa/pcb4/` (gitignored). Labels from `pcb4/image_anno.csv`:
1,005 normal and 100 defective images. Defect types are multi-label (for example `missing,dirt`). Masks
come from the archive.

The official VisA 1-class split is **not** used, because it puts every defect in the test set, so a
supervised method could not be trained. The split below replaces it for this study only.

## 4. Split (fixed by `prepare.py`, seed 20261010, before any image is viewed)

| Role | Content | Used by |
|---|---|---|
| FIT | 600 normals | normal-only model fit; YOLO background images are drawn from here |
| CALIB | 150 normals | every model's threshold; never used for training |
| TEST-N | remaining normals (expected 255) | false-alarm rate, AUROC |
| FOLD 1–5 | the 100 defects, 20 per fold, stratified by the first listed defect type | out-of-fold recall |

- Normals are shuffled with the seed and assigned in the order FIT, CALIB, TEST-N.
- Defects are sorted by (first listed type, image name), shuffled within type with the seed, then dealt
  round-robin into folds 1–5.
- Audits (the script fails if any fails): roles are disjoint; counts are as stated; no exact-duplicate file
  (SHA-256) appears in two roles; every defect has a mask.

## 5. Methods

### 5.1 Supervised detectors: YOLOv8s and YOLO11s

- Software: `ultralytics` (AGPL-3.0), COCO-pretrained official weights `yolov8s.pt` and `yolo11s.pt`,
  installed in a separate virtual environment. Not added to the project's locked requirements. Versions
  are recorded in `results/env.txt`.
- **Labels:** one class, `defect`. Each 8-connected component of a ground-truth mask with an area of at
  least 4 px becomes one box (tight bounding rectangle, no padding).
- **Per fold k (5-fold cross-validation):** the training pool is the 80 defects of the other folds plus 160
  normals drawn once from FIT (seeded; the same 160 for every fold and both models). Ultralytics needs a
  validation set for checkpoint selection, so a seeded 20% of that pool (16 defects + 32 normals) is held
  out as internal validation. The remaining 64 defects + 128 normals are trained on. The fold's 20 test
  defects, CALIB and TEST-N are never in the training or internal validation lists (asserted).
- **Hyperparameters, fixed in advance and identical for both models:** `imgsz=1024`, `epochs=100`,
  `patience=30`, `batch=8`, `seed=0`, `deterministic=True`, `workers=4`, everything else at the
  ultralytics defaults (optimizer `auto`, default augmentation). Checkpoint: ultralytics `best.pt` (its own
  fitness on the internal validation set). **No tuning.** A run is repeated only if it crashes, and the
  repeat is logged.
- **Image score:** the maximum box confidence from `predict(conf=0.001, iou=0.7)` at `imgsz=1024`, or 0 if
  there is no box.

### 5.2 Normal-only reference: the project's pipeline, unchanged code

- **M (primary reference):** the PCB2 v1 model, preprocessing and calibration settings (frozen CNN features,
  per-position Gaussian, full-frame max), with two pre-specified changes: `register: false` and
  `crop_wh: null`. Registration and the board region were developed on PCB1–PCB3 backgrounds. No
  PCB4-specific preprocessing has been developed, and none will be, because that would be new development
  on PCB4. `size_hw` stays `[512, 768]` if the PCB4 aspect ratio (W/H) lies within 1.5 ± 0.05; otherwise the
  height is the multiple of 32 closest to 768 ÷ aspect. This is decided from image dimensions only.
  M is fitted once on FIT (it uses no defects) and scores every image once.
- **M-reg (secondary, conditional):** the same settings as M, but with `register: true` and the original
  board-mask rule. It is run **only if** a FIT-only check passes: the original rule's board mask covers
  between 5% and 80% of the frame on at least 95% of FIT images. If the check fails, M-reg is reported as
  "not run: registration rule does not transfer to PCB4". That outcome is itself a portability
  observation, as on PCB3.

## 6. Threshold (same rule for every model)

The P95 split-conformal threshold on the 150 CALIB normal scores (`ivad.calibration.conformal_threshold`,
alpha 0.05): the 144th of 150 sorted scores. An image is flagged iff its score is greater than the
threshold. Each YOLO fold model gets its own threshold from its own CALIB scores.

## 7. Endpoints (all descriptive; there is no confirmatory hypothesis)

| ID | Endpoint |
|---|---|
| E1 | Image-level recall at P95 on the 100 defects: YOLO out-of-fold (each defect scored by the fold model that did not train on it, at that fold's threshold); M and M-reg on the same 100 images. Clopper–Pearson 95% CI. |
| E2 | False-alarm rate at P95 on TEST-N: M single value; YOLO mean over folds, with the per-fold range. |
| E3 | AUROC, defects vs. TEST-N: M on all images; YOLO mean of the per-fold AUROCs (each fold's 20 defects vs. TEST-N). |
| E4 | E1 by defect size: terciles of total mask area over the 100 defects. |
| E5 | Latency: median per-image GPU time, batch 1, from file path to score, on 50 TEST-N images after 5 warm-up images. Indicative only. |
| E6 | Box-level mAP50 on each fold's test defects (ultralytics `val`), YOLO only. Descriptive. |
| E7 | **Unseen defect type** (secondary, very low power), YOLO11s only. For T in {`missing`, `extra`}: train on every defect whose labels do not include T (same normals, same hyperparameters, seeded 20% internal validation), threshold on CALIB, and report recall on the single-label T images (19 `missing`, 12 `extra`). On the same images, also report the recall that the 5-fold YOLO11s (T seen) and M achieved. |
| P | Paired differences YOLO − M on the same images: recall (100 defects) and false-alarm rate (TEST-N), with paired bootstrap 95% CI (2,000 resamples, seed 7). Descriptive. |

Failure examples: the defects each method misses at P95 (image, defect type, mask area), plus a small
contact sheet of misses and of false alarms, saved in `results/`.

## 8. Interpretation rules, fixed now

1. **Different supervision.** In E1–E6 the detectors see box labels for about 64 defects of the same board
   and the same defect types that they are tested on. M sees only normal images. A higher YOLO recall would
   show what this label budget buys **on PCB4, for defect types seen in training**. It would not show that
   detection is better in general, and it would not apply to defect types absent from training (E7 probes
   that, weakly).
2. **Small samples.** 100 defects, 20 per fold and 255 test normals. Differences of a few points are within
   noise. E7 uses 19 and 12 images and can show only gross effects.
3. **Not comparable with PCB1–PCB3 numbers.** PCB4 is a different board type with its own data. Nothing here
   is a chronological improvement over v2, and nothing here changes v2's recorded results.
4. **Benchmark only.** Public benchmark images, no factory data, no production claim.
5. **No post-hoc rescue.** If a setting turns out to be poor (for example `imgsz`), that is reported as a
   result. It is not re-tuned on these test folds. A follow-up would need new data or a nested design.

## 9. Leakage safeguards

- Roles are disjoint and checked by SHA-256; YOLO list files are generated from the split and asserted
  against it.
- CALIB and TEST-N are never in any training or internal-validation list.
- Hyperparameters are fixed above. Any deviation is logged in `results/deviations.md` with a timestamp.
- All raw per-image scores are saved, so every reported number can be recomputed.

## 10. Outputs

`exploratory/yolo_pcb4/results/`: `split.csv`, `split_audit.json`, per-image score CSVs, `results.json`,
`results.md`, `env.txt`, `deviations.md` and failure-example images (VisA derivatives, CC BY 4.0). Model
weights and training runs go to `artifacts/exploratory_yolo_pcb4/` and are not committed.
