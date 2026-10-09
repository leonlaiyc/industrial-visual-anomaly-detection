# v2 experiment protocol (pre-registration)

**Status:** pre-registered. Committed before any VisA PCB2 image was loaded, fitted, calibrated or
scored. Later sections marked *Addendum* may only be added before PCB2 test scoring, and only for the
reasons allowed in §9.

## 1. Origin of the hypothesis: PCB1 is discovery data

v1 (tag `v1.0-pcb1`) was evaluated once on the official VisA PCB1 test split under a threshold frozen
in advance. It missed the proposed recall target (72/100 at the primary P95 threshold). Post-hoc
diagnostics on that same evaluation found that all 13 false alarms on normal boards (7 official-test,
6 supplementary-holdout) had their anomaly-map maximum on the background cloth, not on the board.

The v2 hypothesis was formed by looking at those PCB1 test results. **PCB1 is therefore discovery data
for v2, not confirmatory data.** Any v2 number computed on PCB1 is retrospective and will be labelled
as such. It can never count as evidence for the hypothesis.

## 2. Scientific hypothesis

> Off-board background texture contributes disproportionately to high anomaly scores. Restricting
> image-level anomaly aggregation to a board-only region may reduce background-driven false alarms and
> permit better anomaly recall at calibrated operating thresholds.

Mechanism: the operating threshold is an upper quantile of calibration-normal scores. If some of those
normal scores are set by background dust or lint, the threshold rises and subtle on-board defects fall
below it. Ignoring the background should lower the calibrated threshold without losing board evidence.

## 3. The single permitted modelling change

| | v1 (baseline) | v2 |
|---|---|---|
| Image score | max of the smoothed anomaly map over the **whole** model frame (768×512) | max of the same map over a fixed **board ROI** |

Nothing else changes. The following stay identical between v1 and v2, and identical to the PCB1 v1
configuration:

- pretrained backbone (torchvision Wide-ResNet-50-2, IMAGENET1K_V1), layers (layer2 + layer3);
- feature dimensionality (128 random channels, seed 0) and 3×3 local averaging;
- Gaussian fitting (per-position mean and covariance, float64 accumulation, Cholesky whitening);
- covariance regularisation (ridge 0.01 · mean diagonal + 1e-6);
- registration method, crop size, input size (crop 1152×768 → 768×512), map smoothing (σ = 4 px);
- calibration rule (split-conformal), candidate thresholds (P90, P95, P99), primary threshold (P95);
- seeds, bootstrap settings, robustness perturbations and the acceptance criteria file.

The only exception is a dataset-specific adaptation that is *essential* to run PCB2 at all (§9).

Because v1 and v2 share the per-pixel anomaly map, the v2 fitted Gaussian state must be bitwise
identical to the v1 fitted Gaussian state on the same data. This is checked automatically.

## 4. ROI construction

The ROI is one binary mask in the model frame (512×768). It is built once, at fit time, from FIT
normal images only, by fixed geometric rules. It is stored inside the fitted model file, so it is
covered by the model hash in the freeze.

1. **Per-image board mask.** For each FIT image: register and crop exactly as for scoring
   (`Preprocessor.view`). Compute the board mask with the existing registration function
   `align.foreground_mask` (HSV rule, largest component, 1/4 scale). Upsample it to 512×768
   (nearest neighbour).
2. **Consensus.** For each pixel, compute the fraction of FIT images whose mask covers it. Keep pixels
   covered in **at least 50%** of FIT images. Keep the largest connected component and fill holes.
3. **Margin.** Dilate the consensus by **16 model pixels** (24 native pixels). This margin absorbs
   residual registration jitter at the board edge and parts that protrude from the board outline. 16 px
   equals the tolerance already used in v1 diagnostics. It is fixed here, before PCB2 is seen.
4. **Never used:** ground-truth defect masks, anomalous images, calibration/holdout/test images, test
   outcomes.

### ROI quality checks (on FIT images; any failure stops v2 — no ad-hoc repair)

| Check | Rule |
|---|---|
| Non-empty, single region | ROI has ≥ 1 pixel and is one connected component |
| Plausible size | ROI area is between 15% and 90% of the model frame |
| Board not discarded | For ≥ 99% of FIT images, ≥ 95% of the image's own board-mask pixels lie inside the ROI |
| Alignment | ROI is drawn over the registered FIT mean image for visual confirmation (figure saved) |

### Per-image fail-safe for unstable registration

The ROI is only valid if the image was registered correctly. For every image scored by v2 (FIT, CALIB,
HOLDOUT, TEST, perturbed copies, single-image prediction), the code computes the image's own board mask
(same rule as step 1) and its **coverage** = fraction of those mask pixels inside the ROI.

- Coverage ≥ 0.95: score = maximum of the anomaly map inside the ROI.
- Coverage < 0.95: the ROI is not trusted. The score falls back to the v1 full-frame maximum, and the
  image is marked `roi_fallback`. The full-frame maximum is never lower than the ROI maximum, so a
  registration failure can only make REVIEW *more* likely (fail-safe).
- No board mask found: the run stops with an error, as in v1.

The same rule applies to every split and to calibration, so calibration and test scores are on the same
scale. Fallback counts are reported for every split.

## 5. Dataset and split

- **Prospective validation category: VisA `pcb2`**, chosen before any PCB2 data was examined, because it
  is the next PCB category in the same benchmark (same acquisition setup, different board).
- Official manifest: `split_csv/1cls.csv`, rows with `object == pcb2`. Expected: 901 train normals,
  100 test normals, 100 test anomalies. Counts are verified; any mismatch stops the run.
- The 901 official-train normals are split deterministically (seed `20261008`, sorted paths, the v1
  `make_split` code): **600 FIT / 150 CALIB / 151 HOLDOUT**.
- The split is audited automatically: unique paths, roles consistent with the official split, no
  identical file contents across roles (SHA-256), exact counts.
- The official test split (100 + 100) is used unchanged. It is used once, for the confirmatory run.
- PCB3 and PCB4 are not used.

## 6. Procedure and freeze

1. Commit this protocol.
2. Implement and unit-test the ROI change.
3. Extract PCB2, build the manifest and split, audit them.
4. Inspect about 20 FIT normals (contact sheet, registration statistics, mean/std images). Confirm or
   reject the registration, crop and board-mask assumptions. Record ROI quality. (§9 governs any
   adaptation.)
5. Fit v1 and v2 on the same 600 FIT images. Calibrate each on the same 150 CALIB images; each gets
   its own P90/P95/P99 thresholds from its own calibration scores.
6. Freeze: each run's `frozen_operating_point.json` stores hashes of its config, fitted model (which
   includes the ROI), split manifest and pretrained weights.
7. **Commit the frozen state** (both configs, split manifest, both freeze files, calibration scores,
   evaluation code). `ivad evaluate` for PCB2 refuses to run unless the config, split manifest and
   freeze file are tracked by git and unchanged from the committed version.
8. Run the single confirmatory evaluation: official test (100 + 100) and holdout (151), both models.
9. Then robustness, comparison, diagnostics, reports.

## 7. Comparison methodology

Both models are evaluated on the **same 200 official test images** and the same 151 holdout images.
Each is evaluated at its own frozen primary threshold (P95), because that is how each would be
operated.

| Quantity | Method |
|---|---|
| AUROC, each model | Mann–Whitney; stratified percentile bootstrap, B = 2000, seed 7 |
| Recall, FPR, TP/FN/FP/TN, each model | counts at the frozen P95 threshold; Clopper–Pearson exact 95% CI |
| Δ Recall (v2 − v1), 100 anomalies | paired: exact two-sided McNemar test on discordant images; 95% CI by paired bootstrap over images (B = 10000, seed 2026) |
| Δ FPR (v2 − v1), 100 test normals; separately 151 holdout normals | same paired methods |
| Δ AUROC (v2 − v1) | paired stratified bootstrap: each replicate resamples images once and scores both models on them (B = 10000, seed 2026) |
| Median latency, each model | batch 1, end-to-end, after warm-up, same GPU |

Holdout results are reported separately from the official test results, never pooled into them.

## 8. Success criteria and outcome classification

**Primary endpoint:** Δ Recall at each model's frozen P95 threshold on the 100 PCB2 test anomalies.

**FPR non-inferiority condition:** v2 FPR ≤ v1 FPR + 0.05 (absolute, point estimates) on the official
test normals **and** on the holdout normals.

| Outcome | Rule |
|---|---|
| **Improvement supported** | Δ Recall > 0, McNemar p < 0.05, and the FPR non-inferiority condition holds |
| **No meaningful effect demonstrated** | McNemar p ≥ 0.05 for recall, and the FPR non-inferiority condition holds |
| **Unfavourable trade-off** | FPR non-inferiority fails, or Δ Recall < 0 with McNemar p < 0.05 |

Secondary (descriptive, no decision attached): Δ AUROC with CI; Δ FPR with CI; fraction of v1 PCB2
false alarms whose peak lies outside the ROI (does the premise hold on PCB2?); ROI fallback counts.

**Absolute acceptance (original, unchanged):** each PCB2 model is also scored against
`config/acceptance_criteria.yaml` exactly as v1 was: Recall ≥ 0.90, FPR ≤ 0.10 (official test and
holdout), AUROC ≥ 0.90, median latency ≤ 1 s, robustness limits; NO-GO if recall < 0.80, FPR > 0.20 on
either normal set, or a robustness recall drop > 0.20. Targets are point estimates; CIs are reported
alongside. These criteria are not edited in v2.

A supported improvement does not imply the absolute criteria are met, and vice versa.

**Failure conditions that stop or invalidate v2:**

- ROI quality check fails on FIT images (§4).
- PCB2 registration/board-mask assumptions fail on FIT inspection and no essential adaptation (§9) is
  possible.
- CALIB fallback rate above 5% (the ROI does not describe normal boards).
- Gaussian states of v1 and v2 differ (the comparison would not isolate the ROI).
- A defect discovered after test scoring (§10).

## 9. Allowed dataset-specific adaptation

If FIT-image inspection shows that the PCB1 registration, crop or board-mask parameters cannot run on
PCB2 (e.g. the board does not fit the crop, or the HSV rule finds no board), the minimal parameter change
needed to run is allowed. It must:

- be decided from FIT images only;
- apply identically to v1 and v2;
- be written in an *Addendum* to this file and committed before fitting.

Model hyperparameters (§3) are never adapted.

## 10. Rules preventing post-test tuning

- No configuration, ROI rule, threshold or code that affects scores changes after the freeze commit.
- The PCB2 test split is scored once. Re-running `evaluate` is only allowed to show reproducibility, and
  must reproduce identical scores.
- PCB2 test results are never used to modify v2. No PCB3/PCB4 search for better results.
- If a serious implementation defect is found after test scoring, the confirmatory run is declared
  **invalidated** in the decision record. A repaired re-run on the same test images is reported as a
  non-independent, post-hoc result — never as a fresh confirmation.
- PCB1 v2 numbers may be computed **after** the PCB2 run, labelled retrospective (discovery data).
- Robustness thresholds are never re-estimated on perturbed or test images.

## 11. Robustness and cost (secondary)

Same four perturbations as v1 (brightness/contrast ×0.9 and ×1.1, Gaussian blur σ = 1.5 px, shift
(+40, +25) px), applied to the native image before registration, scored at each model's frozen P95
threshold, on test anomalies plus test and holdout normals. Comparisons are paired per original image.
The shift is applied before registration, so it tests registration + detection together, not the
scoring model alone. A compact cost-sensitivity table (break-even k range) is reported per model; it
describes, it does not choose thresholds.

---

## Addendum A1 — PCB2 FIT inspection and ROI rule amendment

**Recorded 2026-10-09 11:29 (+08:00; commit `9b69850`), after inspecting PCB2 FIT normals and before any PCB2 model was
fitted, any calibration image was scored, or any test or holdout image was loaded.** Evidence:
`reports/pcb2_inspection/` (FIT images only).

### What the inspection confirmed (no adaptation needed)

| Assumption | PCB2 FIT evidence (600 images unless stated) | Status |
|---|---|---|
| Official counts | 901 train normal / 100 test normal / 100 test anomaly; manifest read from the official archive, SHA-256 identical to the upstream file used for PCB1 | holds |
| Split | 600 / 150 / 151 / 100 / 100; audit PASS; no path or content overlap with PCB1 | holds |
| Image size and camera | all 1404×1070; same overhead camera, cloth and lighting style as PCB1 (20-image contact sheet) | holds |
| Board pose varies; registration needed | centroid x 557–760 px, y 507–560 px, tilt −2.1° to +2.5°; **no 180° rotations** | registration works unchanged: the registered mean image is sharp |
| Crop 1152×768 contains the board | board and header pins fit inside the registered crop (pin tips close to the top edge) | holds |
| Colour board mask finds the board | board area fraction 0.251–0.277; mask found on every FIT image | holds for registration |
| Lighting | mean grey level of the 20 inspected images 83–95 (spread ≈ ±7%, vs ±4% on PCB1) | the pre-registered ±10% photometric test is now ≈ 1.5× the natural spread instead of 2.5× |

Lowest orientation (0°/180°) decision margin on FIT: 0.097 (PCB1: 0.36). No FIT image was flipped.

No registration, crop, resize or model parameter is changed for PCB2. v1 and v2 use the same values as
PCB1 v1.

### Why the pre-registered ROI rule (A0) is rejected

The ROI rule in §4 builds the ROI from the colour-based board mask used for registration. On PCB2 FIT
images that mask **does not contain the metallic header pins, the black header base, or the dark U1 IC
where it meets the board edge** (`reports/pcb2_inspection/roi_rule_A0_vs_A1.png`, left). A ROI built
from it would silently discard valid board features, so a bent or missing pin could never raise the v2
score. This fails the §4 quality check "board not discarded" on visual inspection, although the numeric
coverage check passes (it measures coverage of the same flawed mask). The numeric check alone was not
sufficient; this is recorded as a design lesson.

This also affects how the v1 PCB1 diagnosis must be read: the statement "all 13 false alarms peak on
the background cloth" was computed with the same colour mask, so a peak on the header pins would have
been counted as "background". That statement will be re-checked retrospectively on PCB1 (discovery data)
after the PCB2 confirmatory run, and corrected if needed.

### Amended rule A1 (applies to v2 everywhere, including any retrospective PCB1 run)

Steps 1–2 of §4 are unchanged (colour-mask consensus at 50%). Added before the margin:

3. **Board-attached structure.** Compute the pixel-wise standard deviation of the registered FIT model
   inputs (grey level, model frame). Parts fixed to the board appear at the same place in every
   registered image and have high variance at their edges; the cloth texture moves relative to the board
   and stays low-variance. Measure the cloth level on pixels more than 48 px from the consensus (median
   and robust SD = 1.4826 × MAD). Add pixels whose std exceeds median + 10 × robust SD **and** that are
   connected to the consensus. Close gaps (disk radius 8 px; border-safe) and fill holes.
4. Dilate by 16 px (unchanged).

On PCB2 FIT data (preview): cloth std median 9.4, robust SD 0.91, threshold 18.4; consensus 44.6% of the
frame, object 53.6%, ROI 62.3% (A0 would have been 54.0%). The ROI covers the board, the pins, the
header base and U1 with the margin; about 38% of the frame (cloth) is excluded.

Parameter choice (FIT only): the threshold multiplier 10 was chosen as a conservative "clearly above the
cloth" level; 6 gave a visually identical object. The closing radius 8 px (= one feature-grid cell) and
the 48 px cloth distance (3× the margin) are geometric constants. No calibration, holdout or test image
and no defect information was used.

The quality gates of §4 are unchanged and are re-applied at fit time, and the ROI is drawn on the
registered FIT mean image (`ivad roi` → `reports/pcb2_v2/roi_qc.png`) before calibration.

Config: `config/pcb2_v2.yaml` → `scoring.roi.structure_k`, `structure_far_px`, `closing_px`.
Code: `src/ivad/roi.py` (`attached_structure`, `build_roi`), `src/ivad/pipeline.py` (`fit`).
Tests: `tests/test_roi.py` (pins attached to the board and touching the image border are kept; detached
cloth blobs are not; the A0 rule would exclude them).

v1 is unaffected: it has no ROI.
