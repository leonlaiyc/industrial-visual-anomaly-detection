# Illumination-correction experiment protocol (pre-registration)

**Status:** pre-registered. Committed before any VisA PCB3 file was extracted, read, fitted, calibrated
or scored. Sections marked *Addendum* may only be added before the PCB3 freeze commit, and only for the
reasons allowed in §10. Nothing in this file may change after the freeze commit.

**Scope:** one controlled experiment. One correction, one board type, one planned paired evaluation.
It is not a v3 optimisation campaign. The frozen PCB1 and PCB2 results are not changed, re-scored or
reinterpreted by it.

## 1. Observed failure in prior work

On PCB2 (`docs/v2_decision_record.md` §5, `reports/pcb2_v2/robustness.json`), a 10% darker,
lower-contrast copy of each image (PIL brightness ×0.90 then contrast ×0.90, applied to the native image
before registration) raised false alarms at the frozen P95 threshold on the 251 unseen normal images:

| Model | FPR clean | FPR darker | Δ FPR |
|---|---|---|---|
| PCB2 v1 | 8.4% | 30.7% | +22.3 pts |
| PCB2 v2 | 3.6% | 28.7% | +25.1 pts |

Both exceed the proposed robustness limit (FPR rise ≤ 10 pts, `config/acceptance_criteria.yaml`). The
brighter copy (×1.10 / ×1.10) did not do this (+0.0 / +1.6 pts). Natural mean brightness of the PCB2 FIT
normals already spans about 83–95 grey levels (≈ ±7%), wider than on PCB1 (≈ ±4%).

## 2. Hypothesis

> The darkening-induced false-alarm rise is caused mainly by a **global** shift of image intensity that
> moves every patch feature away from the fitted normal distribution. Standardising each image's global
> luminance statistics to the level of the training normals, before registration and feature extraction,
> will reduce the false-alarm rise caused by global darkening **without** reducing defect recall under
> normal lighting.

Secondary expectation, stated in advance: a global correction cannot undo a **spatially non-uniform**
illumination change, so it should help less, or not at all, under the non-uniform condition (§7, N1).

## 3. The single correction (candidate)

**Global luminance affine standardisation** (`preprocess.photometric.method: global_luma_affine`).

For each native RGB image *I* (uint8):

1. Luminance *Y* = PIL `convert("L")` (ITU-R 601: 0.299 R + 0.587 G + 0.114 B) over the whole native
   frame.
2. μ = mean(*Y*), σ = std(*Y*).
3. *I'* = clip(round((*I* − μ) · σ_ref / σ + μ_ref), 0, 255), the **same** gain and offset applied to
   R, G and B.
4. *I'* then goes through the unchanged v2 pipeline (registration → crop → resize → features → ROI max).

μ_ref and σ_ref are **the medians of the per-image μ and σ over the 600 PCB3 FIT normals** (raw, before
any correction). They are computed once, recorded in an addendum, written into the candidate config
(so they are covered by the config hash) and frozen. There is nothing else to tune.

### Why this correction

- **Matches the observed failure.** The failure appeared under a global brightness/contrast change and not
  under blur or shift. A global intensity standardisation is the smallest change that targets it.
- **No free parameter chosen from outcomes.** The two reference values are descriptive statistics of
  FIT normals. No PCB1/PCB2 test result and no PCB3 calibration, holdout or test image is used to set
  them. There is no search over methods, strengths or regions.
- **Chroma preserved.** One gain for all three channels keeps the ratio between channels' deviations from
  the mean; per-channel ("grey-world") balancing was rejected because it would partly erase
  colour-type defects (e.g. discolouration, melt).
- **Global statistics barely see a defect.** A defect covers well under 1% of the ~1.5 million pixels of
  the frame, so it moves μ and σ very little; the correction cannot "normalise away" a local defect.
- **Before registration.** The registration board mask uses fixed HSV thresholds (`align.py`), which
  darkening can also disturb. Correcting first keeps registration, the board ROI and the features on the
  same footing as in training.
- **Cheap and auditable.** One pass over the pixels; no learned component.

### Known limitations and risks (stated before testing)

1. **Global darkening is inside the correction's model class.** The ×0.90/×0.90 perturbation is itself
   a global affine intensity change with a common gain, so the correction can invert it up to rounding
   by construction. A favourable result on condition G1 therefore shows that the mechanism works and that
   the correction does not damage the model. **It is not evidence of robustness to real lighting
   changes**, which are typically spatially non-uniform and not spectrally flat. N1 (§7) is the
   out-of-class test, and its result must be reported with equal prominence.
2. **Clipping.** Brightening saturates highlights; saturated detail cannot be recovered (condition B1).
3. **Brightness-type defects.** A defect that changes the brightness of a large part of the board would be
   partly normalised away. VisA PCB defects are local, so this risk is expected to be small, but it is
   not zero and is checked through the low-contrast subgroup (§8 C).
4. **Changed normal model.** Removing natural brightness variation from the FIT images tightens the
   fitted normal distribution. This can raise or lower clean recall and clean false alarms; the clean
   gates (§8) exist to catch a loss.
5. **Noise amplification.** Stretching a darker, lower-contrast image amplifies sensor noise and JPEG
   artefacts by the same gain.
6. **Synthetic perturbations are not a manufacturing domain shift.** No claim about real lamps,
   cameras, lines or sites follows from this experiment.

## 4. Dataset and split

- **Board type:** VisA `pcb3`. Never used in this project before this protocol (no file extracted,
  read or scored). Chosen because it is the next unused PCB category of the same benchmark and the
  experiment needs data that did not motivate the correction.
- **Official manifest:** `split_csv/1cls.csv` rows with `object == pcb3`, read from the official archive
  already used for PCB1/PCB2 (SHA-256 recorded). **Expected:** 905 train normals, 101 test normals,
  100 test anomalies. The counts are verified before any image is opened; a mismatch stops the
  experiment until this protocol is amended (§10).
- **Split** (existing `make_split`, seed `20261008`, path-sorted): **600 FIT / 150 CALIB / 155 HOLDOUT**
  from the 905 train normals; the official test split (101 + 100) unchanged. Automatic audit as for
  PCB2 (unique paths, roles consistent with the official split, no identical file content across roles,
  exact counts, no overlap with PCB1/PCB2 files).
- **Same image identities for both variants** in every role and every condition.
- **PCB4 is not used**, before or after this experiment, to obtain a different answer.

## 5. Variants and invariants

| | Baseline | Candidate |
|---|---|---|
| Config | `config/pcb3_baseline.yaml` | `config/pcb3_illum.yaml` |
| Photometric step | none | §3, applied to every image of every split and condition |
| Everything else | v2 pipeline as frozen for PCB2 | identical to baseline |

Identical in both, and identical to `config/pcb2_v2.yaml`: Wide-ResNet-50-2 (IMAGENET1K_V1), layer2 +
layer3, 128 random channels (seed 0), 3×3 local average, per-position Gaussian with ridge 0.01 · mean
diagonal + 1e-6, Cholesky whitening, map smoothing σ = 4 px; registration method; crop 1152×768 →
768×512; board ROI rule A1 with the same parameters (vote 0.5, margin 16 px, structure k = 10, far 48 px,
closing 8 px, coverage fail-safe 0.95, the same QC gates); split-conformal calibration with candidates
P90/P95/P99 and **P95 primary**; seeds and bootstrap settings; acceptance criteria file.

Each variant is fitted on **its own preprocessing of the same 600 FIT images** (its own registration
template, ROI and Gaussian state) and calibrated on **its own preprocessing of the same 150 CALIB
images**. Each therefore has its own frozen threshold, as it would be operated.

## 6. Procedure and freeze

1. Commit this protocol (before any PCB3 file is extracted).
2. Implement and unit-test the correction, the new perturbation (N1) and the paired analysis on
   synthetic data only.
3. `prepare` and `split` PCB3; verify counts; audit.
4. Inspect FIT normals only (contact sheet, registration statistics, mean/std images, luminance
   statistics). Confirm or reject the registration, crop, ROI and lighting assumptions. Record μ_ref,
   σ_ref and the inspection in *Addendum B1*.
5. Fit both variants (FIT only); ROI QC; calibrate both (CALIB only).
6. **Freeze and commit**: both configs (including μ_ref, σ_ref and the perturbation definitions), the
   split manifest and audit, both `frozen_operating_point.json`, calibration scores, all code in `src/`,
   and this protocol. `evaluate` refuses to run unless that state is committed and unchanged
   (`evaluation.require_committed_freeze`). The freeze commit hash is recorded in the results.
7. **One** planned evaluation: `evaluate` (clean test + holdout) and `robustness` (all §7 conditions) for
   both variants, then the paired illumination analysis. Re-running is allowed only to show that it
   reproduces identical scores.
8. Report, whatever the outcome.

## 7. Conditions (fixed now)

Applied to the **native** image, before the candidate's correction and before registration, i.e. as a
drifted camera frame. Matched: the same source images receive the same perturbation for both variants.

| ID | Condition | Definition | Role |
|---|---|---|---|
| P0 | Clean | unchanged | clean gates; reference for every Δ |
| **G1** | Global darkening | PIL brightness ×0.90, then contrast ×0.90 (identical to the earlier `darker_lower_contrast`) | **primary robustness endpoint** |
| N1 | Non-uniform illumination | multiplicative horizontal ramp, gain 1.00 at the left edge → 0.80 at the right edge of the native frame (mean gain 0.90, i.e. the same average darkening as G1's brightness step), same gain on R, G, B, rounded and clipped to uint8 | secondary: out-of-class limitation (question D) |
| B1 | Global brightening | brightness ×1.10, then contrast ×1.10 (earlier `brighter_higher_contrast`) | secondary: opposite direction, with clipping |
| — | Blur σ = 1.5 px; shift (+40, +25) px | the earlier P5 definitions, unchanged | acceptance only (original P5); no comparative endpoint |

N1 is predeclared because a global correction is expected to fail on it; B1 because the correction
must not help one direction by harming the other. Blur and shift are run only so that the original
robustness criterion P5 can be scored as originally defined.

Repeated versions of an image are **not** independent observations: every interval and test below
resamples or pairs **original image identities**.

## 8. Endpoints and statistics

Images: 100 test anomalies; normals = 101 official-test normals + 155 holdout normals (256, pooled for
the robustness endpoints, as in the earlier robustness study; official-test and holdout FPRs are also
reported separately). Every decision uses each variant's own frozen P95 threshold.

### Primary robustness endpoint (question A)

Δ_v = FPR_v(G1) − FPR_v(P0) on the 256 normals, for v ∈ {baseline, candidate}.

**DiD = Δ_candidate − Δ_baseline.** 95% CI by paired percentile bootstrap over normal image identities
(each replicate resamples the 256 images once and recomputes all four flag rates; B = 10000,
seed 2026). Supporting test (reported, not decisive): exact McNemar on the G1 flags of candidate vs
baseline over the same normals.

### Clean-condition gates (question B) — acceptable degradation, fixed now

| Gate | Rule | Fails when |
|---|---|---|
| Clean recall | Δ recall (candidate − baseline) on the 100 clean test anomalies, exact McNemar + paired bootstrap CI | point estimate < **−5 pts**, **or** Δ < 0 with McNemar p < 0.05 |
| Clean FPR | candidate FPR ≤ baseline FPR + **5 pts** (point estimates) on the official-test normals **and** on the holdout normals | either inequality fails |

### Outcome rule (computed by code)

| Outcome | Rule |
|---|---|
| **ROBUSTNESS IMPROVEMENT SUPPORTED** | DiD 95% CI upper bound < 0 **and** both clean gates pass |
| **UNFAVOURABLE TRADE-OFF** | either clean gate fails (whatever the robustness result), **or** DiD 95% CI lower bound > 0 |
| **INCONCLUSIVE** | anything else |

If the baseline's own G1 rise Δ_baseline is below 5 pts, PCB3 did not reproduce the PCB2 failure; the
report must say that the primary endpoint is weakly informative on this board type, whatever the rule
returns.

A positive outcome does **not** mean the original requirements are met (below), and is bounded by the
limitation in §3 risk 1.

### Small and low-contrast anomalies (question C) — descriptive, fixed now

On the clean test anomalies, using the official ground-truth masks (description only, never for any
setting):

- **Size:** mask area fraction; terciles (small / medium / large) over the 100 anomalies.
- **Contrast:** |mean grey inside the mask − mean grey in a ring 1–12 px outside the mask| on the clean
  native image; terciles (low / medium / high).

Recall per tercile for both variants, with discordant counts. **Sensitivity flag:** if the candidate
detects ≥ 3 fewer anomalies than the baseline in the small or the low-contrast tercile, the report must
state a sensitivity risk even if the clean recall gate passes. Recall under G1 per tercile is reported
alongside.

### Non-uniform illumination (question D)

Same quantities for N1 (Δ_v, DiD with CI). **A limitation is exposed** if the candidate's N1 FPR rise
exceeds 10 pts or its N1 recall drop exceeds 10 pts (the original P5 limits), whatever DiD says.

### Runtime and operations (question E)

Median and P95 end-to-end latency (batch 1, after warm-up, same GPU, same 201 test images) for both
variants, by the existing `evaluate` measurement; plus the correction step alone (CPU, per image).
Operational additions (stored reference values, a new failure mode) are listed in the report.

### Absolute acceptance (original, unchanged)

Both variants are scored against `config/acceptance_criteria.yaml` exactly as before: recall ≥ 90%,
FPR ≤ 10% (official test and holdout), AUROC ≥ 0.90, median latency ≤ 1 s, P5 robustness over the
four original perturbations (G1, B1, blur, shift — N1 is excluded from P5 because it was not part of
the original definition), NO-GO rules unchanged. **Nothing is lowered.** Clean AUROC and its paired
Δ (stratified paired bootstrap, B = 10000, seed 2026) are reported as secondary.

## 9. Post-test rules

- No configuration, correction parameter, threshold, perturbation definition or score-affecting code
  changes after the freeze commit.
- The PCB3 test and holdout images are scored once per condition. Results are never used to modify the
  correction, choose another correction, or re-calibrate.
- If the correction fails, the evidence is kept, the failed assumption is named, and a next
  investigation may be **recommended but not implemented** in this project.
- No PCB4, and no re-run on PCB1/PCB2 test data, to obtain a different answer. PCB1/PCB2 results are
  not touched.
- If a serious implementation defect is found after test scoring, the run is declared **invalidated**;
  a repaired re-run on the same images is reported as non-independent and post hoc.
- Post-hoc diagnostics (per-defect-type recall, heat-map peaks) may be computed after the run, labelled
  post hoc, and may not change any decision above.

## 10. Allowed pre-freeze adaptation

If FIT inspection shows that the PCB2 registration, crop or board-mask parameters cannot run on PCB3,
the minimal parameter change needed to run is allowed. It must be decided from FIT images only, apply
identically to both variants, and be written in *Addendum B1* before fitting. Model hyperparameters,
the correction rule (§3) and the endpoints (§8) are never adapted. If the counts in §4 differ from the
expectation, the split sizes are re-derived by the same rule (600 / 150 / remainder) and recorded in the
addendum.

**Stopping conditions (the experiment is reported as failed to run, not as a result):** ROI QC fails for
either variant; more than 5% of CALIB normals hit the ROI fail-safe for either variant; the correction
breaks registration on FIT images (no board mask found).

---

## Addendum B1 — PCB3 FIT inspection, one essential adaptation, reference values

**Recorded 2026-10-09 14:25 (+08:00), after inspecting PCB3 FIT normals and fitting both variants on
FIT images, before any CALIB, HOLDOUT or test image was scored.** Evidence: `reports/pcb3_inspection/`,
`reports/pcb3_baseline/roi_qc.*`, `reports/pcb3_illum/roi_qc.*` (FIT images only).

### Data checks

| Item | PCB3 evidence | Status |
|---|---|---|
| Official counts | 905 train normal / 101 test normal / 100 test anomaly; manifest read from the official archive, upstream SHA-256 identical to the file used for PCB1/PCB2 | as expected |
| Split | 600 FIT / 150 CALIB / 155 HOLDOUT / 101 + 100 test; audit PASS; no identical file content shared with PCB1 or PCB2 | holds |
| Image size | all 1562×960 (PCB2: 1404×1070); the 1152×768 crop contains the board, pins and LEDs after registration | holds |
| Orientation | no 180° rotations among 600 FIT images; lowest 0°/180° template margin 0.32 | holds |

### Essential adaptation (§10): background-hue exclusion in the board mask

**Finding.** The PCB3 background is a **saturated green cloth**, not the grey cloth of PCB1/PCB2. The
original colour rule (saturation > 90 **or** value > 170) therefore marked about **96% of every frame as
board**: on all 600 FIT images the "board" centroid was the image centre and the tilt 0°, i.e.
registration silently did nothing. The pipeline cannot run as intended on PCB3 without a change.

**Change (minimal, both variants, FIT only).** Saturated pixels whose hue lies in the background hue band
are no longer counted as board: foreground = (S > 90 **and** hue ∉ [84, 106]) **or** V > 170 (PIL hue
units 0–255; ≈ 118°–150°). Band = 0.5th–99.5th percentile of the hue of saturated pixels in the 8-pixel
border of all 600 FIT images (6.0 million pixels, median 95, P0.5 92, P99.5 98), widened by 8 units on
each side (`reports/pcb3_inspection/background_hue_fit.json`). The board's blue is near hue 150.

- Config: `preprocess.board_mask.exclude_hue: [84, 106]`, identical in `pcb3_baseline.yaml` and
  `pcb3_illum.yaml`. Code: `align.foreground_mask(exclude_hue=…)`, passed through registration, the ROI
  board mask and inspection. Without the key the rule is unchanged: PCB1/PCB2 configs and hashes are
  untouched, `ivad check` passes, and three PCB2 v2 test images score identically before and after the
  code change.
- Hue is invariant to a global gain, so the band does not by itself favour either variant under the
  darkening conditions.

**After the adaptation (600 FIT):** board area 12.9–16.7% of the frame; centroid x 726–914 px, y 449–532
px; tilt −5.7° to +1.1°; registered mean image sharp (`registered_mean_std_fit_normals.png`).

**Normal variation observed (recorded, not acted on):** the two IR LEDs stand on bendable leads; the
registered std image shows they move relative to the board from unit to unit. This is legitimate normal
variation inside the ROI and a plausible source of false alarms for both variants.

### Lighting on FIT normals

Per-image luminance over the whole frame (600 FIT): mean 82.7–102.1 (P5–P95 89.3–95.0, median 92.00);
std 31.0–34.8 (median 33.12). For orientation only: G1 maps a mean of ≈ 92 to ≈ 83, the bottom of the
natural FIT range. This is descriptive and changes nothing in §7–§8.

**Frozen reference values (§3):** μ_ref = **91.9984**, σ_ref = **33.1160** (medians over the 600 FIT
normals, raw images; `reports/pcb3_inspection/photometric_fit_stats.json`), written into
`config/pcb3_illum.yaml`.

### ROI (rule A1, unchanged parameters)

| | Baseline | Candidate |
|---|---|---|
| ROI area | 48.0% of the model frame | 47.8% |
| FIT images with ≥ 95% board-mask coverage | 600/600 | 600/600 |
| Visual check (`roi_qc.png`) | board, header pins and both LEDs inside, with margin | same |

Both pass the §4 (v2) quality gates. No CALIB, HOLDOUT or test image has been scored. Next step:
calibration of both variants on the 150 CALIB normals, then the freeze commit (§6).
