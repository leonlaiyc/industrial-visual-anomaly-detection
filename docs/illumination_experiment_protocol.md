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

---

## Addendum B2 — independent-review follow-up: chronology disclosure and FIT-only diagnostics

**Recorded 2026-10-09 ~16:55 (+08:00), before any PCB3 CALIB, HOLDOUT or test image was scored.** Added at
the request of an independent code review of the snapshot `review/pcb3-site-20261009` (`b27539b`). It does
**not** invoke §10: nothing is adapted. It discloses one procedural deviation and records descriptive
diagnostics. (The status line at the top of this file allows addenda only for §10 reasons; B2 is added as a
disclosure outside those reasons, and that is itself noted here.) Primary endpoint, conditions (§7), statistics and decision thresholds (§8), post-test rules
(§9) and the acceptance criteria are **unchanged**.

### B2.1 Chronology of Addendum B1 (procedural deviation)

§10 requires the FIT-only adaptation to be "written in *Addendum B1* before fitting". The recorded
chronology (development-machine file modification times and git commit times; nothing backdated, no git
timestamp altered):

| Time 2026-10-09 (+08:00) | Event | Evidence |
|---|---|---|
| 14:01:57 | Protocol committed, before any PCB3 file was read | commit `d7f5e81` |
| 14:08:07 | Correction, N1, paired analysis committed (synthetic tests only) | commit `caca753` |
| 14:11:04 | FIT luminance reference values computed | `reports/pcb3_inspection/photometric_fit_stats.json` |
| 14:12:12 | Background hue band determined from FIT border pixels | `reports/pcb3_inspection/background_hue_fit.json` |
| 14:13–14:14 | FIT inspection re-run with the adapted mask | `inspection_stats.json`, figures |
| 14:19:14 | Baseline fitted (FIT only) | `artifacts/pcb3_baseline/model.pt` |
| 14:22:04 | Candidate fitted (FIT only) | `artifacts/pcb3_illum/model.pt` |
| 14:24:03 | Addendum B1 **committed**, together with the split manifest, the `exclude_hue` code and config change and the FIT-only ROI QC | commit `5ba6aca` |

**Deviation:** the adaptation was decided at 14:12, before fitting, but Addendum B1 and the code, config and
split manifest implementing it were committed about five minutes **after** both variants were fitted. The
required order (addendum committed → fit) was not followed. This does not by itself establish CALIB, HOLDOUT
or test leakage: no such image had been scored at that time (none has been scored at the time of writing).
Corroborating checks made now: (a) both `model.pt` files record config hashes equal to the committed
`config/pcb3_baseline.yaml` / `config/pcb3_illum.yaml` (`9197db33…`, `f224e250…`); (b) re-running the
committed mask, registration and ROI code on the 600 FIT images reproduces both stored ROIs pixel for pixel
(0 differing pixels, `fit_diagnostics_B2.json` → `roi_specificity`). These show consistency between the
committed state and the fitted models; they cannot prove the order of events, which is stated above as it
happened.

### B2.2 Why the diagnostics were added

The reviewer asked (1) whether the expected G1 effect could be a registration/segmentation effect rather
than a feature-representation effect, (2) whether the PCB3 ROI (≈ 48% of the model frame, vs a native-frame
board mask of 13–17% — different coordinate systems) really excludes background, and (3) whether a
correction estimated from the whole frame is defensible when the background dominates the frame.

**Executed:** 2026-10-09 16:40–16:50 (+08:00), `python -m ivad fit-diagnostics` (`src/ivad/fitdiag.py`).
**Data:** the 600 FIT normals only (role `fit`, the images both models were fitted on), conditions P0, G1,
B1, N1 exactly as defined in §7 (no others), each variant with its own fitted template and ROI. No CALIB,
HOLDOUT or test image and no ground-truth mask was read. **Outputs:**
`reports/pcb3_inspection/fit_diagnostics_B2.json` (summary), `fit_diagnostics_B2_per_image.csv` (4 800
rows), `fit_diag_B2_roi.png`, `fit_diag_B2_photometric.png`.

**Heuristic, not ground truth:** the colour board mask, the "cloth" colour class (S > 90 and hue in the B1
band), the board/background luminance regions, the FIT template, and each image's own P0 registration (used
as the reference for its perturbed copies). The measurements describe **input-processing sensitivity**; they
are not a causal decomposition of score changes and are not model-selection criteria.

### B2.3 Registration and segmentation under the conditions (600 FIT images per cell)

| | Baseline G1 | Candidate G1 | Baseline B1 | Candidate B1 | Baseline N1 | Candidate N1 |
|---|---|---|---|---|---|---|
| Native mask IoU vs own P0, median [min] | 0.90 [0.40] | 1.00 [0.94] | 0.93 [0.81] | 0.99 [0.87] | 0.95 [0.57] | 0.99 [0.88] |
| Registration change vs own P0 (RMS px over the crop), median / P95 / max | 10.7 / 33.6 / 237 | 0.2 / 1.4 / 19 | 12.3 / 30.0 / 37 | 1.9 / 12.0 / 35 | 9.3 / 26.7 / 171 | 3.8 / 16.9 / 35 |
| Images with change > 20 px / > 50 px | 106 / 11 | 0 / 0 | 184 / 0 | 11 / 0 | 66 / 1 | 23 / 0 |
| ROI fail-safe events (coverage < 0.95) | **8** | 0 | 0 | 0 | 1 | 0 |
| Invalid / empty masks; 0°/180° decision changes | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |

For scale: on clean images (P0) the residual translation of a registered image against the FIT template
is median 5.7 px, P95 15.8 px (both variants) — the natural registration residual.

**Reading.** The colour board mask is brightness-dependent (V > 170 and S > 90 are absolute thresholds), so
without correction a 10% darkening moves the estimated pose by about 11 px (median) — more than the natural
residual — and grossly (> 50 px) on 11 of 600 FIT images, with 8 ROI fail-safe events. The candidate's
correction runs before the mask, so under G1 its segmentation and registration are essentially those of the
clean image (600/600 images have a higher IoU than the baseline's). **Implication for interpreting the
planned result:** a lower G1 false-alarm rise for the candidate may come partly or wholly from more stable
registration and ROI coverage, not only from features closer to the normal model. The experiment was not
designed to separate these, and the report will not claim such a decomposition. B1 and N1 also perturb the
baseline's registration (median 12 px and 9 px); the candidate reduces but does not remove this for N1.

### B2.4 Does the ROI exclude background? (same registered model frame, 768×512)

| Quantity (baseline; candidate within 0.5 pts) | Value |
|---|---|
| Fixed ROI area | 48.0% of the frame (candidate 47.8%) |
| Consensus board core (≥ 50% of FIT board masks) | 25.6% of the frame |
| Board-mask coverage by the ROI | 1.000 on all 600 images |
| ROI pixels outside the image's own board mask, median [P5, P95] | 47% [45%, 52%] |
| ROI pixels of cloth colour, median [min, max] | 31% [28%, 35%] |
| ROI construction stages: share of ROI (cloth-colour share within the stage) | consensus core 53% (2%) · attached structure 29% (45%) · closing 2% (80%) · 16 px margin 16% (≈ 100%) |

**Reading.** The ROI does exclude the background far from the board: 52% of the model frame lies outside
it. But it is **not** a tight board mask: about 31% of ROI pixels are cloth, i.e. ≈ 15% of the frame. The
cloth inside the ROI comes from the pre-registered 16 px margin (by design) and from the Addendum A1
"attached structure" step, whose std-based rule also picks up a band of cloth around the board where the
pixel std is raised by the natural registration residual, the board's shadow and the bendable LEDs;
connected-component growth plus closing then adds it to the object. Board-attached parts that the colour
mask misses (pin header, LED bodies) are correctly inside the ROI (`fit_diag_B2_roi.png`). 100% coverage
shows that the ROI contains the board; it does **not** show that the ROI is specific to the board.

**Assessment:** a material **specificity limitation**, not an experimental-integrity failure. The ROI rule
and parameters are the pre-registered ones, applied identically to both variants, and the two ROIs are
nearly identical (48.0% vs 47.8%), so the paired comparison is not biased by it. Consequence for
interpretation: lint or texture on the cloth within roughly 15–40 px of the board can still set an image's
score in both variants. The ROI construction rule is **not** changed.

### B2.5 Whole-frame photometric correction on a background-dominated frame

| Quantity (600 FIT, raw P0 unless stated) | Value |
|---|---|
| Board share of the native frame (colour mask) | 14.8% (mean) |
| Correlation of the whole-frame mean with: background mean / board mean | 0.97 / 0.62 |
| Luma mean, median [min, max]: whole frame / board / background | 92.0 [82.7, 102.1] / 125 [107, 140] / 87 [79, 97] |
| Board-mean spread across images (SD): raw → after correction | 4.4 → 3.2 grey levels |
| Candidate gain, median [min, max]: P0 / G1 / B1 / N1 | 1.00 [0.95, 1.07] / 1.24 [1.18, 1.32] / 0.87 [0.84, 0.92] / 1.08 [1.03, 1.17] |
| Board mean after the condition minus the same image's P0, median: baseline → candidate | G1 −16.4 → −0.07 · B1 +11.9 → −0.7 · N1 −12.9 → −1.7 |
| RGB values clipped (0 or 255) in the board region, median [max]: raw P0 / corrected P0 | 2.3% [4.1%] / 2.1% [8.0%] |
| Baseline B1 board clipping (highlight detail the correction cannot restore), median | 17.3% |
| Board mean saturation / local contrast (Sobel), P0 median: raw → corrected | 110 → 110 / 105 → 105 |

**Reading.** The whole-frame statistics are dominated by the cloth (r = 0.97), so the correction
standardises mainly the **background** brightness. On these FIT images board and background brightness
co-vary, so the correction still reduces the board's brightness spread (SD 4.4 → 3.2) and restores the
board mean under G1 to within half a grey level. Recorded caveats: (1) on single images the correction can
increase highlight clipping (board up to 8% vs 4% raw); (2) wherever the background changes independently of
the board (a different cloth, a fixture, a shadow), the correction will mis-scale the board — a cross-site
risk not tested here; (3) a common RGB gain keeps channel ratios about the mean, but with the offset and
clipping it does not guarantee that colour-type defects are preserved, and that is not claimed.
**Assessment:** the pre-registered correction remains defensible for this experiment and is run unchanged.
No board-only normalisation or gain cap is introduced.

### B2.6 What changed in the repository with this addendum

- **No score-affecting code or configuration changed.** `photometric.py`, `align.py`, `roi.py`,
  `robustness.py`, `pipeline.py`, `features.py`, `gaussian.py`, `calibration.py` and both PCB3 configs are
  byte-identical to `5ba6aca`; both fitted models are unchanged.
- New: `src/ivad/fitdiag.py` and the `ivad fit-diagnostics` command (read-only, FIT images only).
- `src/ivad/illum.py` (the post-freeze paired analysis): input validation only. It now refuses missing or
  incomplete score artifacts, non-finite scores, score rows not in split-manifest order, a freeze that does
  not match the current config or split manifest, metrics produced under another freeze, inconsistent ROI
  metadata, robustness rows that are not test + holdout with the evaluated clean scores, missing predeclared
  conditions, and invalid ground-truth mask metadata. For valid inputs every statistic is computed exactly
  as before.
- New end-to-end test `tests/test_illum_e2e.py`, on synthetic artifacts only (no PCB3 output used).

### B2.7 Status

No stopping condition of §10 is met (0 invalid masks with the correction on FIT; ROI QC unchanged). Next:
calibrate both variants on the 150 CALIB normals, then the freeze commit (§6).
