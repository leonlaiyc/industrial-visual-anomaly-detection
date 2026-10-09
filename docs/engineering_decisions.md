# Engineering decisions log

Records what was decided, why, what was rejected, which code-level risks were found and how they were
fixed, and the evidence. Most of the code was written with AI coding assistance (Claude Code); this log
exists so that a human owner can audit the reasoning without re-deriving it.

Measured numbers live in `reports/results.md` (generated). This log quotes only numbers needed to
explain a decision, with their source.

---

## Protocol timeline (verifiable in `git log`)

| Step | Evidence |
|---|---|
| Environment inspected, project created | 2026-10-08 23:39 (Taiwan) — session log |
| Official manifest confirmed: PCB1 = 904 train-normal / 100 test-normal / 100 test-anomaly | `manifests/visa_1cls_pcb1.provenance.json` |
| Split frozen and audited | `manifests/split_pcb1.csv`, `manifests/split_audit.json` |
| Inspection of **fit-role** normals only → registration + resolution decided | `reports/inspection/`, `docs/dataset_inspection.md` |
| Model fitted; tests passing; config with pre-registered thresholds committed | commit `1c5b121` |
| Thresholds frozen from calibration normals (before any test scoring) | commit `fdd1f35` |
| Line-ending fix; re-freeze reproduced identical thresholds | commit `f643526` |
| Official test + holdout evaluated (first time any of these images was scored) | `reports/metrics.json` → `evaluated_at_utc` |
| **v2:** protocol pre-registered before any PCB2 data access | commit `ad8a386` |
| v2 code (ROI, paired comparison, PCB2 configs) | commit `fd1caf8` |
| PCB2 split + FIT inspection; ROI rule amendment A1 (before any PCB2 model) | commit `9b69850` |
| PCB2 v1 and v2 fitted, calibrated, frozen | commit `862c28f` |
| PCB2 single confirmatory evaluation (evaluated at HEAD `862c28f`) | commit `2b775be` |
| PCB1 v2 retrospective (post hoc, discovery data) | commits after `736dcc7` |

---

## Architecture decisions

### D1. One dataset, official manifest, extra normal-only split
- VisA `pcb1` only, per scope. The official 1-class split is used unchanged for testing.
- The 904 official-train normals are split 600 fit / 150 calib / 154 holdout with seed `20261008`.
  Paths are sorted before permuting, so the split does not depend on file-system or CSV order.
- **Rejected:** fitting on all 904 (the usual benchmark protocol). That leaves no independent normals
  to set an operating threshold without touching test labels. The price is a smaller fit set, so
  results are not directly comparable to published PCB1 numbers (see evaluation methodology).

### D2. Board registration before position-specific modelling
- **Finding (fit normals only):** the camera may be fixed, but the board is not. 116/600 fit
  normals are rotated 180°, centroids range ≈ ±70 px, and tilt is about −2.7°…+2.3°
  (`reports/inspection/inspection_stats.json`). Assumption F1 (fixed pose) is false for this data.
- **Decision:** register every image identically, whatever its split (`align.py`).
  1. Find the board mask from HSV (saturated or bright pixels), clean it morphologically, keep the
     largest component.
  2. Take the centroid and principal-axis angle from image moments.
  3. Rotate about the centroid and translate it to the image centre.
  4. Choose 0° or 180° by normalised correlation with a template, the mean of registered fit normals.
- **Evidence:** the registered mean image is sharp (silkscreen legible) where the raw mean was ghosted
  (`reports/inspection/registered_mean_std_fit_normals.png`). The orientation decision margin is
  ≥ 0.33 on all fit/calib/holdout images.
- **Rejected alternatives:**
  - *No registration (plain PaDiM-style).* Each position's Gaussian would mix two orientations and
    ±70 px offsets, wasting the position-specific model. It would also expose the model to the
    orientation confound found later (D12).
  - *Position-invariant model (global pooling, or a nearest-neighbour memory bank over all positions).*
    Global average pooling would dilute small local defects, as the brief warned. A memory bank is a
    different method family from the requested statistical model.
  - *Rotation augmentation.* It widens every per-position distribution and so lowers sensitivity.
- **Residual risk:** a defect that changes the board outline (e.g. a large missing part) could shift the
  pose estimate. Test image `Anomaly/049` has the lowest orientation margin (0.099). It was still
  flagged, but its registration is less certain (`reports/diagnostics.json`).

### D3. Input resolution: crop 1152×768 of the registered frame → 768×512
- The usual 256×256 resize is a ≈5.5× downsample of the 1404×1070 originals and visibly smears vias and
  traces (`reports/inspection/resize_loss_crops.png`). The crop keeps the board, the pins and a margin,
  and drops ≈ 2/3 of the background cloth. 768×512 is only a 1.5× downsample.
- Chosen from fit normals only, before any calibration, holdout or test image was scored.

### D4. Backbone and layers: torchvision Wide-ResNet-50-2 (IMAGENET1K_V1), layer2 + layer3
- Mid-level layers balance locality (stride 8 → 64×96 grid) against semantic robustness. This follows
  the PaDiM/PatchCore literature.
- **Rejected:** `layer1` (4× more positions, so 4× covariance memory, and mostly low-level texture
  noise); `layer4` (stride 32 is too coarse for ≈1 mm defects, and its features are the most
  ImageNet-specific). ResNet-18 is a lighter option kept for tests only, not compared, per the
  "no architecture search" scope.

### D5. Dimensionality reduction: fixed random 128 of 1536 channels, then 3×3 local mean
- Random channel selection is the PaDiM reduction; their paper reports it performs comparably to PCA.
  It needs no fitting and is reproducible from `feature_seed`.
- d = 128 keeps N/d ≈ 4.7 (600 fit images) so the per-position covariance is estimable. Memory:
  6144 positions × 128² × 4 B ≈ 0.40 GB for the whitening matrices (`artifacts/model.pt`, 406 MB).
- A 3×3 average adds tolerance to the ≤ 1–2 px residual misregistration.
- **Rejected:** PCA (an extra fitted projection, and a full 1536×1536 covariance to estimate);
  larger d (N/d → 1 gives an ill-conditioned covariance).

### D6. Covariance estimation and numerics (`gaussian.py`, implemented in-project)
- Streaming float64 accumulation with a pseudo-mean shift, which avoids the cancellation in
  E[xxᵀ] − E[x]E[x]ᵀ. Unit-tested against `np.cov` with a +10⁴ offset.
- Ridge: Σ + (0.01 · mean diag Σ + 10⁻⁶) I, scale-aware with an absolute floor. Guarantees positive
  definiteness even for singular Σ (N ≤ d or constant channels); tested.
- No explicit inverse. Store W = L⁻¹ from the Cholesky factor, then distance² = ‖W(x − μ)‖². A failed
  Cholesky raises loudly instead of silently pseudo-inverting.
- Scoring uses one `bmm` with positions as the batch dimension (see R3).
- Validated against `scipy.spatial.distance.mahalanobis` (relative error < 1e-9 in float64; < 1e-4 for
  the float32 production path).
- Not claimed: that features are Gaussian. The distance is an anomaly score; its threshold is empirical.

### D7. Image score = max of the upsampled, Gaussian-smoothed (σ = 4 px) distance map
- The maximum keeps a single small defect visible. Mean or top-k pooling would dilute it in proportion
  to the defect's area.
- Consequence (found post hoc, D12): the maximum is equally sensitive to *any* abnormal patch,
  including dust on the background. This is the main driver of false alarms.

### D8. Thresholds: split-conformal order statistics on 150 calibration normals
- Candidates P90/P95/P99 (target FPR 10/5/1%), pre-registered in the committed config before
  calibration. Flag iff score > s₍ₖ₎ with k = ⌈(n+1)(1−α)⌉.
- Under exchangeability, the realised FPR ~ Beta(n+1−k, k). For P95 that is a 95% range of
  ≈ 1.9%–8.5% from calibration sampling alone, before any drift. This is reported next to the measured
  FPR so that target and measured values are never conflated.
- **Primary = P95**, chosen *before* testing. Reason: the proposed tolerance is FPR ≤ 10%, and a
  threshold targeting exactly 10% exceeds it about half the time. P95 leaves margin.
- Note: P99 with n = 150 gives k = n, i.e. the calibration *maximum*. It is a single extreme value and
  unstable, as documented.
- **Rejected:** Hotelling T² / χ² theoretical limits. They assume Gaussian features, known parameters
  and independent patches; none holds.

### D9. Freeze guard
- `calibrate` writes `reports/frozen_operating_point.json` with SHA-256 hashes of the semantic config,
  `model.pt`, the split manifest (line-ending normalised) and the pretrained weights.
- `evaluate`, `robustness`, `report` and `predict` refuse to run if any hash differs. The freeze was
  committed to git before evaluation (commits above).

### D10. Statistics
- Recall and FPR use Clopper–Pearson exact intervals (conservative; well-defined at 0 or n).
- AUROC uses a stratified percentile bootstrap (B = 2000) that resamples anomalies and normals
  separately.
- Robustness Δs use a paired bootstrap over **original images**, stratified by class. Each perturbed
  copy always travels with its original, because they are not independent samples.

### D11. Robustness design
- Perturbations are applied to the native image *before* registration, so they mimic a drifted camera
  frame. Severities were fixed from fit-normal inspection before any test scoring:
  - Natural brightness spread is ±4%; the photometric tests use ×0.9 / ×1.1 brightness and contrast.
  - Blur is a Gaussian with σ = 1.5 native px (mild defocus).
  - The shift is (+40, +25) px, inside the natural placement range, so it tests registration.

### D12. Post-hoc diagnostics (labels used for description only)
- Results and recall by defect type and size: `reports/results.md` §6.
- Peak location: false alarms peak on the background cloth, missed anomalies mostly peak on the
  defect itself.
- **Orientation confound:** 95/100 test anomalies are rotated 180°, against 32/100 test normals and
  19% of training normals. An unregistered position-specific model could therefore score orientation
  rather than defects.
- Checked post hoc on normals: rotated and unrotated normals have indistinguishable score
  distributions (Mann–Whitney p ≈ 0.89; orientation-vs-score AUROC ≈ 0.505). AUROC within the
  rotated-only test subset is 0.910 [0.837, 0.970], consistent with the headline.

---

### D13. v2: board-ROI score aggregation (the single v2 change)
- Image score = max of the anomaly map inside a fixed board ROI instead of the whole frame
  (`roi.py`). Rationale and evidence: `docs/v2_decision_record.md`; protocol:
  `docs/v2_experiment_protocol.md`.
- The ROI is one mask per model, built at fit time from FIT normals only and stored in `model.pt`, so
  the freeze hash covers it.
- **Rejected:** a per-image ROI from each image's own board mask. A defect that changes the mask (a
  missing part) would shrink its own ROI and hide itself.
- Per-image fail-safe: if < 95% of an image's own board mask lies inside the ROI, the score falls back
  to the v1 full-frame max, which is never lower. Fallbacks are counted per split (all zero on PCB2).

### D14. v2 ROI rule amendment A1 (before any PCB2 model)
- The pre-registered rule (consensus of colour board masks + 16 px) excluded the header pins, the header
  base and part of an IC on PCB2 FIT images. A numeric coverage check against the same mask could not
  see this; the visual check did.
- A1 adds board-attached structure: pixels whose grey-level std across registered FIT images exceeds
  the cloth level (median + 10 robust SD) and that connect to the board.
- **Rejected:** a convex hull of board + pins (adds large cloth regions beside the pins).
- Lesson: a QC metric measured against the same mask it checks cannot detect that mask's blind spots.

### D15. v1 vs v2 comparison design
- Same images, each model at its own frozen threshold (as each would be operated); thresholds never
  matched on test data.
- Exact McNemar for paired recall/FPR, paired bootstrap CIs, stratified paired bootstrap for ΔAUROC;
  pre-registered outcome rule implemented in code (`compare.classify`).
- Pairing is verified, not assumed: identical Gaussian states, fit lists and templates; v2's
  full-frame score equals v1's score on every image.

## Code-level risks discovered and fixed

| # | Risk | How found | Fix | Evidence |
|---|---|---|---|---|
| R1 | Phase-correlation shift estimate against the mean image was biased to 0 (each image is part of the mean) and reported "no misalignment" | Visual contact sheet contradicted it | Replaced with mask-moment pose; misleading statistic removed | `inspection_stats.json` now reports pose percentiles |
| R2 | Fixed-pose assumption F1 false (180° rotations, shifts) | Inspection | Registration (D2) | Registered mean image; orientation margins |
| R3 | `W[None] @ diff` broadcasting could materialise B×P×d×d (≈3.2 GB at batch 8) | Code review before first run | Single `torch.bmm` over positions | `test_mahalanobis_matches_scipy` |
| R4 | Windows cp950 default encoding crashed config reads (non-ASCII comments) | Test failure | Explicit UTF-8 for all text I/O | Test suite passes |
| R5 | CSV/JSON written with CRLF on Windows while git stores LF. The freeze hash of the split manifest would fail on a clean clone, blocking evaluation | Git warning on commit | LF-only writers, line-ending-normalised manifest hash, `.gitattributes`. Re-freeze **before** any test scoring reproduced byte-identical thresholds and model/config hashes | `test_manifest_hash_ignores_line_endings`; commit `f643526` |
| R6 | An AI-generated bulk regex edit mangled several lines (merged method calls, literal newlines) | `grep` review before running | Fixed by hand; full test suite re-run | Commit history; tests |
| R7 | Possible orientation leakage into scores (D12) | Post-hoc registration QC | Verified absent on normals; documented | `reports/diagnostics.json`, this log |
| R8 | `ivad report` wrote the README results block using the hard-coded repository root, not the config's root. Running it against another project root (e.g. the synthetic test) would overwrite the real README with foreign numbers | Self-review while extending the CLI smoke test | Paths resolved via `project_root(cfg)` | CLI test asserts the real README is untouched |
| R9 | Report figures hard-coded split sizes in labels ("n=150") and a fixed threshold-colour map | Extended CLI smoke test (different sizes and threshold names) | Labels from actual counts; colour fallback | `tests/test_cli.py` |
| R10 | Orientation-confound summary crashed when a class had no rotated images | Same smoke test | Guarded; summary skipped when not computable | `tests/test_cli.py` |
| R11 | Colour board mask misses metallic/black parts (pins, header base). It is used for registration (fine) and was used for the v1 "false alarms on background" diagnosis and for the first ROI rule | PCB2 FIT visual QC | ROI rule A1 (D14); v1 diagnosis re-checked with the A1 ROI: still 13/13 off-board | `reports/pcb2_inspection/roi_rule_A0_vs_A1.png`, `reports/pcb1_comparison/` |
| R12 | Freeze guard proved internal consistency only; it could not show the frozen state was recorded before testing | Design review for v2 | `evaluation.require_committed_freeze`: evaluate refuses unless config, split, freeze file and `src/` are committed and unchanged; HEAD recorded in `metrics.json` | `tests/test_cli.py::test_committed_freeze_guard` |
| R13 | Split audit file name was shared by all datasets and would be overwritten by a second dataset | PCB2 split | Per-dataset name `split_<obj>_audit.json` | `manifests/` |

## Verification evidence

- `pytest`: unit tests for split isolation and the leakage audit; shapes and input validation;
  Mahalanobis vs SciPy; streaming covariance vs NumPy (including a large offset); singular and
  non-PD covariance; conformal ranks and the Beta FPR law (checked by simulation); AUROC vs
  scikit-learn with ties; Clopper–Pearson vs SciPy; paired bootstrap; cost model; registration
  recovering flip, shift and tilt on synthetic boards; perturbation determinism; PDF number
  traceability. Plus an end-to-end CLI run on synthetic data that checks repeatability and that the
  freeze guard blocks evaluation after a config change.
- **Repeatability on real data:** re-running `calibrate` reproduced byte-identical thresholds on the
  GPU (R5).
- **Clean-environment reproduction:** a fresh clone and fresh venv reproduced byte-identical model,
  thresholds, per-image scores and metrics (`docs/handoff.md`, end). That run used code that already
  included a post-evaluation refactor of `align.py`, which shows the refactor did not change
  behaviour.
- `ivad check`: metrics recomputed from per-image scores; freeze integrity and ordering; split audit;
  every number in the PDF traced to recorded outputs; README block equals metrics; repo hygiene.
- Tests prove the code does what the tests state. They do not prove the requirements, thresholds or
  diagnosis are right for a real line.

## Remaining technical limitations

1. Recall at the primary operating point is below the proposed 90% target on both board types
   (PCB1 v1 72%, PCB2 v2 80%). On PCB2 the misses are mostly the smallest defects.
2. The board ROI (v2) removed most background-driven false alarms on PCB2 but did not raise recall
   there (`docs/v2_decision_record.md`).
3. Lighting: on PCB2 a 10% darkening raises false alarms by more than 20 points. On PCB1, blur raised
   false alarms by about 4 points.
4. Registration relies on the blue-board / grey-background colour contrast; a different product or
   background would need new mask thresholds.
5. 600 fit images × d = 128 is modest. Per-position covariances in highly variable regions (sensor
   meshes) are broad, which reduces sensitivity there.
6. Only one dataset and synthetic perturbations; no real-line or multi-camera evidence.
7. Model state is ≈ 406 MB (float32 whitening matrices). Fine on a workstation, but large for edge
   devices; float16 or a low-rank form would need validation.
