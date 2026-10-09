# Evaluation methodology

How the numbers in `reports/results.md` were produced, and what they can and cannot support.

This file describes the protocol as first applied to PCB1 (v1). The PCB2 study (v1 baseline vs v2 board
ROI) follows the same rules with these additions, pre-registered in `docs/v2_experiment_protocol.md`:

- PCB2 split: 600 FIT / 150 CALIB / **151** HOLDOUT / 100 + 100 official test (`manifests/split_pcb2.csv`).
- Both PCB2 models share the FIT and CALIB images and are calibrated independently; the frozen state must
  be committed to git before `evaluate` will run, and the evaluation HEAD is recorded.
- v1 vs v2 is analysed **paired** on the same images: exact McNemar and paired bootstrap for recall and
  FPR, stratified paired bootstrap for AUROC, and a pre-registered outcome rule (`src/ivad/compare.py`).
- PCB1 is discovery data for v2; any PCB1 v2 number is retrospective.

## 1. Data roles

| Role | Images | Source | Used for | Never used for |
|---|---|---|---|---|
| fit | 600 normal | official train | Gaussian model, registration template | thresholds, evaluation |
| calib | 150 normal | official train | choosing thresholds | fitting, evaluation |
| holdout | 154 normal | official train | supplementary FPR estimate | any development decision |
| test | 100 normal + 100 anomaly | official test | the benchmark result | any development decision |

- The split is deterministic: seed `20261008`, applied to path-sorted files.
- It is stored per file with SHA-256 in `manifests/split_pcb1.csv`.
- It is audited automatically: unique paths, roles consistent with the official split, no identical
  file content across roles, and exact counts (`manifests/split_audit.json`). The audit runs in
  `ivad split` and again in `ivad check`.

### Why there is an additional split, and how it differs from the official protocol
The official VisA 1-class protocol fits on all 904 train normals and reports threshold-free metrics
(AUROC) on the test split. An operational review tool also needs a **threshold**. Choosing it on the
test set would make the reported recall and FPR optimistic. We therefore hold out 150 normals to
choose thresholds and 154 more as an untouched false-alarm check.

Consequences:
- The model sees 600 rather than 904 normals, so results are expected to be slightly worse than an
  identical method under the official protocol.
- We also register and crop the boards (`docs/engineering_decisions.md` D2–D3).
- For both reasons, **published PCB1 numbers are not a like-for-like comparison** and none are quoted.

## 2. What was frozen, and when

All model settings, input resolution, the registration method, candidate thresholds (P90/P95/P99), the
primary threshold (P95) and the robustness perturbations were fixed in `config/default.yaml`. That file
was committed (`1c5b121`) before calibration.

Calibration scored only calib images and wrote `reports/frozen_operating_point.json`. It records the
SHA-256 of:
- the semantic config,
- the fitted model,
- the split manifest,
- the pretrained weights.

The freeze was committed (`fdd1f35`, re-frozen identically in `f643526` after a line-ending fix). Only
then were test and holdout images scored for the first time. `evaluate` refuses to run if any hash
changed, and `ivad check` verifies that the freeze timestamp precedes the evaluation timestamp.

## 3. Thresholds: target FPR vs measured FPR

For target FPR α and n = 150 calibration scores, the threshold is the k-th smallest score,
k = ⌈(n+1)(1−α)⌉, and an image is flagged iff its score exceeds it (split-conformal).

Three different quantities are reported and must not be conflated:

1. **Target FPR (α):** a design choice (10%, 5%, 1%).
2. **Expected FPR and its sampling range:** under exchangeability, the true FPR of a threshold
   estimated from n normals is Beta(n+1−k, k). This gives the range caused purely by having only 150
   calibration images.
3. **Measured FPR:** the observed false-alarm rate on unseen normals (official test normals; separately,
   the supplementary holdout), with Clopper–Pearson CIs.

The measured FPR need not equal the target. Agreement within the sampling range supports the
exchangeability assumption on *this* data. It does not show stability under line drift.

## 4. Metrics and uncertainty

| Quantity | Estimator | Interval |
|---|---|---|
| Image AUROC | Mann–Whitney (ties = ½) | stratified percentile bootstrap, B = 2000, seed 7 |
| Recall, FPR | proportions at a frozen threshold | Clopper–Pearson exact 95% |
| Confusion matrix | counts at each frozen threshold | — |
| Latency | median / p95 over 200 images, batch 1, after warm-up; decode → registration → CNN → score | — |
| Robustness Δ | paired per-image change at the frozen threshold | percentile bootstrap over original images, stratified, B = 2000 |

- Precision / PPV is **not** reported as an operational metric: the 50/50 test set has no relation to
  real prevalence.
- A pooled FPR over all 254 unseen normals is shown as supplemental description only. It mixes the
  official and supplementary sets and is not the benchmark result.

Acceptance status ("Met / NOT met") is computed by code from point estimates against
`config/acceptance_criteria.yaml`, the *proposed* targets from `docs/requirements.md`, which are pending
user confirmation.

## 5. Robustness protocol

- Four conditions from three families, applied to the native image before registration:
  - brightness and contrast ×0.9 / ×1.1;
  - Gaussian blur, σ = 1.5 px;
  - shift of (+40, +25) px.
- Images: 100 test anomalies and 254 unseen normals (test + holdout). Threshold unchanged.
- Perturbed copies are **paired** with their originals. Intervals resample original images, so the
  dependence between an image and its perturbed copy is respected. Flip counts (newly flagged / newly
  cleared) are reported alongside.
- Synthetic perturbations are a sensitivity probe. They are **not** evidence of robustness across real
  cameras, lines or factories.

## 6. Cost sensitivity

C(k, t) = k·(1 − Recall(t)) + FPR(t), with k = (π/(1−π))·(C_FN/C_FP). The terms:
- π is the defect prevalence;
- C_FN is the cost of a missed defect;
- C_FP is the cost of an unnecessary review;
- correct decisions are assumed to cost nothing.

C is the expected cost per board divided by (1−π)·C_FP. Reference policies: review none, C = k;
review all, C = 1. A threshold is only useful for k where it beats both, i.e.
FPR/Recall < k < (1−FPR)/(1−Recall).

- The plot uses the **official-test** Recall/FPR at the **pre-registered** thresholds. Comparing
  thresholds on it is retrospective sensitivity analysis. It is not threshold optimisation, and the
  primary threshold was not chosen from it.
- k is unknown. A real user supplies π, C_FN and C_FP, computes k, reads off which policy is cheapest,
  and should re-check with Recall/FPR measured on their own line data.
- No prevalence, cost, saving or ROI figure is claimed.

## 7. Post-hoc diagnostics

Recall by defect type and size, heat-map peak location, and the orientation confound all use test
labels, masks and annotations. They are computed after the freeze, by code that cannot change the model
or threshold, and are labelled post hoc everywhere. They explain failures. They must not be used to
re-tune this evaluation; any change they motivate needs a new freeze and fresh evaluation data.

## 8. Benchmark validation vs factory readiness

This evaluation measures separation of normal vs anomalous PCB1 images from a public benchmark, under a
frozen protocol, with quantified uncertainty. It does not measure:
- performance on a real line;
- performance under real prevalence;
- long-term drift;
- real costs.

See `docs/requirements.md` §I.
