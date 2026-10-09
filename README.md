# Industrial Visual Anomaly Detection: PCB image review support

An independent proof of concept on public benchmark data. The prototype scores each printed-circuit-board
(PCB) image, recommends **REVIEW / NO REVIEW** at a threshold fixed before testing, and draws a heat map of
where the board looks abnormal. A person makes every disposition.

**Bottom line.** On a board type held back for an independent test (VisA PCB2), the refined version (v2)
cut false alarms on normal boards from 13% to 3% and kept defect recall at 80%. Recall did not improve and
stays below the proposed 90% target, and false alarms rise sharply on darker images.
Although v2 improved false-positive performance, it did not meet the proposed detection-sensitivity and robustness requirements. It is therefore not recommended for a manufacturing pilot without further work.

One-page summary: [`reports/executive_summary.pdf`](reports/executive_summary.pdf).

## Results

Official VisA test split per board type (100 normal + 100 defective images). Each model is operated at a
threshold frozen beforehand (P95 of 150 separate normal images). 95% confidence intervals are in brackets.

<!-- RESULTS:START -->
| Official VisA test split (100 normal + 100 defective) | Proposed target | PCB1 v1 (original) | PCB2 v1 | PCB2 v2 |
|---|---|---|---|---|
| Recall (defects sent to review) | >= 90% | 72% (72/100) [62%, 81%] | 80% (80/100) [71%, 87%] | 80% (80/100) [71%, 87%] |
| False-alarm rate, official test normals | <= 10% | 7% (7/100) [3%, 14%] | 13% (13/100) [7%, 21%] | 3% (3/100) [1%, 9%] |
| False-alarm rate, supplementary normal holdout | <= 10% | 3.9% (6/154) [1.4%, 8.3%] | 5.3% (8/151) [2.3%, 10.2%] | 4.0% (6/151) [1.5%, 8.4%] |
| Image AUROC | >= 0.90 | 0.938 [0.901, 0.966] | 0.927 [0.889, 0.956] | 0.979 [0.962, 0.991] |
| Median latency per image (GPU) | <= 1000 ms | 139 ms | 194 ms | 224 ms |
| TP / FN / FP / TN at the frozen P95 threshold |  | 72 / 28 / 7 / 93 | 80 / 20 / 13 / 87 | 80 / 20 / 3 / 97 |
| Status against the proposed criteria |  | NO-GO (recall below floor) | not met: recall, test FPR, robustness | not met: recall, robustness |

Paired PCB2 comparison (v2 − v1, same images, each at its own frozen threshold): recall +0 pts [-3, +3] (McNemar p = 1.00); test false alarms -10 pts [-17, -3] (p = 0.013); holdout false alarms -1.3 pts (p = 0.69); AUROC +0.052 [+0.028, +0.082]. Pre-registered outcome on the primary endpoint (recall): **no meaningful effect demonstrated**.
<!-- RESULTS:END -->

- **PCB2 is the confirmatory test.** The v2 protocol was committed before any PCB2 image was read, and
  both variants were evaluated once. **PCB1** produced the idea for v2, so it cannot confirm it.
- **Primary endpoint (recall): no change.** False-alarm reduction and the AUROC gain are secondary
  findings. Why recall did not move is analysed in [`docs/v2_decision_record.md`](docs/v2_decision_record.md) §5.
- **Do not mix the two false-alarm rows.** The official-test row is the benchmark result; the holdout row
  is an additional 151–154 normal images per board type, used only as a check.
- **Retrospective only:** v2 on PCB1 gives recall 88/100 and 6/100 test false alarms
  ([`reports/pcb1_comparison/`](reports/pcb1_comparison/comparison.md)). It was computed on the data that
  motivated v2 and is not independent evidence.
- **Robustness (PCB2, both variants):** a 10% darker, lower-contrast image raises false alarms on the 251
  normal images by more than 20 points at the frozen threshold.

## Decision status

Three different questions, answered separately:

| Question | Answer |
|---|---|
| A. Pre-registered statistical test of v2 vs v1 (paired, PCB2) | Recall: **no meaningful effect demonstrated**. False-alarm non-inferiority holds. |
| B. Prototype against the proposed acceptance criteria | Recall (≥ 90%) **not met**; lighting robustness **not met**; false alarms, AUROC and latency met. Rule-based verdict **INCONCLUSIVE**: no NO-GO rule fired, because recall sits exactly at the 80% floor. INCONCLUSIVE does not mean the criteria were met. |
| C. Recommendation for manufacturing use | Not recommended for a manufacturing pilot without further work. |

The criteria and decision rules were fixed before testing and have not been changed after seeing results
([`docs/requirements.md`](docs/requirements.md)). They are a proposal that no manufacturing user has
confirmed.

## Problem definition

- **Decision supported:** which board images need a manual review. One fixed overhead camera, one board per
  image.
- **Learning:** from defect-free images only (one-class), because real defects are rare and varied.
- **In scope:** image-level score and REVIEW flag, heat maps, empirical threshold calibration, statistics
  with confidence intervals, robustness checks, a reproducible command-line workflow, tests.
- **Out of scope:** defect classification, pixel-accurate segmentation, automatic disposition, line or MES
  integration, factory data, and any claim about yield, cost savings or ROI.

## Architecture

```text
image -> registration -> crop 1152x768, resize 768x512 -> Wide-ResNet-50-2 layer2+3 features
      -> per-position Gaussian (Mahalanobis distance) -> smoothed anomaly map
      -> image score: max over the frame (v1) or over a fixed board region (v2)
      -> REVIEW if score > frozen threshold
```

| Step | Module | What it does |
|---|---|---|
| Registration | `src/ivad/align.py` | board mask from colour, centroid and tilt from image moments, 0°/180° chosen by correlation with a template of training normals |
| Features | `src/ivad/features.py` | frozen ImageNet Wide-ResNet-50-2; fixed random 128 of 1536 channels; 3×3 local mean; 64×96 positions |
| Normal model | `src/ivad/gaussian.py` | per-position mean and ridge-regularised covariance from 600 normals; Mahalanobis distance via Cholesky whitening (implemented here, no Anomalib) |
| Board region (v2) | `src/ivad/roi.py` | fixed region from training normals only; fail-safe falls back to the v1 score if an image's board does not fit it |
| Threshold | `src/ivad/calibration.py` | split-conformal order statistic of 150 separate normal images |
| Freeze and evaluation | `src/ivad/pipeline.py`, `metrics.py`, `compare.py` | hashes config, model and split; refuses to evaluate if anything changed; exact and paired statistics |

Features are not assumed to be Gaussian: the distance is a score, and its threshold is set empirically.

## v1 → v2: the single change

PCB1 diagnostics showed that every false alarm on a normal board peaked on the background cloth (dust,
lint), not on the board. v1 takes the maximum of the anomaly map over the whole frame, so background
clutter can set an image's score and, through the calibration normals, the threshold. v2 changes only
where the maximum is taken: inside a board region built from training normals. Registration, features and
the Gaussian model are bit-identical between v1 and v2, and v2's full-frame score reproduces v1's score on
every image. One pre-test amendment to the region rule (to keep header pins that the colour mask missed)
was made from training images only, before any PCB2 model existed, and is disclosed in
[`docs/v2_experiment_protocol.md`](docs/v2_experiment_protocol.md).

## Evaluation methodology

- **Data roles** per board type: 600 FIT (model), 150 CALIB (threshold), 151–154 HOLDOUT (extra false-alarm
  check), official test 100 + 100. Deterministic, content-hashed and audited for leakage.
- **Freeze:** threshold, config, model and split hashes committed before any test image is scored; for
  PCB2, `evaluate` refuses to run unless that state is committed.
- **Statistics:** exact (Clopper–Pearson) intervals for recall and false-alarm rate; bootstrap for AUROC;
  paired exact McNemar tests and paired bootstrap for v1 vs v2.
- **Consistency gates:** `ivad check` recomputes every metric from per-image scores and traces every number
  in the executive PDF to a recorded output.

Details: [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md).

## Reproduce

Python 3.11, optional NVIDIA GPU (CUDA 12 wheels pinned), about 4 GB disk.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
uv pip install --python .venv -e . --no-deps
.venv/Scripts/python -m pytest          # on Linux: .venv/bin/python
```

The PCB2 study against the committed frozen thresholds (the first command downloads VisA, about 1.9 GB):

```bash
python -m ivad --config config/pcb2_v2.yaml prepare
python -m ivad --config config/pcb2_v2.yaml split
python -m ivad --config config/pcb2_v1.yaml fit
python -m ivad --config config/pcb2_v2.yaml fit
python -m ivad --config config/pcb2_v1.yaml evaluate
python -m ivad --config config/pcb2_v2.yaml evaluate
python -m ivad --config config/pcb2_v1.yaml robustness
python -m ivad --config config/pcb2_v2.yaml robustness
python -m ivad compare
python -m ivad summary
python -m ivad check
```

`evaluate` refuses to run if the fitted model, config or split differs from the committed freeze. Do not
re-run `calibrate` to reproduce results; it writes a new freeze. Full instructions, PCB1 and single-image
scoring: [`docs/handoff.md`](docs/handoff.md).

## Limitations

Two board types from one public benchmark, laboratory images, 100 defective images per board type (recall
is uncertain by roughly ±8 points), synthetic perturbations, and a 50% defect rate that says nothing about
a real line. Each new product, camera, background or lighting setup needs re-fitting, re-calibration and
fresh validation.

## Next steps (not done)

1. Lighting robustness: photometric normalisation or a brightness-range guard.
2. Small-defect sensitivity: higher-resolution features or more training normals.
3. Confirm targets, costs, prevalence and must-catch defect types with a real user.

Each needs a new freeze and an evaluation on data that did not motivate it.

## Documents

| Document | Content |
|---|---|
| [`docs/v2_decision_record.md`](docs/v2_decision_record.md) | why v2, how it was tested, what the results do and do not show |
| [`docs/v2_experiment_protocol.md`](docs/v2_experiment_protocol.md) | pre-registered hypothesis, single change, PCB2 protocol, amendment A1 |
| [`docs/requirements.md`](docs/requirements.md) | proposed requirements, acceptance criteria, go/no-go rules, open user questions |
| [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md) | splits, freeze, statistics |
| [`docs/engineering_decisions.md`](docs/engineering_decisions.md) | design decisions, rejected alternatives, risks found and fixed |
| [`docs/handoff.md`](docs/handoff.md) | setup, operation, failure modes, what a production version would need |
| [`reports/pcb2_comparison/comparison.md`](reports/pcb2_comparison/comparison.md) | generated paired comparison |

## Development provenance

This project was developed with AI coding assistance (Anthropic Claude Code), which wrote most of the
implementation and documentation under the project owner's direction. Design decisions, evidence and the
protocol history are documented in `docs/`. The project owner's technical review of the final deliverables
is pending.

This repository starts from a curated snapshot. Commit identifiers cited in `docs/` refer to the original
development history, which is kept separately because it also contains personal working notes. The
ordering it documents (protocol before data access, freeze before evaluation, one evaluation) is also
recorded in the timestamps and hashes inside `reports/*/frozen_operating_point.json` and
`reports/*/metrics.json`.

## Data, weights and licensing

- **Dataset:** VisA (Visual Anomaly), Amazon Science, `pcb1` and `pcb2` subsets, **CC BY 4.0**.
  Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*,
  ECCV 2022, <https://github.com/amazon-science/spot-diff>. Images are not redistributed; figures in
  `reports/` show downscaled, registered VisA images with overlays. See [`NOTICE.md`](NOTICE.md).
- **Pretrained weights:** torchvision `Wide_ResNet50_2_Weights.IMAGENET1K_V1`, downloaded at run time,
  not redistributed. Trained on ImageNet, whose terms limit use to non-commercial research and education.
  Legal review is needed before any commercial use.
- **Libraries:** PyTorch, torchvision, NumPy, SciPy, Pillow, Matplotlib, ReportLab, pypdf, PyYAML;
  pytest and scikit-learn for tests.
- **This project's source code:** no licence has been applied yet (all rights reserved). The proposed
  licence is MIT, covering this repository's own code and documentation only; it would not apply to the
  VisA data, the pretrained weights or third-party libraries, which keep their own terms.
- No endorsement by the dataset authors or by any company is implied.
