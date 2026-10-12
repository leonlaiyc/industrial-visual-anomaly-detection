# Exploring PCB Anomaly Detection with CNN and YOLO

A hands-on Computer Vision PoC by Yu-Chen (Leon) Lai. I wanted to understand visual anomaly detection by
building a small experiment with the public VisA PCB dataset: prepare the data, inspect failure cases,
try normal-only CNN modelling, improve and stress-test preprocessing, and train supervised YOLO models.
The website is the concise showcase; this repository retains the technical source and evidence.

- **Round A / PCB1:** inspect orientation and registration, extract pretrained WideResNet-50-2 features,
  fit per-position Gaussians and use Mahalanobis distances for anomaly scores and heatmaps.
- **Round B / PCB2 and PCB3:** ROI scoring on PCB2 reduced test-normal false alarms from 13% to 3% while
  recall stayed at 80%. A separate PCB3 illumination-correction experiment improved some conditions but
  worsened false alarms under blur; the candidate was not adopted.
- **Round C / PCB4:** convert defect masks to boxes and actually train YOLOv8s / YOLO11s using one generic
  `defect` class. Observed recall was higher in this supervised setting, with brightness confounding,
  unseen-defect generalization and localization limitations. This is exploratory, not a controlled claim
  of architecture superiority. The PCB datasets and supervision regimes differ.

Project sheet (English, one self-contained A4 page): [executive-summary-one-page.pdf](reports/executive-summary-one-page.pdf).
Bilingual project showcase (English / 繁體中文): built from `site/` (see *Website and PDF* below).
Retired briefs are labelled historical evidence in `docs/history/`; they are not website downloads.

AI-assisted development; public benchmark images only. No validation on real manufacturing-line images
and **no production-readiness claim**. Source is publicly available under AGPL-3.0-only.
D5 was accepted by the project owner on October 11, 2026; overall acceptance of the final outward-facing
portfolio is separate. The original protocols, scores, thresholds, splits, repair and deviation records remain.

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

### Recall by board type — separate datasets, not a progression

PCB1, PCB2 and PCB3 are different VisA board types. Each row was fitted, calibrated and evaluated on its own
board type, so values in different board-type rows are **not** a matched comparison and do not show an
improvement over time. Only rows on the same board type are paired comparisons (PCB2 v1 vs v2; PCB3
baseline vs candidate).

| Board type | Version / arm | Evidence | Recall, official test [95% CI] | FPR, official-test normals | FPR, holdout normals | AUROC |
|---|---|---|---|---|---|---|
| PCB1 | v1 (original) | development board type | 72% (72/100) [62%, 81%] | 7% (7/100) | 3.9% (6/154) | 0.938 |
| PCB1 | v2, retrospective | post hoc, on the data that motivated v2 | 88% (88/100) [80%, 94%] | 6% (6/100) | 1.9% (3/154) | 0.974 |
| PCB2 | v1 | confirmatory, held-back board type | 80% (80/100) [71%, 87%] | 13% (13/100) | 5.3% (8/151) | 0.927 |
| PCB2 | v2 | confirmatory, held-back board type | 80% (80/100) [71%, 87%] | 3% (3/100) | 4.0% (6/151) | 0.979 |
| PCB3 | v2 pipeline = baseline arm of the illumination experiment | confirmatory, new board type | 90% (90/100) [82%, 95%] | 3.0% (3/101) | 5.2% (8/155) | 0.982 |
| PCB3 | v2 + illumination correction = candidate arm | confirmatory, new board type | 91% (91/100) [84%, 96%] | 3.0% (3/101) | 3.9% (6/155) | 0.983 |

The PCB3 baseline is the unchanged v2 pipeline fitted to PCB3, not a new model version. Its 90/100 is a
point estimate on 100 benchmark images and does not guarantee a 90% recall on a real line.

### Follow-up: pre-registered lighting-correction experiment (PCB3)

A third board type, never used before, tested one change: standardise each image's global brightness and
contrast before registration. Protocol, addenda and freeze were committed before any PCB3 test image was
scored; the evaluation ran once ([`docs/pcb3_illumination_results.md`](docs/pcb3_illumination_results.md)).

Two decisions are kept separate:

- **A. Pre-registered experimental outcome: "ROBUSTNESS IMPROVEMENT SUPPORTED", weakly informative.**
  Under the same 10% darkening (G1) the corrected pipeline's false-alarm rise on 256 unseen normals was
  −0.4 pts (9 → 8 false alarms) against +4.3 pts uncorrected (11 → 22); difference −4.7 pts, 95% CI −8.6 to
  −0.8. The uncorrected rise was below the protocol's 5-pt informativeness line, so PCB3 barely reproduced
  the PCB2 failure, and a global darkening is what a global correction undoes by construction. The finding
  means only that the correction did what it was designed to do on this synthetic condition.
- **B. Engineering adoption decision: DO NOT ADOPT the illumination candidate as the preferred pipeline.**
  It improves global-brightness behaviour (brightening rise +15.2 → +2.7 pts) but raises false alarms under
  a mild blur by **+14.5 pts** (9 → 46 of 256; uncorrected +0.4), which fails the original P5 robustness
  criterion (≤ 10 pts). Non-uniform lighting was not improved. Clean results are unchanged (recall 90/100 vs
  91/100, official-test FPR 3/101 for both, AUROC 0.982 vs 0.983).
- **The baseline is not production-ready either.** It fails P5 under brightening (+15.2 pts), and neither
  variant has real end-user validation, a factory pilot or a measured operational return. Against the
  unchanged criteria both PCB3 variants are **INCONCLUSIVE**. The PCB2 result above is unchanged by this
  experiment.
- **Mechanism not separated:** FIT-only diagnostics show the correction also stabilises board registration
  under darkening; the experiment cannot split registration from feature effects.

**Post-evaluation analysis-code repair (disclosed).** After the PCB3 test scores had been generated and
summary test metrics seen, the pre-registered paired-analysis command refused to run (twice, unchanged
code) because an input check assumed VisA labels anomalies as `anomaly`; VisA stores the defect type. The
check was corrected in `7dbd302` (one condition in `defect_descriptors`, which feeds only a descriptive
subgroup table). No score, threshold, prediction, primary statistic or decision rule changed, nothing was
re-scored, and an independent script that imports none of the project code reproduces every recorded
PCB3 number from the raw score files (90/90 checks). Full record:
[`docs/pcb3_post_evaluation_repair.md`](docs/pcb3_post_evaluation_repair.md); all post-test events:
[`docs/illumination_experiment_deviations.md`](docs/illumination_experiment_deviations.md). An earlier,
pre-test deviation (an addendum committed after fitting rather than before) is disclosed in protocol
Addendum B2.

## Decision status

Three different questions, answered separately:

| Question | Answer |
|---|---|
| A. Pre-registered statistical test of v2 vs v1 (paired, PCB2) | Recall: **no meaningful effect demonstrated**. False-alarm non-inferiority holds. |
| B. Prototype against the proposed acceptance criteria | Recall (≥ 90%) **not met**; lighting robustness **not met**; false alarms, AUROC and latency met. Rule-based verdict **INCONCLUSIVE**: no NO-GO rule fired, because recall sits exactly at the 80% floor. INCONCLUSIVE does not mean the criteria were met. |
| C. Recommendation for manufacturing use | Not ready for production-affecting deployment; proposed shadow validation requires site authorization and entry checks. |

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
  integration, factory data, and any claim about yield, cost savings or financial return on investment.
  ("ROI" in the code means the image *region of interest*, never business return.)

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
- **Consistency gates:** `ivad check` recomputes every PCB1/PCB2 metric from per-image scores and traces every
  number in the executive PDF to a recorded output; `python verification/recompute_pcb3.py` does the same
  for PCB3 without importing the project code.

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

## Website and PDF

[Live website](https://leonlaiyc.github.io/industrial-visual-anomaly-detection/) · [繁體中文](https://leonlaiyc.github.io/industrial-visual-anomaly-detection/zh-tw/)

```bash
python site/build.py            # -> site/dist/ (English at /, Traditional Chinese at /zh-tw/)
python -m http.server 8765 --directory site/dist
python site/render.py pdf       # -> reports/executive-summary-one-page.pdf (+ identical site/dist/ copy)
python -m ivad check            # page size/count, recorded-score consistency and numeric provenance
```

`site/facts.py` is the only source of numbers for both languages and the PDF; the build fails if a content
file contains a literal metric or the two languages define different keys. Existing genuine figures are
reused; building this portfolio does not train or rescore models. Historical figure generation is separate.
`main` contains sanitized source and technical evidence; `gh-pages` contains only built static files.
`site/release_links.json` records actual deployed URLs only. An empty website URL means hosting is pending,
and the PDF says so. Public source links lead to the AGPL-licensed repository.

## Limitations

Three board types from one public benchmark, laboratory images, 100 defective images per board type (recall
is uncertain by roughly ±8 points), synthetic perturbations, and a 50% defect rate that says nothing about
a real line. Each new product, camera, background or lighting setup needs re-fitting, re-calibration and
fresh validation.

## Exploratory: YOLOv8s / YOLO11s on PCB4 (separate from the frozen results)

To test the main alternative rather than describe it, off-the-shelf YOLOv8s and YOLO11s were trained on the
fourth VisA board type (PCB4, never used before) under a protocol committed before the data was opened
([`exploratory/yolo_pcb4/PROTOCOL.md`](exploratory/yolo_pcb4/PROTOCOL.md)): fixed default settings, five-fold
cross-validation over the 100 defects (80 box-labelled defects per fold), the project's own threshold rule
(P95 of 150 separate normals). Results ([`results.md`](exploratory/yolo_pcb4/results/results.md)), all
descriptive:

The detector has one class, `defect`: ground-truth segmentation components are converted to boxes. It outputs
generic defect locations and confidence scores, not Missing/Extra/Scratch subtype predictions. Training needs
representative box-labelled examples; expected defect variation, label availability and maintenance cost must
be evaluated. There is no universal requirement for many examples of every subtype.

- With labels for the same defect types, recall was 98% (YOLOv8s) and 95% (YOLO11s) against 60% for the
  project's normal-only pipeline run on PCB4 without board-specific preprocessing, at similar overall
  false-alarm rates (3.1–4.3% of test normals; the same 255 normals are scored by all five fold models).
  This compares learning regimes with different label budgets, not architectures; YOLOv8s and YOLO11s
  differed on only 5 of the 100 defects and cannot be ranked on this sample.
- Post-hoc checks (added after the results were seen): the box that set the image score touched the
  defect (mask grown by 16 px, no IoU requirement) in 92/98 (YOLOv8s) and 89/95 (YOLO11s) flagged images;
  at IoU ≥ 0.5 with a ground-truth box, 79/98 and 74/95. Defect images are darker on average than normals,
  and YOLO scores on defect-free images rise with darkness. At the original frozen thresholds, the observed
  pooled FPRs were 4.0% and 3.1%. Post-hoc exploratory reweighting of normals to an artificial distribution
  matching defect-image brightness estimated conditional FPRs of about 9.0% and 6.9% (normal-only 2.1%).
  These estimates are not directly observed real manufacturing FPR. The previously recorded sensitivity
  analysis with alternative brightness-matched thresholds retained 96% and 91% image-level recall; it neither
  establishes the contribution of acquisition cues nor shows that brightness explains the entire recall
  advantage. Image-level flagging recall is distinct from correct defect localization. See [`exploratory/yolo_pcb4/audit/`](exploratory/yolo_pcb4/audit/AUDIT.md).
- On a defect type removed from training, YOLO11s fell to about the normal-only level (`missing`: 19/19 seen
  vs 11/19 unseen; normal-only 10/19), on tiny samples; none of the 11 unseen-type flags had a box at
  IoU ≥ 0.5 on the missing part.
- Latency figures (YOLO 14–15 ms, normal-only 32 ms, with registration 135 ms) are per-pipeline
  measurements at different input sizes and boundaries, not a speed ranking.
- It changes no PCB1–PCB3 result, threshold or decision; PCB4 numbers are not comparable with them.
  Deviations and the post-hoc check are logged in `exploratory/yolo_pcb4/results/deviations.md`.

A [supplementary false-alarm contact sheet](exploratory/yolo_pcb4/audit/FALSE_ALARM_SUPPLEMENT.md), produced
after the original evaluation, shows six verified TEST-N normal images at the original frozen thresholds.
It uses saved scores only, draws no unsupported boxes and preserves the original missing-deliverable record.

## Next steps (not done)

1. Lighting robustness: a global photometric correction was tested on PCB3 (see above) and traded darkening
   sensitivity for blur sensitivity; a per-image brightness/contrast **check** and controlled lighting are the
   safer next candidates.
2. Small-defect sensitivity: higher-resolution features or more training normals.
3. Confirm targets, costs, prevalence and must-catch defect types with a real user.

Each needs a new freeze and an evaluation on data that did not motivate it.

## Documents

| Document | Content |
|---|---|
| [`docs/v2_decision_record.md`](docs/v2_decision_record.md) | why v2, how it was tested, what the results do and do not show |
| [`docs/illumination_experiment_protocol.md`](docs/illumination_experiment_protocol.md) | PCB3 pre-registration, Addendum B1 (board-mask adaptation), Addendum B2 (chronology disclosure, FIT-only diagnostics) |
| [`docs/pcb3_illumination_results.md`](docs/pcb3_illumination_results.md) | PCB3 results, the two decisions (rule outcome vs adoption), all conditions, interpretation |
| [`docs/pcb3_post_evaluation_repair.md`](docs/pcb3_post_evaluation_repair.md) | incident record: analysis-code repair after PCB3 test scoring, chronology, patch, hashes |
| [`docs/illumination_experiment_deviations.md`](docs/illumination_experiment_deviations.md) | every post-freeze event of the PCB3 experiment (the protocol itself is frozen) |
| [`reports/pcb3_verification/recompute.md`](reports/pcb3_verification/recompute.md) | programmatic recomputation of every PCB3 number from the raw score files by a separate script |
| [`docs/v2_experiment_protocol.md`](docs/v2_experiment_protocol.md) | pre-registered hypothesis, single change, PCB2 protocol, amendment A1 |
| [`docs/requirements.md`](docs/requirements.md) | proposed requirements, acceptance criteria, go/no-go rules, open user questions |
| [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md) | splits, freeze, statistics |
| [`docs/engineering_decisions.md`](docs/engineering_decisions.md) | design decisions, rejected alternatives, risks found and fixed |
| [`docs/handoff.md`](docs/handoff.md) | setup, operation, failure modes, what a production version would need |
| [`reports/pcb2_comparison/comparison.md`](reports/pcb2_comparison/comparison.md) | generated paired comparison |
| [`reports/pcb3_illum_comparison/comparison.md`](reports/pcb3_illum_comparison/comparison.md) | generated PCB3 paired illumination analysis |
| [`exploratory/yolo_pcb4/PROTOCOL.md`](exploratory/yolo_pcb4/PROTOCOL.md) | exploratory YOLOv8s/YOLO11s study on PCB4: protocol, split audit, scores, results, deviations |

## Development provenance

This project was developed with AI coding assistance from Claude Code and Codex under the project
owner's direction. Design decisions, evidence and the
protocol history are documented in `docs/`. Verification is (A) programmatic — `ivad check` and
`verification/recompute_pcb3.py` recompute the reported metrics from the recorded per-image scores — and
(B) AI-assisted technical review; neither is a human sign-off, and no independent human audit has occurred
(scope and limits: `reports/project_milestones.json`). (C) The project owner's (Yu-Chen (Leon) Lai)
technical review of the final deliverables is pending.

This repository starts from a curated snapshot. Commit identifiers cited in `docs/` refer to the original
development history, which is kept separately because it also contains personal working notes. The
ordering it documents (protocol before data access, freeze before evaluation, one evaluation) is also
recorded in the timestamps and hashes inside `reports/*/frozen_operating_point.json` and
`reports/*/metrics.json`.

## Data, weights and licensing

- **Dataset:** VisA (Visual Anomaly), Amazon Science, `pcb1`, `pcb2` and `pcb3` subsets (`pcb4` only in the
  exploratory study), **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>). Zou et al., *SPot-the-Difference Self-Supervised
  Pre-training for Anomaly Detection and Segmentation*, ECCV 2022, <https://github.com/amazon-science/spot-diff>.
  The dataset itself is not redistributed (`ivad prepare` downloads it). The repository and website **do**
  contain a small number of derivative images: figures in `reports/` and the website images in
  `site/static/img/` show VisA images that were registered, cropped and resized, with heat-map, outline and
  mask overlays added. See [`NOTICE.md`](NOTICE.md).
- **Pretrained weights:** torchvision `Wide_ResNet50_2_Weights.IMAGENET1K_V1`, downloaded at run time,
  not redistributed. Trained on ImageNet, whose terms limit use to non-commercial research and education.
  Legal review is needed before any commercial use.
- **Libraries:** PyTorch, torchvision, NumPy, SciPy, Pillow, Matplotlib, ReportLab, pypdf, PyYAML;
  pytest and scikit-learn for tests.
- **This project's source code:** GNU **AGPL-3.0-only**, including the CNN implementation and
  YOLO exploratory scripts. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). VisA images,
  pretrained weights and third-party libraries retain their own terms; this is not an MIT-only project.
- No endorsement by the dataset authors or by any company is implied. The CC BY 4.0 licence applies to the
  VisA images and their derivatives only, not to this project's code.

## Release preparation and navigation

D5 is **ACCEPTED BY PROJECT OWNER**, dated 2026-10-11 (Asia/Taipei), by Leon's explicit conversational confirmation. The non-score-affecting post-evaluation repair does not invalidate PCB3's primary evaluation; complete chronology and deviation disclosure remain in [the repair record](docs/pcb3_post_evaluation_repair.md). The owner authorized public source release and GitHub Pages deployment on 2026-10-12. Overall portfolio acceptance remains separate.

Canonical website/source/evidence URLs are configured once in `site/release_links.json`. The website URL is recorded after GitHub Pages assigns it and anonymous access is verified. `python site/build.py` builds a local preview; `python site/build.py --release` rejects missing or anonymously inaccessible URLs. See [release and hosting steps](docs/release_preparation.md), [licensing review](docs/licensing_review.md), and [canonical artifact reconciliation](docs/artifact_reconciliation.md). Recipient PDFs state that the website link is pending first deployment and contain no localhost links. Re-render the active one-page PDF after URL changes, then verify and regenerate the release manifest. No model or experiment rerun is required.
