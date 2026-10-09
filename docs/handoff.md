# Engineering handoff

For an engineer who did not build this proof of concept and must reproduce, operate or extend it.

**Status.** Although v2 improved false-positive performance, it did not meet the proposed detection-sensitivity and robustness requirements. It is therefore not recommended for a manufacturing pilot without further work.
It is not production-ready (`docs/v2_decision_record.md`, "Decision summary").

## 1. What it is

- **Input:** one RGB image of a VisA PCB1- or PCB2-style board from a fixed overhead camera.
- **Output:** an anomaly score, **REVIEW / NO_REVIEW** at the frozen threshold, and an optional heat map PNG.
- **Learning:** defect-free images only; each board type has its own fitted model, board region and threshold.
- **Variants:** v1 scores the maximum of the anomaly map over the whole frame; v2 (recommended of the two)
  takes it over a fixed board region (ROI).
- **Not included:** disposition, defect classification, precise segmentation, service/API/UI, MES
  integration, any evidence about a real line.

## 2. Run it

Python 3.11, [uv](https://docs.astral.sh/uv/) or pip, about 4 GB free disk; an NVIDIA GPU is optional.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
uv pip install --python .venv -e . --no-deps
python -m pytest
```

- **CPU only:** install `torch`/`torchvision` from the CPU index and set `runtime.device: cpu`. `device`,
  `batch_size` and `num_workers` are excluded from the frozen config hash.
- **Score one image:** `python -m ivad --config config/pcb2_v2.yaml predict path/to/image.JPG --out heatmap.png`
  prints JSON with score, threshold, recommendation and ROI coverage.

## 3. Reproduce the reported metrics

| Run | Config | Outputs |
|---|---|---|
| PCB1 v1 (original) | `config/default.yaml` | `reports/` (top level) |
| PCB2 v1 baseline | `config/pcb2_v1.yaml` | `reports/pcb2_v1/` |
| PCB2 v2 (board ROI) | `config/pcb2_v2.yaml` | `reports/pcb2_v2/`, `reports/pcb2_comparison/` |
| PCB1 v2, retrospective | `config/pcb1_v2_retrospective.yaml` | `reports/pcb1_v2_retrospective/`, `reports/pcb1_comparison/` |
| PCB3 baseline (v2 pipeline) | `config/pcb3_baseline.yaml` | `reports/pcb3_baseline/` |
| PCB3 candidate (v2 + global luminance correction) | `config/pcb3_illum.yaml` | `reports/pcb3_illum/`, `reports/pcb3_illum_comparison/` |

The frozen operating points are committed. Re-fit and evaluate against them; **do not re-run
`calibrate`**, which writes a new freeze (and, for PCB2, blocks `evaluate` until it is committed).

```bash
python -m ivad --config config/pcb2_v2.yaml prepare     # official archive (1.93 GB, once) + extract
python -m ivad --config config/pcb2_v2.yaml split       # `git diff manifests/` must stay empty
python -m ivad --config config/pcb2_v1.yaml fit
python -m ivad --config config/pcb2_v2.yaml fit
python -m ivad --config config/pcb2_v1.yaml evaluate    # refuses unless model hash == committed freeze
python -m ivad --config config/pcb2_v2.yaml evaluate
python -m ivad --config config/pcb2_v1.yaml robustness
python -m ivad --config config/pcb2_v2.yaml robustness
python -m ivad compare                                  # paired PCB2 v1 vs v2
python -m ivad summary                                  # README results block + executive PDF
python -m ivad check                                    # consistency gates; exit code 1 on any failure
```

PCB1 works the same way with `config/default.yaml`. PCB3 (illumination experiment,
`docs/illumination_experiment_protocol.md`): `fit` and `evaluate`/`robustness` both PCB3 configs against the
committed freeze, then `python -m ivad illum-compare`; the FIT-only diagnostics of Addendum B2 are
`python -m ivad fit-diagnostics`. **Do not re-run PCB3 `calibrate`, `evaluate` or `robustness`: its test data
is spent.** To verify the recorded PCB3 numbers without the project code, run
`python verification/recompute_pcb3.py --check` (reads only the committed score files; with the dataset
present it also recomputes the subgroup table from the masks). The analysis program `illum.py` was repaired
once after PCB3 test scoring (`docs/pcb3_post_evaluation_repair.md`). The illumination correction is **not
adopted**: keep `preprocess.photometric` disabled (`config/pcb3_baseline.yaml` is the reference). Website and PDF: README → *Website and PDF* (run `site/build.py` before
`ivad check`, whose PDF gate traces numbers through `site/data/facts.json`). Afterwards `git diff reports/` should show only
timestamps, git HEAD and latency. Seeds and cuDNN determinism are fixed; another GPU or driver may change
late decimals. `ivad check` checks internal consistency, so also compare the regenerated `metrics.json`
with the committed one.

## 4. Model assets

| Asset | Where | In git? |
|---|---|---|
| Fitted state per run (`model.pt`: means, whitening matrices, channel indices, registration template, v2 ROI) | `artifacts/<run>/model.pt`, about 406 MB each | no; regenerated by `fit`, SHA-256 in the freeze file |
| Frozen operating point (thresholds, hashes) | `reports/<run>/frozen_operating_point.json` | yes |
| Pretrained weights | torch hub cache, `wide_resnet50_2-95faca4d.pth` | no; downloaded, SHA-256 in the freeze file |
| Dataset | `data/` | no; `ivad prepare` downloads the official archive |

**Pretrained weights terms.** torchvision code is BSD-3-Clause, but the ImageNet-trained weights carry
terms derived from ImageNet (non-commercial research and education). Adequate for this PoC; **legal
review is required before any commercial use.** Alternatives: weights with explicit commercial terms, or
self-supervised pre-training on the company's own images. Not redistributing the weights does not remove
these conditions.

## 5. Configuration ownership

| File | Content | Owner (proposed) | Changing it means |
|---|---|---|---|
| `config/<run>.yaml` | preprocessing, model, calibration candidates, robustness, ROI rules | ML engineering | new `fit` + `calibrate` + fresh evaluation data |
| `config/acceptance_criteria.yaml` | proposed targets and NO-GO rules | the process / quality owner (to be named) | changes reported status, never the model; must not be edited after seeing results |
| `reports/<run>/frozen_operating_point.json` | the operating threshold | the threshold owner (to be named) | a new freeze, recorded and committed, previous one kept for rollback |

No real user has been assigned to any of these roles.

## 6. Alignment and board-region assumptions

- **Registration** (`align.py`) assumes one rigid blue board on a grey cloth, fixed camera scale, and colour
  contrast between board and background. On a saturated background (PCB3: green cloth) the original colour
  rule silently marks almost the whole frame as board; PCB3 needs `preprocess.board_mask.exclude_hue`, a
  background hue band measured on training images (Addendum B1). The mask uses absolute S/V thresholds, so a
  10% darkening moved the estimated PCB3 pose by about 11 px (median) and triggered the ROI fail-safe on
  8/600 training images (Addendum B2). It corrects shift (about ±70–100 px), tilt (about ±3°) and 180°
  rotation; 0° vs 180° is decided by correlation with a template, and the margin is logged per image.
- **Board ROI** (`roi.py`, v2) is built once from 600 training normals: majority-vote board mask, plus
  board-attached structure such as header pins (amendment A1), plus a 16 px margin. It is stored inside
  `model.pt`, so it is covered by the model hash. It is **not** a tight board mask: on PCB3 about 31% of ROI
  pixels are background cloth (margin, shadow, and a cloth band the A1 std rule picks up).
- **Fail-safe:** if less than 95% of an image's own board mask lies inside the ROI, the image gets the v1
  full-frame score, which can only make REVIEW more likely. Fallbacks on PCB2: 0 of 200 test images and
  0 of 151 holdout images. Calibration stops if more than 5% of calibration normals fall back.
- A different product, background colour, camera or lighting breaks the colour mask and the learned
  normal model; re-validate the mask thresholds, re-fit and re-calibrate.

## 7. Operating threshold

- REVIEW iff score > the frozen **P95** threshold: the 144th of 150 sorted calibration-normal scores
  (split-conformal). PCB2 v2: 51.20; PCB2 v1: 55.03.
- It targets about 5% false alarms **only while** future normal images resemble the calibration images.
  With 150 calibration images the achieved rate can plausibly lie between about 2% and 9%.
- P90 and P99 candidates are recorded in the same file. Choosing among them for a real line needs the
  user's prevalence and costs (`docs/requirements.md` §3), not the benchmark.

## 8. Known failure modes

1. **Small and subtle defects are missed.** PCB2: recall about 56% in the smallest third of defects;
   PCB1: "melt" defects. Their heat maps usually peak on the defect, but below the threshold. The board ROI
   does not help here.
2. **Lighting sensitivity.** A 10% darker, lower-contrast image raises false alarms on the 251 PCB2 normal
   images from 3.6% to 28.7% (v2) and from 8.4% to 30.7% (v1). Normal PCB2 brightness already varies
   (training range 83–95 grey levels).
3. **Background dust (v1).** v2 removes most of these false alarms; dust inside the 16 px margin can still
   trigger.
4. **Blur** raised false alarms on PCB1 (+4 points) but not on PCB2. With the PCB3 global luminance
   correction, a mild blur raised false alarms by +14.5 points (uncorrected +0.4): the correction trades
   brightness sensitivity for blur sensitivity (`docs/pcb3_illumination_results.md`).
5. **Registration** can be biased when a defect changes the board outline; watch the orientation margin.

## 9. Support considerations

- **Drift monitoring:** track the weekly REVIEW rate on known-good boards and compare the score
  distribution with `reports/<run>/scores_calib.csv`. Lamp ageing, a new cloth, a board revision or dust
  build-up silently changes the false-alarm rate.
- **Re-calibration:** collect at least 300 recent known-good images, run `calibrate` (new freeze), record
  and commit it, keep the previous freeze for rollback, then validate on fresh images.
- **Over-trust:** NO_REVIEW is not a pass. The tool must not replace existing inspection.
- **Alarm fatigue:** if background or lighting drives flags, reviewers learn to ignore them.
- **Benchmark confound:** most PCB1 test anomalies are rotated 180°; any future model without registration
  could look better than it is.

## 10. What a production version would need

Each item needs a new freeze and an evaluation on data that did not motivate it.

1. **Requirements confirmed by a real user:** cost ratio and prevalence, tolerable review workload,
   must-catch defect types and minimum size, takt time, acquisition stability, who owns the threshold,
   and the go/no-go criteria themselves (`docs/requirements.md` §4). None is confirmed.
2. **Lighting robustness:** a global photometric normalisation was tested on PCB3; it removed the
   darkening/brightening rises there but added blur sensitivity and did not help non-uniform lighting, so it
   is not adopted. The uncorrected baseline is not production-ready either (brightening +15.2 points).
   Prefer controlled lighting plus a brightness-range guard that routes out-of-range images to review.
3. **Small-defect sensitivity:** higher-resolution features, more training normals, per-region covariance
   shrinkage, colour-aware features.
4. **Input quality guards:** focus (e.g. Laplacian variance) and brightness checks against the
   calibration range.
5. **Line data and a shadow-mode trial** with ground truth before any pilot decision.
6. **Packaging and service:** the model state is about 406 MB of float32 whitening matrices (consider
   float16 or on-the-fly computation, validated numerically); an API or batch service, logging, model and
   threshold versioning, monitoring and rollback.
7. **Licensing:** commercial-use review of the pretrained weights (§4).
8. Optional: an Anomalib PaDiM/PatchCore comparison under the same split and freeze.

---

## Clean-environment reproduction

Run on 2026-10-09 (Taiwan time) for the **v1 PCB1** pipeline by the implementer (AI-assisted), on the
same Windows 11 workstation; not yet repeated independently. For the v2 code, the regression check is
that it reproduces the PCB1 v1 calibration scores bit-for-bit, and that v2's full-frame score equals v1's
score on every PCB2 image (`reports/pcb2_comparison/comparison.json` → `pairing_checks`).

| Step | What was done |
|---|---|
| Source | `git clone` of commit `5a66747` into an empty scratch directory |
| Environment | new `uv` venv, Python 3.11.15, installed only from `requirements.lock.txt` (torch 2.14.1+cu126) + `pip install -e . --no-deps` |
| Data | already-extracted `data/visa/pcb1` copied in to avoid a second 1.93 GB download (`ivad prepare` verified all 1,104 files); torch weight cache shared (weight SHA-256 re-verified identical) |
| Command | `pytest` (all passed), then `python -m ivad all` (split → fit → calibrate → evaluate → robustness → report → pdf → check), 179 s wall time |
| Result | `ivad check`: all checks pass. **Byte-identical** to the original run: `scores_calib.csv`, `scores_test.csv`, `scores_holdout.csv`, `robustness_scores.csv`; identical model SHA-256, thresholds, `official_test`, `supplementary_holdout` and `robustness.json` contents |
| Not covered | fresh dataset download inside the clone; a different OS, GPU or driver; a CPU-only full run (CPU latency was measured separately on 20 images) |
