# v2 decision record

**Status:** draft for the project owner's review. Prepared with AI assistance; not yet reviewed or
approved by the owner.

Sources for every number: `reports/metrics.json` (PCB1 v1), `reports/pcb2_v1/`, `reports/pcb2_v2/`,
`reports/pcb2_comparison/comparison.json`, `reports/pcb1_comparison/comparison.json`. The protocol is
`docs/v2_experiment_protocol.md`.

## Decision summary

Three separate questions. Their answers must not be merged.

| Level | Question | Answer | Source |
|---|---|---|---|
| A | Pre-registered statistical decision rule: does v2 improve recall over v1 on PCB2 (paired)? | **NO MEANINGFUL EFFECT DEMONSTRATED** (Δ recall +0 pts, exact McNemar p = 1.00). FPR non-inferiority holds. | §5 D; `reports/pcb2_comparison/comparison.json` |
| B | Does the prototype meet the proposed acceptance criteria? | No. Recall (80% vs ≥ 90%) and lighting robustness are **not met**. Rule-based verdict: **INCONCLUSIVE** (no NO-GO trigger; recall exactly at the 80% floor). INCONCLUSIVE is not a pass. | §6; `config/acceptance_criteria.yaml` |
| C | Is it ready for a manufacturing pilot? | No. | this record, §7 |

In plain language: Although v2 improved false-positive performance, it did not meet the proposed detection-sensitivity and robustness requirements. It is therefore not recommended for a manufacturing pilot without further work.

The decision rules and acceptance criteria are the ones fixed before testing. They have not been changed
after the results were seen; known gaps in them (§6) are recorded for a future evaluation, not applied
retroactively.

## 1. Why v1 did not meet the requirements

v1 was evaluated once on the official VisA PCB1 test split at a threshold frozen beforehand (P95 of 150
calibration normals).

| Proposed criterion | v1 on PCB1 | Status |
|---|---|---|
| Recall ≥ 90% | 72/100 (95% CI 62–81%) | not met; below the 80% NO-GO floor |
| FPR ≤ 10%, test normals | 7/100 | met |
| FPR ≤ 10%, holdout normals | 6/154 (3.9%) | met |
| AUROC ≥ 0.90 | 0.938 | met |
| Median latency ≤ 1 s | 139 ms (RTX 4070 Ti) | met |

The ranking was good, but at the frozen threshold too many defects scored below it.

## 2. Evidence that motivated the ROI change

Post-hoc diagnostics on the PCB1 test evaluation:

- All 13 false alarms on normal boards (7 test + 6 holdout) had their anomaly-map peak off the board,
  on the background cloth (dust, lint). Re-checked after v2 with a board region that includes the header
  pins: still 13/13 outside the board (`reports/pcb1_comparison/`).
- The image score is the maximum over the whole frame, so background clutter sets the score of many
  normal images. The threshold is an upper quantile of those scores, so background clutter can set the
  threshold and push subtle on-board defects below it.

Because this hypothesis came from the PCB1 test results, PCB1 cannot confirm it.

## 3. Why only one modelling change

- **Attribution.** With one change, any difference between v1 and v2 can be attributed to it. v1 and v2
  share bit-identical fitted Gaussian states, and v2's full-frame score reproduces v1's score exactly on
  every image (`comparison.json` → `pairing_checks`).
- **The diagnosis pointed at aggregation, not at the features.** Missed defects mostly peaked on the
  defect itself; false alarms peaked off the board. Changing where the maximum is taken tests that
  diagnosis directly.
- **Avoiding test-set tuning.** Several simultaneous changes chosen after looking at PCB1 results would
  amount to fitting the method to one test set.

Considered and deferred: more FIT images, per-region covariance shrinkage, colour-aware features for
"melt" defects, a focus guard. None was tested.

## 4. How v2 was evaluated independently

1. Protocol committed (`ad8a386`) before any PCB2 file was read: hypothesis, single change, ROI rule,
   split, primary endpoint (paired Δ recall at each model's P95 threshold, exact McNemar), FPR
   non-inferiority margin of 5 points, outcome rule.
2. PCB2 (another VisA PCB category, same acquisition setup) prepared from the official archive. Split
   600 FIT / 150 CALIB / 151 HOLDOUT / 100 + 100 test, audited.
3. **One protocol amendment, before any PCB2 model existed (`9b69850`).** FIT-image inspection showed
   that the pre-registered ROI rule, built from the colour board mask, would exclude the header pins,
   the header base and part of an IC (`reports/pcb2_inspection/roi_rule_A0_vs_A1.png`). The rule was
   amended (A1) to add structure fixed to the board, using FIT images only. The amendment is disclosed in
   the protocol. It was a design defect caught by visual QC; the numeric coverage gate could not detect
   it.
4. v1 and v2 fitted on the same 600 FIT images and calibrated independently on the same 150 CALIB
   images (v1 P95 = 55.03, v2 P95 = 51.20). Frozen state committed (`862c28f`). Evaluation refuses to run
   unless that state is committed and unchanged.
5. Single confirmatory evaluation of both models (`2b775be`, evaluated at HEAD `862c28f`). Nothing was
   changed afterwards.

## 5. Results

### A. PCB1 v1 (original) — see §1

### B and C. PCB2, official test (100 normal + 100 defective), each at its own frozen P95 threshold

| | PCB2 v1 | PCB2 v2 |
|---|---|---|
| Recall | 80/100 [70.8, 87.3]% | 80/100 [70.8, 87.3]% |
| FPR, test normals | 13/100 [7.1, 21.2]% | 3/100 [0.6, 8.5]% |
| AUROC | 0.927 [0.889, 0.956] | 0.979 [0.962, 0.991] |
| TP / FN / FP / TN | 80 / 20 / 13 / 87 | 80 / 20 / 3 / 97 |
| Median latency | 194 ms | 224 ms |

### D. Paired change, v2 − v1 (same images)

| Endpoint | Δ [95% CI] | Discordant images (v2 only / v1 only) | Exact McNemar p |
|---|---|---|---|
| **Recall (primary)** | **+0 pts [−3, +3]** | 1 / 1 | **1.00** |
| FPR, test normals | −10 pts [−17, −3] | 2 / 12 | 0.013 |
| FPR, holdout normals | −1.3 pts [−4.6, +2.0] | 2 / 4 | 0.69 |
| AUROC | +0.052 [+0.028, +0.082] (paired bootstrap) | — | — |

**Pre-registered outcome: no meaningful effect demonstrated** on the primary endpoint. The FPR
non-inferiority condition holds.

Secondary findings (not the decision endpoint; no multiplicity adjustment): the official-test
false-alarm rate fell from 13% to 3% and AUROC rose by 0.05. The holdout false-alarm rate did not change
significantly. The premise held on PCB2: 17 of 21 v1 false alarms (12/13 test, 5/8 holdout) peaked
outside the board ROI.

**Why recall did not move (descriptive).** The ROI can raise recall at a calibrated threshold only if
background peaks set the upper tail of the calibration normals. On PCB1 all 10 highest calibration
scores were background-driven, and the threshold would have dropped from 63.7 to 44.2. On PCB2 only 4 of
the 10 highest were; the threshold dropped from 55.0 to 51.2, and only one test anomaly scored in that
gap. PCB2's remaining misses are mostly small defects (recall about 56% in the smallest-defect tercile),
which ROI aggregation cannot fix.

### E. Supplementary normal holdout (separate from the official test)

PCB1 v1: 6/154 (3.9%). PCB2 v1: 8/151 (5.3%). PCB2 v2: 6/151 (4.0%). All within the 10% target.

### Retrospective, not evidence: v2 on PCB1

Computed after the PCB2 run, on the data that produced the hypothesis: recall 88/100 (vs 72), test FPR
6/100, holdout 3/154, AUROC 0.974. This shows the change does what it was designed to do on the data it
was designed from. It is not independent confirmation.

### Robustness at the frozen thresholds (PCB2; 100 anomalies + 251 normals; paired per image)

| Perturbation | v1 Δ recall / Δ FPR | v2 Δ recall / Δ FPR |
|---|---|---|
| Darker, lower contrast (×0.9) | +5 / **+22.3** pts | +9 / **+25.1** pts |
| Brighter, higher contrast (×1.1) | +4 / +0.0 pts | +5 / +1.6 pts |
| Gaussian blur σ = 1.5 px | −5 / +0.4 pts | −3 / −0.8 pts |
| Shift (+40, +25) px, before registration | −1 / −2.8 pts | +2 / −2.0 pts |

On PCB2, a 10% darkening raises false alarms by more than 20 points for both models. Normal
brightness on PCB2 is lower and more variable (FIT range 83–95 grey levels) than on PCB1. The proposed
robustness criterion (FPR rise ≤ 10 points) is not met. The blur sensitivity seen on PCB1 (+4.3 points
FPR) did not reappear on PCB2. The shift test exercises registration plus detection, not the scoring
model alone.

## 6. Against the original acceptance criteria (unchanged)

| | PCB2 v1 | PCB2 v2 |
|---|---|---|
| Recall ≥ 90% | not met (80%) | not met (80%) |
| FPR ≤ 10% test / holdout | not met (13%) / met | met / met |
| AUROC ≥ 0.90 | met | met |
| Latency ≤ 1 s | met | met |
| Robustness (Δ recall ≥ −10, Δ FPR ≤ +10 pts) | not met | not met |
| NO-GO triggers | none (recall exactly at the 80% floor) | none (recall exactly at the 80% floor) |
| Rule-based verdict | INCONCLUSIVE | INCONCLUSIVE |

The NO-GO rules were kept as written. They trigger on a robustness *recall* collapse but not on a
false-alarm surge. A future revision should probably add one. That would be a change for the next
evaluation, not for this one.

## 7. What these results do and do not show

**They show:**
- On a second, held-back PCB category, scoring only the board region cut test-set false alarms
  substantially (13 → 3 per 100) and improved ranking (AUROC +0.05) at no cost in recall, under a
  pre-registered, single-run protocol.
- The pre-registered hypothesis about **recall** was not confirmed on PCB2. The mechanism analysis
  explains why: it depends on how much the background drives the calibration tail.
- The prototype meets AUROC, false-alarm and latency targets on PCB2, but not the 90% recall target or
  the lighting-robustness criterion.

**They do not show:**
- That the method or the ROI change works on PCB3/PCB4, other products, other cameras, or a real line.
- Real-line prevalence, costs, savings or ROI.
- Readiness for a pilot. Recall and lighting robustness are below the proposed criteria, and those
  criteria themselves await confirmation by a real user.
- Independent confirmation from PCB1 v2 numbers. They are retrospective.

## 8. Recommended next steps (not done)

1. Lighting robustness: photometric normalisation or a brightness-range guard, re-validated prospectively.
2. Small-defect recall: higher-resolution features or more FIT images, tested on fresh data.
3. Confirm targets, costs, prevalence and must-catch defect types with a real user before any pilot.
