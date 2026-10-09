# Requirements baseline — PCB image review support

**Status: proposed.** Drafted with AI assistance for this proof of concept. No manufacturing engineer
has reviewed or approved it. Every operational number here is a placeholder chosen to make the prototype
testable; §4 lists what must be confirmed by a real user.

The document separates six kinds of statement:

| § | Kind | Source |
|---|---|---|
| 1 | Hypothetical manufacturing need | assumption |
| 2 | Proposed acceptance criteria | proposal; fixed before any test result |
| 3 | Go / no-go rules | proposal; fixed before any test result |
| 4 | Questions for a real user | open |
| 5 | Measured benchmark results | `reports/` (generated) |
| 6 | Out of scope | decision |

---

## 1. Hypothetical need

**User (assumed):** a process or quality engineer at an automated optical inspection station that
photographs one printed circuit board per image with a fixed overhead camera.

**Decision supported:** for each image, *review* or *no review*. The tool flags images for a person to
look at, with a heat map of where the board looks abnormal. It never passes, scraps or dispositions a
board on its own.

**Why a one-class approach:** defects are rare and varied, so the model learns only from defect-free
boards and flags whatever differs from them.

**Outputs per image:** an anomaly score, a REVIEW / NO REVIEW recommendation at a threshold frozen in
advance, and an anomaly heat map.

## 2. Proposed acceptance criteria

Fixed before v1 testing; unchanged in v2 (`config/acceptance_criteria.yaml`). Evaluated on the official
VisA test split of each board type (100 normal + 100 defective) at the pre-registered primary threshold.
Targets are point estimates; 95% confidence intervals are always reported next to them.

| ID | Criterion | Target | Why |
|---|---|---|---|
| P1 | Recall (defective boards flagged) | ≥ 90% | missed defects assumed costlier than reviews |
| P2 | False-alarm rate on normal boards, official test **and** a separate normal holdout | ≤ 10% | proxy for wasted review time |
| P3 | Image-level AUROC | ≥ 0.90 | threshold-free ranking quality |
| P4 | Median end-to-end latency, one workstation GPU | ≤ 1 s | placeholder; real takt time unknown |
| P5 | Under mild brightness/contrast (±10%), blur (σ 1.5 px), shift (40/25 px), at the same threshold | recall drop ≤ 10 pts and FPR rise ≤ 10 pts | sensitivity to acquisition drift |

Validity requirements (must always hold): the fit, calibration, holdout and test images are disjoint
(checked by path and content hash); no test label influences any setting or threshold (settings and
thresholds are hashed and committed before evaluation); every reported number is regenerated from saved
per-image scores; results are reproducible with fixed seeds.

Precision is not a criterion: the benchmark is 50% defective, which says nothing about a real line.

## 3. Go / no-go rules for proposing a pilot

Cost model: with defect prevalence π, cost of a missed defect C_FN and of an unnecessary review C_FP,
the normalised expected cost of a threshold is `C = k·(1 − recall) + FPR` with `k = π/(1−π) · C_FN/C_FP`.
"Review nothing" costs k and "review everything" costs 1. A threshold is only worth using where it beats
both.

- **GO to pilot:** validity holds; P1–P5 met; and, once a user supplies k, the primary threshold beats
  both reference policies.
- **NO-GO** (any one): recall < 80%; FPR > 20% on either normal set; a robustness recall drop > 20 pts;
  results depend on test-set tuning or cannot be reproduced.
- **INCONCLUSIVE** otherwise: collect more evidence before deciding.

The rules are applied by code (`src/ivad/acceptance.py`). Known gap, found in v2 and left unchanged for
this evaluation: a large false-alarm rise under perturbation is not a NO-GO trigger.

## 4. Questions a real user must answer before any pilot

1. **Costs:** what does an escaped defect cost compared with one unnecessary review? (sets k)
2. **Prevalence and capacity:** what fraction of boards is defective, and how many flagged boards per
   shift can reviewers absorb? (sets the FPR tolerance, P2)
3. **Criticality:** which defect types and sizes must never be missed? (per-type recall, resolution)
4. **Acquisition stability:** how stable are fixture, lighting and focus, and how often do boards or
   fixtures change? (P5, re-calibration cadence)
5. **Integration:** takt time, where the recommendation is consumed, and who owns the threshold. (P4)

None of these has been answered. All targets in §2 are placeholders until they are.

## 5. Measured results (benchmark only)

Generated tables: `README.md` (results block), `reports/*/results.md`,
`reports/pcb2_comparison/comparison.md`. Interpretation: `docs/v2_decision_record.md`.

| Run | Recall | FPR test | FPR holdout | AUROC | Status |
|---|---|---|---|---|---|
| PCB1 v1 (original) | 72% | 7% | 3.9% | 0.938 | NO-GO (recall below 80%) |
| PCB2 v1 | 80% | 13% | 5.3% | 0.927 | not met: recall, test FPR, robustness |
| PCB2 v2 (board region) | 80% | 3% | 4.0% | 0.979 | not met: recall, robustness |

These are public-benchmark measurements under lab conditions. They are not evidence of performance on a
real line.

## 6. Out of scope

Other products or datasets beyond the two VisA PCB categories used; defect classification;
pixel-accurate segmentation; automatic disposition; line, MES or cloud integration; real-factory data;
claims about yield, prevalence, cost savings or ROI.

Factory readiness would additionally need a shadow-mode pilot on line images with ground truth,
user-agreed criteria, a re-calibration and drift-monitoring procedure, and sign-off by quality
engineering.
