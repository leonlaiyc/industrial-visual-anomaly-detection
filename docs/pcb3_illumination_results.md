# PCB3 illumination-correction experiment — results and interpretation

Protocol: `docs/illumination_experiment_protocol.md` (pre-registered `d7f5e81`; Addendum B1 `5ba6aca`;
Addendum B2 `d8b5f1e`). Freeze: `eb4e7a9`. Single planned evaluation: `115c54b` (both `metrics.json`
record `git_head_at_evaluation = eb4e7a9`). Paired analysis: `reports/pcb3_illum_comparison/`
(`9c9f7c5`, analysis code `7dbd302`). Everything below is generated from those recorded files; nothing was
re-scored, re-calibrated or tuned after test scoring. **Post-evaluation analysis-code repair:** the
paired-analysis program needed one fix after test scoring (`7dbd302`); it changed no score, threshold,
prediction, primary statistic or decision rule — full incident record in
[`pcb3_post_evaluation_repair.md`](pcb3_post_evaluation_repair.md), deviation log in
[`illumination_experiment_deviations.md`](illumination_experiment_deviations.md). Every number below is
independently recomputed from the raw score files by `verification/recompute_pcb3.py`
([`reports/pcb3_verification/recompute.md`](../reports/pcb3_verification/recompute.md), 90/90 agree).

## 0. Two decisions, kept separate

| | Decision | Basis |
|---|---|---|
| **A. Pre-registered experimental outcome** | **ROBUSTNESS IMPROVEMENT SUPPORTED** — exactly as computed by the frozen rule; **weakly informative** on PCB3 | protocol §8 rule, computed by code (§1 below) |
| **B. Engineering adoption decision** | **DO NOT ADOPT the illumination candidate as the preferred pipeline at this stage** | the full set of conditions and the original acceptance criteria (§2, §3 E) |

**A, narrowly.** The rule asks one question: under a synthetic 10% global darkening (G1), does the
candidate's false-alarm rise on the same 256 unseen normals fall below the baseline's, without losing clean
recall or clean false alarms beyond fixed margins? It does (DiD −4.7 pts, CI −8.6 to −0.8). That is all the
finding says. The baseline's own G1 rise (+4.3 pts) was below the pre-registered 5-pt informativeness line,
so PCB3 barely exhibited the failure the correction targets, and G1 is a global change that a global
correction undoes almost by construction. It is not evidence of robustness to real lighting.

**B.** The candidate improves global-brightness behaviour (G1, and B1 brightening +15.2 → +2.7 pts) but
raises the false-alarm rate under a mild blur by **+14.5 pts** (9 → 46 of 256 normals), against +0.4 pts for
the baseline. Under the original P5 criterion (no perturbation may raise false alarms by more than 10 pts)
that is an unacceptable robustness trade-off: the candidate swaps one failing condition for another.

**"Candidate rejected" does not mean "baseline approved for production."** The baseline also fails P5
(brightening +15.2 pts), and neither variant has any real manufacturing end-user validation, factory pilot
or measured operational return. Both are INCONCLUSIVE against the unchanged criteria:

| Original criterion (unchanged) | Baseline | Candidate |
|---|---|---|
| P1 recall ≥ 90% (100 official-test anomalies) | met: 90.0% (90/100) | met: 91.0% (91/100) |
| P2 FPR ≤ 10%, official-test normals | met: 3.0% (3/101) | met: 3.0% (3/101) |
| P2 FPR ≤ 10%, supplementary holdout normals | met: 5.2% (8/155) | met: 3.9% (6/155) |
| P3 AUROC ≥ 0.90 | met: 0.982 | met: 0.983 |
| P4 median latency ≤ 1 s | met: 187 ms | met: 292 ms |
| P5 robustness (G1, B1, blur, shift; FPR rise ≤ 10 pts, recall drop ≤ 10 pts) | **not met**: B1 +15.2 pts | **not met**: blur +14.5 pts |
| NO-GO triggers | none | none |
| **Verdict** | **INCONCLUSIVE** | **INCONCLUSIVE** |

P1 is met on PCB3 at a point estimate of exactly 90/100 (95% CI 82.4%–95.1%); this does not establish
that a 90% recall would hold on another board type or on a real line.



## 1. Pre-registered outcome

**ROBUSTNESS IMPROVEMENT SUPPORTED — primary endpoint weakly informative on PCB3.**

| Quantity (pooled 256 unseen normals: 101 official test + 155 holdout) | Baseline | Candidate |
|---|---|---|
| False-alarm rate, clean (P0) | 4.3% (11/256) | 3.5% (9/256) |
| False-alarm rate, global darkening (G1) | 8.6% (22/256) | 3.1% (8/256) |
| Rise under G1 | +4.3 pts | −0.4 pts |
| **DiD (candidate − baseline), paired bootstrap 95% CI** | | **−4.7 pts [−8.6, −0.8]** |
| Exact McNemar, G1 flags candidate vs baseline | | 17 baseline-only vs 3 candidate-only, p = 0.003 |

Clean gates (both pass):

| Clean metric | Baseline | Candidate | Δ [95% CI] | McNemar p |
|---|---|---|---|---|
| Recall, 100 test anomalies | 90.0% | 91.0% | +1.0 pts [−2.0, +4.0] | 1.00 |
| FPR, 101 official-test normals | 3.0% | 3.0% | +0.0 pts [−3.0, +3.0] | 1.00 |
| FPR, 155 holdout normals | 5.2% | 3.9% | −1.3 pts [−3.2, +0.0] | 0.50 |
| AUROC | 0.982 | 0.983 | +0.001 [−0.003, +0.007] | — |

**Weakly informative (protocol §8):** the baseline's own G1 false-alarm rise is +4.3 pts, below the 5-pt
line. PCB3 did **not** reproduce the PCB2 darkening failure (+22.3 / +25.1 pts on PCB2). The rule's
outcome is reported as computed, with this qualification attached, as the protocol requires.

## 2. All conditions (nothing omitted)

False-alarm rate on the 256 pooled normals and recall on the 100 test anomalies, each variant at its own
frozen P95 threshold (baseline 46.60, candidate 47.72):

| Condition | Role | FPR baseline | FPR candidate | Rise baseline | Rise candidate | DiD [95% CI] | Recall Δ base / cand |
|---|---|---|---|---|---|---|---|
| P0 clean | reference | 4.3% | 3.5% | — | — | — | — |
| G1 darker ×0.90/×0.90 | **primary** | 8.6% | 3.1% | +4.3 | −0.4 | −4.7 [−8.6, −0.8] | 0 / 0 |
| B1 brighter ×1.10/×1.10 | secondary | 19.5% | 6.2% | +15.2 | +2.7 | −12.5 [−17.6, −7.4] | +1 / −1 |
| N1 L→R ramp 1.0→0.8 | secondary (limitation) | 6.2% | 5.9% | +2.0 | +2.3 | +0.4 [−2.7, +3.9] | 0 / −3 |
| Blur σ = 1.5 px | acceptance only (P5) | 4.7% | **18.0%** | +0.4 | **+14.5** | not an endpoint | −2 / −4 |
| Shift (+40, +25) px | acceptance only (P5) | 4.3% | 4.3% | +0.0 | +0.8 | not an endpoint | +1 / −2 |

Exact counts behind the table (absolute false alarms of 256 pooled normals; recall hits of 100 anomalies;
the change columns above are differences of these counts ÷ 256 or ÷ 100):

| Condition | FP baseline | FP candidate | TP baseline | TP candidate |
|---|---|---|---|---|
| P0 clean | 11/256 | 9/256 | 90/100 | 91/100 |
| G1 | 22/256 | 8/256 | 90/100 | 91/100 |
| B1 | 50/256 | 16/256 | 91/100 | 90/100 |
| N1 | 16/256 | 15/256 | 90/100 | 88/100 |
| Blur σ = 1.5 px | 12/256 | 46/256 | 88/100 | 87/100 |
| Shift (+40, +25) px | 11/256 | 11/256 | 91/100 | 89/100 |

The 256 pooled normals are the 101 official-test normals plus the 155 supplementary holdout normals;
official-test and holdout FPRs are reported separately in §1 and §0.

ROI fail-safe on the clean official-test and holdout images: 0 for both variants. Under G1 the baseline
hit the fail-safe on 3 of 356 images and the candidate on none (consistent with the FIT-only diagnostic,
8/600 vs 0/600).

**Blur (unplanned observation, reported because no failing condition may be hidden).** The correction
made the model markedly more sensitive to a mild blur: +14.5 pts false alarms (9 → 46 of 256) versus
+0.4 pts for the baseline. Blur was in the protocol only to score the original P5 criterion, so this is not
a comparative endpoint and no test was pre-registered for it. *Post-hoc, FIT-only mechanism check (50 FIT
images, changes nothing):* blur lowers the global luminance SD by about 7% (median 33.1 → 30.9), so the
correction applies a ≈ 7% contrast stretch (median gain 1.00 → 1.07) on top of the blur. That is a plausible
cause, not an established one.

## 3. Interpretation, kept separate

**A. Global-darkening correction.** Under the pre-registered rule the candidate's false-alarm rise under G1
is smaller than the baseline's (DiD −4.7 pts, CI excludes 0). But (i) the baseline's rise was only +4.3 pts,
so there was little to correct, and (ii) G1 is a global affine change that the correction can undo almost
by construction (protocol §3, risk 1). This shows the mechanism works and did not damage clean
performance; it is **not** evidence of robustness to real lighting changes.

**B. Registration / segmentation stability.** The FIT-only diagnostic (Addendum B2) showed that without
correction a 10% darkening moved the estimated board pose by a median 10.7 px (8/600 ROI fail-safe events),
and with correction by 0.2 px (0 events). On test/holdout under G1 the baseline used the fail-safe on 3
images, the candidate on none. Part or all of the G1 and B1 improvement may therefore come from more stable
board segmentation and registration rather than from features closer to the normal model. The experiment
was not designed to separate these, and no causal decomposition is claimed.

**C. Anomaly-score behaviour.** Clean ranking and operating point are essentially unchanged (AUROC +0.001;
recall +1 pt; FPR equal on test, −1.3 pts on holdout). Under brightening (B1) the candidate's false-alarm rise
fell from +15.2 to +2.7 pts (secondary, DiD −12.5 pts). Under blur the candidate is clearly worse (+14.5 vs
+0.4 pts). Net: the correction trades sensitivity to global brightness for a new sensitivity to blur.
Small/low-contrast subgroups: no sensitivity flag (small tercile 26 → 27 detected, low-contrast 29 → 28).

**D. Non-uniform illumination (N1).** No improvement: DiD +0.4 pts [−2.7, +3.9]; candidate recall −3 pts.
Both variants' N1 rises were small on PCB3 (+2.0 / +2.3 pts), so the predeclared limitation rule (> 10 pts)
is not triggered, but the expectation stated in advance — a global correction does not address a spatial
lighting gradient — is consistent with the data.

**E. Readiness for a manufacturing pilot.** **Not ready.** Against the unchanged acceptance criteria both
variants are **INCONCLUSIVE**: clean recall, FPR, AUROC and latency targets are met on PCB3, but the
original P5 robustness criterion fails for both (baseline: brightening +15.2 pts; candidate: blur
+14.5 pts). The correction adds ≈ 105 ms median latency (187 → 292 ms; correction alone 71 ms on CPU). All
perturbations are synthetic; no real camera, lamp, line or site was involved. PCB3 also does not replace the
PCB2 result, where recall (80%) remained below target.

## 4. Deviations and post-freeze events (complete list)

1. **Addendum B1 commit order** (disclosed in Addendum B2.1): the FIT-only board-mask adaptation was
   decided before fitting but committed about five minutes after both variants were fitted.
2. **Addendum B2 added outside §10's reasons** (disclosed in B2 itself): a disclosure and FIT-only
   diagnostics, no adaptation.
3. **Post-evaluation analysis-code repair** (`7dbd302`, 17:14:22, after test scoring; full record in
   [`pcb3_post_evaluation_repair.md`](pcb3_post_evaluation_repair.md)): `illum-compare` refused **twice**
   with unchanged code (17:09 and 17:10; the second refusal is the file
   `reports/pcb3_illum_comparison/attempt1_refused.log`) before producing any output, because the input
   validation added in B2 required `label == "anomaly"` in VisA's `image_anno.csv`, which actually stores
   the defect type. The fix changes only that check, in `defect_descriptors`, which feeds only the
   descriptive §8 C subgroup table. No score, threshold, configuration, scoring code, primary statistic or
   decision rule changed; evaluation and robustness were not re-run; the recorded scores (`115c54b`) are the
   single planned evaluation. Summary test metrics had been seen on the evaluation console before the
   repair. Because the validation and its synthetic fixture shared the wrong premise, the fixture was
   corrected and a regression test reads the real annotation file. An independent recomputation
   reproduces every recorded value, including the subgroup table from the original masks (90/90).
4. **Post-hoc material (labelled):** the blur mechanism check in §2. It changes no decision.

No configuration, threshold, correction parameter, perturbation definition or score-affecting code changed
after the freeze (`git diff eb4e7a9 9c9f7c5 -- config manifests src/ivad/{pipeline,photometric,align,roi,robustness,features,gaussian,calibration}.py`
is empty). No PCB4 data was used. The test was not repeated.

## 5. What would be investigated next (recommended, not implemented)

- Why the correction amplifies blur sensitivity (contrast stretch vs registration vs features), on FIT data.
- Whether a lighting **check** (flag images whose global brightness/contrast is outside the FIT range)
  is safer operationally than a lighting **correction**.
- Real lighting variation on site images in shadow mode, which no synthetic condition here represents.
