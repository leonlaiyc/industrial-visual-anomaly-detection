# PCB3 illumination experiment — post-test deviation log

The protocol ([`illumination_experiment_protocol.md`](illumination_experiment_protocol.md)) states that
nothing in it may change after the freeze commit (`eb4e7a9`). It is therefore left byte-identical, and every
event after the freeze is logged here instead. This file is **not** an amendment: it changes no rule,
endpoint, threshold or decision; it records what happened. Written 2026-10-09 after the evaluation.

| # | When (+08:00) | Event | Effect on results | Where documented |
|---|---|---|---|---|
| 1 | before test (B1) | Board-mask adaptation committed ~5 min after fitting instead of before | none on test scoring; disclosed pre-test | protocol Addendum B2.1 |
| 2 | before test (B2) | Addendum added for a disclosure and FIT-only diagnostics, outside §10's listed reasons | none | protocol Addendum B2 |
| 3 | 17:09 and 17:10 | Pre-registered `ivad illum-compare` refused twice (unchanged code); the second refusal was captured as `attempt1_refused.log` | no output produced | [`pcb3_post_evaluation_repair.md`](pcb3_post_evaluation_repair.md) §2–3 |
| 4 | 17:14:22 | **Analysis-code repair after test scoring** (`7dbd302`): GT-mask label check in `defect_descriptors` accepted VisA defect-type labels | no score, threshold, prediction, primary statistic or decision rule changed; enables the descriptive §8 C table only; independently recomputed 90/90 | [`pcb3_post_evaluation_repair.md`](pcb3_post_evaluation_repair.md) |
| 5 | 16:59–17:14 | Summary test metrics visible on the evaluation console before the repair | none possible through the repair (§8 of the record) | repair record §8 |
| 6 | — | §9 bullet 5 ("serious implementation defect → run invalidated") judged not applicable: the defect was in the analysis input validation, not in the evaluated pipeline | run not invalidated, not repeated | repair record §12 D5 — **for the auditor to confirm** |
| 7 | 17:15:17 | Post-hoc blur mechanism check on 50 FIT images | labelled post hoc; changes no decision | [`pcb3_illumination_results.md`](pcb3_illumination_results.md) §2 |
| 8 | after 17:15 | Reporting-code changes: `ivad check` C5/C7 gates, website and PDF builders, independent verifier `verification/recompute_pcb3.py` | reporting only | README, release QA record |
| 9 | after 17:15 | Engineering adoption decision (do not adopt the candidate) written **in addition to** the pre-registered rule outcome; the rule outcome is reported unchanged | none on the pre-registered outcome | [`pcb3_illumination_results.md`](pcb3_illumination_results.md) §1 |
| 10 | 2026-10-09, after the audit snapshot | AI-assisted technical cross-check of the audit snapshot (`3ea038a`) — not a human audit; scope limits in `reports/project_milestones.json`: thresholds, confusion matrices, recall, FPR, holdout false alarms, AUROC, perturbation false alarms and the G1 difference-in-differences recomputed from the raw score files and found to agree; repair `7dbd302` judged a non-score-affecting analysis-input validation repair (answers row 6 provisionally: run not invalidated, not repeated; the project owner's confirmation is pending); this disclosure must remain | none | `reports/project_milestones.json` (`reviews`) |
| 11 | 2026-10-09, after row 10 | Owner-handoff delivery refinements: one-page executive PDF added, two-page PDF kept as the extended brief, website wording (accuracy, Traditional Chinese copyedit), `ivad check` C5/C8 extended to both PDFs | reporting only; no score, threshold, prediction or decision changed | README → *Website and PDF* |
| 12 | 2026-10-10 | Review-provenance correction after an AI-assisted review of the owner-handoff snapshot (`17c690e`): website, PDF and record wording that could imply an independent human engineering review replaced by what was verified — (A) programmatic recomputation, (B) AI-assisted technical review, (C) project-owner review pending; no independent human audit has occurred | reporting text only; no score, threshold, prediction or decision changed | `reports/project_milestones.json` (`verification`, `reviews`) |
| 13 | 2026-10-11 (Asia/Taipei) | D5 **ACCEPTED BY PROJECT OWNER**: Leon's explicit confirmation in this conversation accepts the disclosed non-score-affecting descriptive anomaly-label validation repair and retains the PCB3 primary evaluation. Rows 6 and 10 remain the original chronology; their provisional D5 disposition is superseded by this decision. Post-test timing, all deviations and technical limitations remain disclosed. No signature or third-party human audit is claimed. Overall project review is still PENDING; publication, GitHub changes and email delivery are NOT AUTHORIZED | governance status only; no experiment artifact or historical result changed | repair record §12, dated D5 owner-decision addendum |

Unchanged since the freeze (verified): configs, split manifest, thresholds, correction parameters,
perturbation definitions, fitted models, and every score-affecting source file
(`git diff eb4e7a9 HEAD -- config manifests src/ivad/{pipeline,photometric,align,roi,robustness,features,gaussian,calibration,data,config,metrics}.py`
is empty). The PCB3 test and holdout images were scored once. No PCB4 data was used.
