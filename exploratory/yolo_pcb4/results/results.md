# Exploratory E1 results (PCB4) — descriptive, see PROTOCOL.md §8 before reading

| Method | Recall (100 defects) | 95% CI | FPR test normals (mean; fold range) | AUROC | Latency median | mAP50 |
|---|---|---|---|---|---|---|
| m | 60% (60/100) | 50%–70% | 4.3%  | 0.941 | 32 ms | — |
| m_reg | 50% (50/100) | 40%–60% | 7.5%  | 0.876 | 135 ms | — |
| YOLOv8s | 98% (98/100) | 93%–100% | 4.0% 2.0%–5.5% | 0.996 | 15 ms | 0.32 |
| YOLO11s | 95% (95/100) | 89%–98% | 3.1% 1.6%–3.5% | 0.994 | 14 ms | 0.29 |

Recall by defect size (mask-area terciles):

- ref_m: 47% (16/34) · 61% (20/33) · 73% (24/33)
- ref_m_reg: 44% (15/34) · 45% (15/33) · 61% (20/33)
- YOLOv8s: 94% (32/34) · 100% (33/33) · 100% (33/33)
- YOLO11s: 91% (31/34) · 94% (31/33) · 100% (33/33)

Paired vs M (normal-only, no registration), descriptive bootstrap 95% CI:

- m_reg: Δrecall −10 pts (−20 to +0); ΔFPR +3 pts (+0 to +7); caught only by this 9, only by M 19
- YOLOv8s: Δrecall +38 pts (+29 to +48); ΔFPR +0 pts (−3 to +3); caught only by this 38, only by M 0
- YOLO11s: Δrecall +35 pts (+24 to +45); ΔFPR −1 pts (−4 to +1); caught only by this 38, only by M 3

Unseen defect type (YOLO11s), single-label images of the held-out type:

- missing: type seen (5-fold) 19/19; type not in training 11/19; normal-only M 10/19
- extra: type seen (5-fold) 10/12; type not in training 9/12; normal-only M 7/12

Missed defects per method: see results.json → methods.<name>.missed. Failure contact sheet: fig_failures.jpg.
