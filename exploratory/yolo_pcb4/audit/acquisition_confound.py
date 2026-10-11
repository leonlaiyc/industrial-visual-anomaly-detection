"""Audit (2026-10-10, after the experiment): could an acquisition cue separate PCB4 defect images from normal images
without looking at the defect? (On PCB1 an orientation/capture confound existed; see the project's failure analysis.)

For every defect image and every TEST-N normal, computes defect-blind global features: grey-level mean and std, board
mask area fraction, board centroid (x, y), board bounding-box aspect, and principal-axis angle of the board mask (the
project's existing `ivad.align.foreground_mask` rule, unchanged). Reports the AUROC of each single feature for
defect-vs-normal (0.5 = no separation; values far from 0.5 in either direction = a cue). Reads images only; no model.

    .venv/Scripts/python.exe exploratory/yolo_pcb4/audit/acquisition_confound.py

Writes exploratory/yolo_pcb4/audit/acquisition_confound.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

HERE = Path(__file__).resolve().parent
E1 = HERE.parent
ROOT = E1.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivad.align import foreground_mask  # noqa: E402

DATA = ROOT / "data" / "visa"


def auroc(pos, neg) -> float:
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def feats(p: Path) -> dict[str, float]:
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    g = np.asarray(rgb.convert("L"), dtype=np.float32)
    m = foreground_mask(rgb)
    ys, xs = np.nonzero(m)
    h, w = m.shape
    cov = np.cov(np.stack([xs / w, ys / h]))
    ev, evec = np.linalg.eigh(cov)
    major = evec[:, int(np.argmax(ev))]
    ang = float(np.degrees(np.arctan2(major[1], major[0])) % 180)
    return {"grey_mean": float(g.mean()), "grey_std": float(g.std()), "board_area_frac": float(m.mean()),
            "board_cx": float(xs.mean() / w), "board_cy": float(ys.mean() / h),
            "board_bbox_aspect": float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)),
            "board_axis_angle_deg": min(ang, 180 - ang)}


def main() -> None:
    split = list(csv.DictReader(open(E1 / "results" / "split.csv", newline="", encoding="utf-8")))
    pos = [feats(DATA / r["image"]) for r in split if r["label"] == "anomaly"]
    neg = [feats(DATA / r["image"]) for r in split if r["role"] == "test_n"]
    out = {"note": "defect-blind global features; AUROC defect (n=100) vs TEST-N normal (n=255); 0.5 = no separation",
           "features": {}}
    for k in pos[0]:
        a, b = np.array([x[k] for x in pos]), np.array([x[k] for x in neg])
        au = auroc(a, b)
        out["features"][k] = {"auroc": au, "separation_|auroc-0.5|": abs(au - 0.5),
                              "defect_median": float(np.median(a)), "normal_median": float(np.median(b))}
    out["max_separation"] = max(v["separation_|auroc-0.5|"] for v in out["features"].values())

    # does a model's SAVED score follow the cue where no defect exists? Spearman rho between grey mean and the score
    # over the 255 TEST-N normals (YOLO: mean of the five fold-model scores; also per fold), and over the 100 defects.
    res = E1 / "results"
    n_ids = [r["image"] for r in split if r["role"] == "test_n"]
    a_ids = [r["image"] for r in split if r["label"] == "anomaly"]
    gm_n = np.array([x["grey_mean"] for x in neg]); gm_a = np.array([x["grey_mean"] for x in pos])
    out["score_vs_grey_mean_spearman"] = {}
    for m in ("m", "m_reg"):
        s_ = {r["image"]: float(r["score"]) for r in csv.DictReader(open(res / f"scores_{m}.csv", encoding="utf-8"))}
        out["score_vs_grey_mean_spearman"][m] = {
            "test_n": float(stats.spearmanr(gm_n, [s_[i] for i in n_ids])[0]),
            "defects": float(stats.spearmanr(gm_a, [s_[i] for i in a_ids])[0])}
    for a in ("yolov8s", "yolo11s"):
        per, dsc = [], {}
        for k in range(1, 6):
            rows = list(csv.DictReader(open(res / f"scores_{a}_cv_fold{k}.csv", encoding="utf-8")))
            s_ = {r["image"]: float(r["score"]) for r in rows if r["set"] == "test_n"}
            per.append([s_[i] for i in n_ids])
            dsc.update({r["image"]: float(r["score"]) for r in rows if r["set"] == "test"})
        per = np.array(per)
        out["score_vs_grey_mean_spearman"][a] = {
            "test_n_mean_of_folds": float(stats.spearmanr(gm_n, per.mean(0))[0]),
            "test_n_per_fold": [float(stats.spearmanr(gm_n, per[f])[0]) for f in range(5)],
            "defects_out_of_fold": float(stats.spearmanr(gm_a, [dsc[i] for i in a_ids])[0])}
    out["score_vs_grey_mean_note"] = ("negative rho on TEST-N = darker normal images receive higher scores, i.e. the model "
                                      "responds to the acquisition cue where no defect exists")

    # sensitivity from SAVED scores and the thresholds actually used (no rescoring): false-alarm rate on TEST-N normals
    # reweighted to the defect images' grey-mean distribution (5 bins at pooled quintiles; weight = defect share /
    # normal share of the bin), and on the darkest third of TEST-N. Post hoc, audit-time, descriptive.
    def thr(cal):
        c = np.sort(np.asarray(cal)); return float(c[int(np.ceil((len(c) + 1) * 0.95 - 1e-12)) - 1])
    edges = np.quantile(np.concatenate([gm_a, gm_n]), [0.2, 0.4, 0.6, 0.8])
    ba, bn = np.digitize(gm_a, edges), np.digitize(gm_n, edges)
    wbin = {b: (ba == b).mean() / max((bn == b).mean(), 1e-12) for b in range(5)}
    wt = np.array([wbin[b] for b in bn])
    dark = gm_n <= np.quantile(gm_n, 1 / 3)
    sens = {"normals_per_bin": [int((bn == b).sum()) for b in range(5)], "defects_per_bin": [int((ba == b).sum()) for b in range(5)],
            "darkest_third_n": int(dark.sum()), "methods": {}}
    cal_ids = [r["image"] for r in split if r["role"] == "calib"]
    for m in ("m", "m_reg"):
        s_ = {r["image"]: float(r["score"]) for r in csv.DictReader(open(res / f"scores_{m}.csv", encoding="utf-8"))}
        t = thr([s_[i] for i in cal_ids])
        f = np.array([s_[i] > t for i in n_ids], float)
        sens["methods"][m] = {"fpr_all": float(f.mean()), "fpr_brightness_matched": float((f * wt).sum() / wt.sum()),
                              "fpr_darkest_third": float(f[dark].mean())}
    for a in ("yolov8s", "yolo11s"):
        fs = []
        for k in range(1, 6):
            rows = list(csv.DictReader(open(res / f"scores_{a}_cv_fold{k}.csv", encoding="utf-8")))
            t = thr([float(r["score"]) for r in rows if r["set"] == "calib"])
            s_ = {r["image"]: float(r["score"]) for r in rows if r["set"] == "test_n"}
            fs.append([s_[i] > t for i in n_ids])
        f = np.array(fs, float).mean(0)
        sens["methods"][a] = {"fpr_all": float(f.mean()), "fpr_brightness_matched": float((f * wt).sum() / wt.sum()),
                              "fpr_darkest_third": float(f[dark].mean())}
    # brightness-matched CALIBRATION: the same bin weights applied to the 150 CALIB normals, threshold = weighted 95th
    # percentile (flag iff score > it); recall on the 100 defects (YOLO out of fold) and matched FPR on TEST-N.
    cal_feats = [feats(DATA / i) for i in cal_ids]
    wc = np.array([wbin[b] for b in np.digitize([x["grey_mean"] for x in cal_feats], edges)])

    def wthr(cal):
        o = np.argsort(cal); c = np.asarray(cal)[o]; cw = np.cumsum(wc[o]) / wc.sum()
        return float(c[int(np.searchsorted(cw, 0.95))])
    for m in ("m", "m_reg"):
        s_ = {r["image"]: float(r["score"]) for r in csv.DictReader(open(res / f"scores_{m}.csv", encoding="utf-8"))}
        t = wthr([s_[i] for i in cal_ids])
        f = np.array([s_[i] > t for i in n_ids], float)
        sens["methods"][m]["matched_calibration"] = {"threshold": t, "recall": float(np.mean([s_[i] > t for i in a_ids])),
                                                     "fpr_brightness_matched": float((f * wt).sum() / wt.sum()), "fpr_all": float(f.mean())}
    for a in ("yolov8s", "yolo11s"):
        hits, fs, ts = 0, [], []
        for k in range(1, 6):
            rows = list(csv.DictReader(open(res / f"scores_{a}_cv_fold{k}.csv", encoding="utf-8")))
            c_ = {r["image"]: float(r["score"]) for r in rows if r["set"] == "calib"}
            t = wthr([c_[i] for i in cal_ids]); ts.append(t)
            hits += sum(float(r["score"]) > t for r in rows if r["set"] == "test")
            s_ = {r["image"]: float(r["score"]) for r in rows if r["set"] == "test_n"}
            fs.append([s_[i] > t for i in n_ids])
        f = np.array(fs, float).mean(0)
        sens["methods"][a]["matched_calibration"] = {"thresholds": ts, "recall": hits / 100,
                                                     "fpr_brightness_matched": float((f * wt).sum() / wt.sum()), "fpr_all": float(f.mean())}
    sens["matched_calibration_rule"] = ("CALIB normals weighted by the defect/normal grey-mean bin ratio; threshold = weighted "
                                        "95th percentile; effective CALIB sample is small (brightest bin weight 0), so noisy")
    out["brightness_matched_fpr_sensitivity"] = sens
    (HERE / "acquisition_confound.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
