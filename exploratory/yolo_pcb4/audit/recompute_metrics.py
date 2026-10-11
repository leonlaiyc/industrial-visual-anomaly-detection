"""Independent recomputation of the exploratory E1 metrics from the SAVED per-image scores (audit, 2026-10-10).

Deliberately does not import the project's `ivad` package or `analyze.py`: the threshold rule, confusion matrices,
intervals and AUROC are re-implemented here with numpy/scipy only, then compared with the recorded
results/results.json. No model is run and no image is opened. Runs in a clean checkout:

    python exploratory/yolo_pcb4/audit/recompute_metrics.py

Writes exploratory/yolo_pcb4/audit/recomputed_metrics.json and recomputed_metrics.md. Exit code 1 if any recorded
number fails to recompute.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE  # optional output directory (tests)
RES = HERE.parent / "results"
ALPHA, LEVEL = 0.05, 0.95
B, SEED = 2000, 20261011  # audit bootstrap; a different seed from the experiment's (7) on purpose
ARCHS = ("yolov8s", "yolo11s")
REFS = ("m", "m_reg")
NAMES = {"m": "Normal-only M (no registration)", "m_reg": "Normal-only M-reg (registration)",
         "yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}


def read(name: str) -> list[dict[str, str]]:
    with open(RES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def thr_rank(n: int, alpha: float) -> int:
    """Split-conformal rank: the ceil((n+1)(1-alpha))-th smallest calibration score."""
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    assert 1 <= k <= n
    return k


def threshold(cal: np.ndarray) -> float:
    return float(np.sort(cal)[thr_rank(len(cal), ALPHA) - 1])


def cp(x: int, n: int) -> list[float]:
    a = 1 - LEVEL
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x))
    return [lo, hi]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cm(tp: int, fn: int, fp: int, tn: int) -> dict:
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall": tp / (tp + fn) if tp + fn else None, "recall_ci95_cp": cp(tp, tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None, "fpr_ci95_cp": cp(fp, fp + tn) if fp + tn else None}


def main() -> None:
    split = read("split.csv")
    role = {r["image"]: r["role"] for r in split}
    types = {r["image"]: r["types"] for r in split}
    defects = sorted(i for i, r in role.items() if r.startswith("fold"))
    calib = sorted(i for i, r in role.items() if r == "calib")
    test_n = sorted(i for i, r in role.items() if r == "test_n")
    D, C, N = len(defects), len(calib), len(test_n)
    di, ci, ni = ({x: j for j, x in enumerate(s)} for s in (defects, calib, test_n))

    # score matrices: per method, per fold (references: one model, replicated over the 5 "folds")
    S: dict[str, dict] = {}
    checks: list[dict] = []
    for m in REFS:
        rows = read(f"scores_{m}.csv")
        s = {r["image"]: float(r["score"]) for r in rows}
        assert not any(role[r["image"]] == "fit" for r in rows), "FIT image scored"
        S[m] = {"cal": np.array([[s[i] for i in calib]] * 5), "neg": np.array([[s[i] for i in test_n]] * 5),
                "pos": np.array([s[i] for i in defects]), "pos_fold": np.array([int(role[i][4:]) for i in defects])}
    for a in ARCHS:
        cal = np.zeros((5, C)); neg = np.zeros((5, N)); pos = np.full(D, np.nan); posf = np.zeros(D, int)
        zero_box = []
        for k in range(1, 6):
            rows = read(f"scores_{a}_cv_fold{k}.csv")
            cnt = Counter(r["set"] for r in rows)
            for r in rows:
                v = float(r["score"])
                if r["set"] == "calib":
                    assert role[r["image"]] == "calib"; cal[k - 1, ci[r["image"]]] = v
                elif r["set"] == "test_n":
                    assert role[r["image"]] == "test_n"; neg[k - 1, ni[r["image"]]] = v
                else:
                    assert r["set"] == "test" and role[r["image"]] == f"fold{k}", (a, k, r["image"])
                    assert np.isnan(pos[di[r["image"]]]), "defect scored twice out of fold"
                    pos[di[r["image"]]] = v; posf[di[r["image"]]] = k
            zero_box.append({"fold": k, "rows": dict(cnt),
                             "score_eq_0_(no_box)": {st: sum(float(r["score"]) == 0.0 for r in rows if r["set"] == st)
                                                     for st in ("test", "calib", "test_n")}})
        assert not np.isnan(pos).any() and sorted(Counter(posf).values()) == [20] * 5
        S[a] = {"cal": cal, "neg": neg, "pos": pos, "pos_fold": posf, "zero_box": zero_box}

    # point estimates
    out: dict = {"source": "results/*.csv (saved per-image scores) and results/split.csv; independent code",
                 "threshold_rule": f"flag iff score > the {thr_rank(C, ALPHA)}-th smallest of {C} CALIB scores "
                                   f"(split conformal, alpha {ALPHA}); YOLO image score = max box confidence, 0 if no box",
                 "denominators": {"defects_unique": D, "calib_normals_unique": C, "test_normals_unique": N,
                                  "yolo_test_normal_evaluations_per_arch": 5 * N,
                                  "note": "the SAME 255 TEST-N images and the SAME 150 CALIB images are scored by all 5 "
                                          "fold models of each architecture; pooled YOLO FPR has 1,275 evaluations of 255 "
                                          "unique images, which are NOT independent observations"},
                 "methods": {}}
    flags_pos, flags_neg = {}, {}
    for m, s in S.items():
        thr = [threshold(s["cal"][f]) for f in range(5)]
        fp_pos = np.array([s["pos"][j] > thr[s["pos_fold"][j] - 1] for j in range(D)])
        fn_neg = np.array([[s["neg"][f, j] > thr[f] for j in range(N)] for f in range(5)])
        flags_pos[m], flags_neg[m] = fp_pos, fn_neg
        is_ref = m in REFS
        per_fold = []
        for f in range(5):
            sel = s["pos_fold"] == f + 1
            tp = int(fp_pos[sel].sum()); fp = int(fn_neg[f].sum())
            e = cm(tp, int(sel.sum()) - tp, fp, N - fp)
            e.update({"fold": f + 1, "threshold": thr[f],
                      "calib_ties_at_threshold": int((s["cal"][f] == thr[f]).sum()),
                      "auroc_fold_defects_vs_test_n": auroc(s["pos"][sel], s["neg"][f])})
            per_fold.append(e)
        tp = int(fp_pos.sum())
        flagged_per_normal = fn_neg.sum(0)  # 0..5 folds flagged this unique normal
        entry = {"name": NAMES[m],
                 "pooled_out_of_fold_defects": cm(tp, D - tp, 0, 0) | {"fpr": None, "fpr_ci95_cp": None},
                 "per_fold": None if is_ref else per_fold}
        if is_ref:
            fp = int(fn_neg[0].sum())
            entry["single_model"] = cm(tp, D - tp, fp, N - fp) | {"threshold": thr[0],
                                                                   "auroc_all_defects_vs_test_n": auroc(s["pos"], s["neg"][0])}
        else:
            fpe = int(fn_neg.sum())
            entry["test_normal_evaluations_pooled"] = {
                "FP_evaluations": fpe, "TN_evaluations": 5 * N - fpe, "evaluations": 5 * N, "unique_images": N,
                "fpr_mean_of_folds": float(fn_neg.mean()),
                "fpr_fold_range": [float(min(fn_neg.mean(1))), float(max(fn_neg.mean(1)))],
                "unique_normals_flagged_by_k_folds": {str(k): int((flagged_per_normal == k).sum()) for k in range(6)},
                "unique_normals_flagged_at_least_once": int((flagged_per_normal > 0).sum()),
                "naive_cp_on_1275_INVALID_shown_only_for_contrast": cp(fpe, 5 * N)}
            entry["auroc_mean_of_fold_aurocs"] = float(np.mean([e["auroc_fold_defects_vs_test_n"] for e in per_fold]))
            entry["thresholds"] = thr
            entry["calib_scores_eq_0"] = [int((s["cal"][f] == 0).sum()) for f in range(5)]
            entry["zero_box_counts"] = s["zero_box"]
        entry["conformal_marginal_fpr_per_threshold"] = {
            "distribution": f"Beta({C + 1 - thr_rank(C, ALPHA)}, {thr_rank(C, ALPHA)})",
            "expected": (C + 1 - thr_rank(C, ALPHA)) / (C + 1),
            "central_95": [float(v) for v in stats.beta.ppf([0.025, 0.975], C + 1 - thr_rank(C, ALPHA), thr_rank(C, ALPHA))]}
        out["methods"][m] = entry

    # cluster bootstrap over UNIQUE images: resample defects, CALIB normals and TEST-N normals; recompute every fold
    # threshold from the resampled CALIB set, so threshold-calibration noise and normal-image reuse are both carried
    rng = np.random.default_rng(SEED)
    keys = list(S)
    pairs = (("yolov8s", "yolo11s"), ("yolov8s", "m"), ("yolo11s", "m"))
    fixed = {m: np.array([threshold(S[m]["cal"][f]) for f in range(5)]) for m in keys}
    q = lambda v: [float(np.quantile(v, (1 - LEVEL) / 2)), float(np.quantile(v, (1 + LEVEL) / 2))]  # noqa: E731
    out["cluster_bootstrap"] = {"B": B, "seed": SEED}
    for variant in ("thresholds_fixed", "thresholds_recalibrated"):
        bs = {m: {"recall": [], "fpr": []} for m in keys}
        pair = {xy: {"d_recall": [], "d_fpr": []} for xy in pairs}
        for _ in range(B):
            dd = rng.integers(0, D, D); cc = rng.integers(0, C, C); nn = rng.integers(0, N, N)
            vals = {}
            for m in keys:
                s = S[m]
                thr = fixed[m] if variant == "thresholds_fixed" else np.array([threshold(s["cal"][f, cc]) for f in range(5)])
                rec = float((s["pos"][dd] > thr[s["pos_fold"][dd] - 1]).mean())
                fpr = float((s["neg"][:, nn] > thr[:, None]).mean())
                vals[m] = (rec, fpr)
                bs[m]["recall"].append(rec); bs[m]["fpr"].append(fpr)
            for (x, y), d in pair.items():
                d["d_recall"].append(vals[x][0] - vals[y][0]); d["d_fpr"].append(vals[x][1] - vals[y][1])
        out["cluster_bootstrap"][variant] = {
            "ci95": {m: {"recall": q(bs[m]["recall"]), "fpr": q(bs[m]["fpr"])} for m in keys},
            "paired_differences_ci95": {f"{x}_minus_{y}": {k: q(v) for k, v in d.items()} for (x, y), d in pair.items()}}
    out["cluster_bootstrap"]["resampled_units"] = (
        "unique defect images and unique TEST-N normals (all five fold evaluations of a resampled normal move together); "
        "'thresholds_fixed' keeps the thresholds actually used, 'thresholds_recalibrated' also resamples the 150 unique "
        "CALIB normals and recomputes every fold threshold, adding calibration noise at the 144th-of-150 rank")

    # exact McNemar on the 100 shared defects
    mc = {}
    for x, y in (("yolov8s", "yolo11s"), ("yolov8s", "m"), ("yolo11s", "m"), ("m_reg", "m")):
        b = int((flags_pos[x] & ~flags_pos[y]).sum()); c = int((~flags_pos[x] & flags_pos[y]).sum())
        p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        mc[f"{x}_vs_{y}"] = {"only_first": b, "only_second": c, "exact_mcnemar_p_two_sided": p}
    out["mcnemar_defects"] = mc
    out["observed_paired"] = {f"{x}_minus_{y}": {
        "d_recall": float(flags_pos[x].mean() - flags_pos[y].mean()),
        "d_fpr": float(flags_neg[x].mean() - flags_neg[y].mean())} for (x, y) in pairs}

    # E7 unseen type (YOLO11s): recompute and check the training/test construction from the saved scores
    seen = flags_pos["yolo11s"]
    unseen = {}
    for t in ("missing", "extra"):
        rows = read(f"scores_yolo11s_unseen_{t}.csv")
        cal = np.array([float(r["score"]) for r in rows if r["set"] == "calib"])
        thr = threshold(cal)
        test = [r for r in rows if r["set"] == "test"]
        ids = [r["image"] for r in test]
        assert all(types[i] == t for i in ids)
        multi_with_t = sorted(i for i in defects if t in types[i].split(",") and types[i] != t)
        neg = np.array([float(r["score"]) for r in rows if r["set"] == "test_n"])
        hit = int(sum(float(r["score"]) > thr for r in test))
        unseen[t] = {"n_test_single_label": len(ids), "n_multi_label_with_type_excluded_from_both": len(multi_with_t),
                     "unseen_hits": hit, "unseen_recall_ci95_cp": cp(hit, len(ids)),
                     "seen_5fold_hits_same_images": int(sum(seen[di[i]] for i in ids)),
                     "m_hits_same_images": int(sum(flags_pos["m"][di[i]] for i in ids)),
                     "threshold": thr, "fpr_test_n": float((neg > thr).mean()), "fp_test_n": int((neg > thr).sum())}
        unseen[t]["exact_mcnemar_seen_vs_unseen"] = None  # different models on the same images; counts reported only
    out["unseen_type"] = unseen

    # latency records as saved (only what the experiment kept)
    lat = {}
    for m in ("m", "m_reg", "yolov8s", "yolo11s"):
        lat[m] = json.loads((RES / f"latency_{m}.json").read_text(encoding="utf-8"))
    out["latency_as_recorded"] = lat

    # compare with the recorded results.json
    rec = json.loads((RES / "results.json").read_text(encoding="utf-8"))

    def chk(name, got, want, tol=1e-9):
        ok = (got == want) if isinstance(want, (int, str)) else abs(float(got) - float(want)) <= tol
        checks.append({"check": name, "recomputed": got, "recorded": want, "ok": bool(ok)})

    for m in keys:
        r = rec["methods"][m]
        chk(f"{m}.recall_x", int(flags_pos[m].sum()), r["recall_x"])
        chk(f"{m}.recall_ci.lo", cp(int(flags_pos[m].sum()), D)[0], r["recall_ci"][0])
        chk(f"{m}.recall_ci.hi", cp(int(flags_pos[m].sum()), D)[1], r["recall_ci"][1])
        chk(f"{m}.fpr_mean", float(flags_neg[m].mean()), r["fpr_mean"])
        if m in ARCHS:
            for f in range(5):
                chk(f"{m}.fold{f + 1}.fpr", float(flags_neg[m][f].mean()), r["fpr_folds"][f])
                chk(f"{m}.fold{f + 1}.auroc", out["methods"][m]["per_fold"][f]["auroc_fold_defects_vs_test_n"], r["auroc_folds"][f], 1e-12)
        else:
            chk(f"{m}.auroc", out["methods"][m]["single_model"]["auroc_all_defects_vs_test_n"], r["auroc_folds"][0], 1e-12)
        chk(f"{m}.missed", sorted(defects[j] for j in range(D) if not flags_pos[m][j]).__repr__(),
            sorted(x["image"] for x in r["missed"]).__repr__())
    for t, u in unseen.items():
        ru = rec["unseen_type"][t]
        chk(f"unseen.{t}.n", u["n_test_single_label"], ru["n"])
        chk(f"unseen.{t}.unseen_hits", u["unseen_hits"], ru["unseen_hits"])
        chk(f"unseen.{t}.seen_hits", u["seen_5fold_hits_same_images"], ru["seen_hits"])
        chk(f"unseen.{t}.m_hits", u["m_hits_same_images"], ru["m_hits"])
    out["checks_vs_recorded_results_json"] = checks
    out["all_checks_ok"] = all(c["ok"] for c in checks)
    (OUTDIR / "recomputed_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_md(out)
    sys.stdout.reconfigure(encoding="utf-8")
    print((OUTDIR / "recomputed_metrics.md").read_text(encoding="utf-8"))
    print(f"checks: {sum(c['ok'] for c in checks)}/{len(checks)} ok")
    if not out["all_checks_ok"]:
        for c in checks:
            if not c["ok"]:
                print("MISMATCH", c)
        raise SystemExit(1)


def p(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def ci(v: list[float], nd: int = 1) -> str:
    return f"{p(v[0], nd)}–{p(v[1], nd)}"


def write_md(o: dict) -> None:
    L = ["# E1 audit: metrics recomputed from saved per-image scores", "",
         "Generated by `audit/recompute_metrics.py` (independent of `analyze.py` and of the `ivad` package). "
         "Descriptive, exploratory study; see `../PROTOCOL.md` §8.", "",
         f"Threshold rule: {o['threshold_rule']}.", "",
         "## Pooled out-of-fold confusion matrices", "",
         "| Method | TP | FN | Recall (CP 95%) | FP | TN | FPR | FPR denominator | Bootstrap 95%, thresholds fixed (recall; FPR) "
         "| Bootstrap 95%, thresholds recalibrated (recall; FPR) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, e in o["methods"].items():
        d = e["pooled_out_of_fold_defects"]
        bc = o["cluster_bootstrap"]["thresholds_fixed"]["ci95"][m]
        br = o["cluster_bootstrap"]["thresholds_recalibrated"]["ci95"][m]
        if "single_model" in e:
            s = e["single_model"]
            L.append(f"| {e['name']} | {s['TP']} | {s['FN']} | {p(s['recall'], 0)} ({ci(s['recall_ci95_cp'], 0)}) | {s['FP']} | "
                     f"{s['TN']} | {p(s['fpr'])} | 255 unique images, 1 model | {ci(bc['recall'], 0)}; {ci(bc['fpr'])} | {ci(br['recall'], 0)}; {ci(br['fpr'])} |")
        else:
            t = e["test_normal_evaluations_pooled"]
            L.append(f"| {e['name']} | {d['TP']} | {d['FN']} | {p(d['recall'], 0)} ({ci(d['recall_ci95_cp'], 0)}) | "
                     f"{t['FP_evaluations']} | {t['TN_evaluations']} | {p(t['fpr_mean_of_folds'])} | "
                     f"1,275 evaluations = 255 unique images × 5 fold models | {ci(bc['recall'], 0)}; {ci(bc['fpr'])} | "
                     f"{ci(br['recall'], 0)}; {ci(br['fpr'])} |")
    L += ["", "Bootstrap (cluster, by unique image): " + o["cluster_bootstrap"]["resampled_units"] + ". The reuse of the "
          "same normal images across the five fold models is therefore carried into every interval. "
          "A Clopper–Pearson interval on 1,275 pooled evaluations would treat repeated images as independent and is "
          "therefore not used.", "", "## Per fold (YOLO)", "",
          "| Model | Fold | Threshold | TP/20 | FP/255 | Recall | FPR (CP 95%) | AUROC |", "|---|---|---|---|---|---|---|---|"]
    for m in ARCHS:
        for f in o["methods"][m]["per_fold"]:
            L.append(f"| {NAMES[m]} | {f['fold']} | {f['threshold']:.6f} | {f['TP']} | {f['FP']} | {p(f['recall'], 0)} | "
                     f"{p(f['fpr'])} ({ci(f['fpr_ci95_cp'])}) | {f['auroc_fold_defects_vs_test_n']:.3f} |")
    L += ["", "## Reuse of the 255 test normals (YOLO)", ""]
    for m in ARCHS:
        t = o["methods"][m]["test_normal_evaluations_pooled"]
        L.append(f"- {NAMES[m]}: unique normals flagged by k of 5 fold models: {t['unique_normals_flagged_by_k_folds']}; "
                 f"flagged at least once: {t['unique_normals_flagged_at_least_once']}/255.")
    L += ["", "## Paired comparisons (same images)", "", "| Comparison | Observed Δrecall | 95% CI (fixed; recalibrated) | "
          "Observed ΔFPR | 95% CI (fixed; recalibrated) |", "|---|---|---|---|---|"]
    for k, ob in o["observed_paired"].items():
        v = o["cluster_bootstrap"]["thresholds_fixed"]["paired_differences_ci95"][k]
        w = o["cluster_bootstrap"]["thresholds_recalibrated"]["paired_differences_ci95"][k]
        L.append(f"| {k} | {100 * ob['d_recall']:+.0f} pts | {100 * v['d_recall'][0]:+.0f} to {100 * v['d_recall'][1]:+.0f}; "
                 f"{100 * w['d_recall'][0]:+.0f} to {100 * w['d_recall'][1]:+.0f} | {100 * ob['d_fpr']:+.1f} pts | "
                 f"{100 * v['d_fpr'][0]:+.1f} to {100 * v['d_fpr'][1]:+.1f}; {100 * w['d_fpr'][0]:+.1f} to {100 * w['d_fpr'][1]:+.1f} |")
    L += ["", "Exact McNemar on the 100 defects (discordant pairs):", ""]
    for k, v in o["mcnemar_defects"].items():
        L.append(f"- {k}: only first {v['only_first']}, only second {v['only_second']}, p = {v['exact_mcnemar_p_two_sided']:.3g}")
    L += ["", "## Unseen defect type (YOLO11s)", "", "| Held-out type | n (single-label) | Type seen (5-fold) | Type not in training "
          "| Normal-only M | Unseen-run FPR on TEST-N | Multi-label images with the type (excluded from train and test) |",
          "|---|---|---|---|---|---|---|"]
    for t, u in o["unseen_type"].items():
        n = u["n_test_single_label"]
        L.append(f"| {t} | {n} | {u['seen_5fold_hits_same_images']}/{n} | {u['unseen_hits']}/{n} "
                 f"({ci(u['unseen_recall_ci95_cp'], 0)}) | {u['m_hits_same_images']}/{n} | {p(u['fpr_test_n'])} "
                 f"({u['fp_test_n']}/255) | {u['n_multi_label_with_type_excluded_from_both']} |")
    L += ["", "## Zero-box images (YOLO score 0)", ""]
    for m in ARCHS:
        L.append(f"- {NAMES[m]}: CALIB scores equal to 0 per fold {o['methods'][m]['calib_scores_eq_0']}; "
                 + "; ".join(f"fold {z['fold']}: {z['score_eq_0_(no_box)']}" for z in o["methods"][m]["zero_box_counts"]))
    L += ["", f"## Recorded-vs-recomputed checks: {sum(c['ok'] for c in o['checks_vs_recorded_results_json'])}/"
          f"{len(o['checks_vs_recorded_results_json'])} match", ""]
    (OUTDIR / "recomputed_metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
