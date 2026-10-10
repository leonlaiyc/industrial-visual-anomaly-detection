"""Exploratory E1 analysis (PROTOCOL.md §6-§7). Project environment, after score_reference.py and train_yolo.py:

    .venv/Scripts/python.exe exploratory/yolo_pcb4/analyze.py

Reads only the per-image score files in results/, the split and the masks. Writes results/results.json (the
`site` block feeds site/facts.py), results/results.md and results/fig_failures.jpg.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivad.calibration import conformal_threshold  # noqa: E402
from ivad.metrics import auroc, clopper_pearson  # noqa: E402

OUT = HERE / "results"
DATA = ROOT / "data" / "visa"
ALPHA, N_BOOT, BOOT_SEED, LEVEL = 0.05, 2000, 7, 0.95
ARCHS = {"yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}
UNSEEN = ("missing", "extra")
MINUS = "−"


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def pts(x: float) -> str:
    v = round(100 * x)
    return f"{0 if v == 0 else v:+d}".replace("-", MINUS)


def read(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(open(OUT / name, newline="", encoding="utf-8")))


def threshold(calib: list[float]) -> float:
    return conformal_threshold(calib, ALPHA)[0]


def boot_ci(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.quantile(b, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
    return float(lo), float(hi)


def main() -> None:
    split = read("split.csv")
    anomalies = [r for r in split if r["label"] == "anomaly"]
    a_ids = [r["image"] for r in anomalies]
    n_ids = [r["image"] for r in split if r["role"] == "test_n"]
    area = {r["image"]: int(r["mask_area_px"]) for r in anomalies}
    types = {r["image"]: r["types"] for r in anomalies}
    q1, q2 = np.quantile([area[i] for i in a_ids], [1 / 3, 2 / 3])
    size_class = {i: ("small" if area[i] <= q1 else "medium" if area[i] <= q2 else "large") for i in a_ids}

    methods: dict[str, dict] = {}
    # normal-only references: one model, one threshold
    for key in ("m", "m_reg"):
        p = OUT / f"scores_{key}.csv"
        if not p.exists():
            continue
        s = {r["image"]: float(r["score"]) for r in read(p.name)}
        thr = threshold([s[r["image"]] for r in split if r["role"] == "calib"])
        a_flag = {i: s[i] > thr for i in a_ids}
        n_flag = np.array([s[i] > thr for i in n_ids], dtype=float)
        lat = json.loads((OUT / f"latency_{key}.json").read_text())["median_ms"]
        methods[key] = {"kind": "normal", "threshold": thr, "a_flag": a_flag, "n_flag_mean": n_flag,
                        "fpr_folds": [float(n_flag.mean())],
                        "auroc_folds": [auroc([1] * len(a_ids) + [0] * len(n_ids), [s[i] for i in a_ids] + [s[i] for i in n_ids])],
                        "latency_ms": lat}
    # YOLO: per-fold thresholds; out-of-fold defect flags
    for arch in ARCHS:
        a_flag, n_flags, fprs, aurocs, maps, thrs = {}, [], [], [], [], []
        for k in range(1, 6):
            rows = read(f"scores_{arch}_cv_fold{k}.csv")
            by = {(r["set"], r["image"]): float(r["score"]) for r in rows}
            thr = threshold([v for (st, _), v in by.items() if st == "calib"])
            thrs.append(thr)
            fold_a = [r["image"] for r in rows if r["set"] == "test"]
            for i in fold_a:
                a_flag[i] = by[("test", i)] > thr
            nf = np.array([by[("test_n", i)] > thr for i in n_ids], dtype=float)
            n_flags.append(nf)
            fprs.append(float(nf.mean()))
            aurocs.append(auroc([1] * len(fold_a) + [0] * len(n_ids),
                                [by[("test", i)] for i in fold_a] + [by[("test_n", i)] for i in n_ids]))
            maps.append(json.loads((OUT / f"map50_{arch}_cv_fold{k}.json").read_text())["map50"])
        assert sorted(a_flag) == sorted(a_ids), "every defect must be scored exactly once out of fold"
        lat = json.loads((OUT / f"latency_{arch}.json").read_text())["median_ms"]
        methods[arch] = {"kind": "boxes", "thresholds": thrs, "a_flag": a_flag, "n_flag_mean": np.mean(n_flags, axis=0),
                         "fpr_folds": fprs, "auroc_folds": aurocs, "map50_folds": maps, "latency_ms": lat}

    # E7: unseen defect type (YOLO11s)
    unseen = {}
    for t in UNSEEN:
        rows = read(f"scores_yolo11s_unseen_{t}.csv")
        by = {(r["set"], r["image"]): float(r["score"]) for r in rows}
        thr = threshold([v for (st, _), v in by.items() if st == "calib"])
        imgs = [r["image"] for r in rows if r["set"] == "test"]
        assert all(types[i] == t for i in imgs)
        hit = sum(by[("test", i)] > thr for i in imgs)
        unseen[t] = {"n": len(imgs), "unseen_hits": int(hit),
                     "seen_hits": int(sum(methods["yolo11s"]["a_flag"][i] for i in imgs)),
                     "m_hits": int(sum(methods["m"]["a_flag"][i] for i in imgs)),
                     "unseen_fpr_test_n": float(np.mean([by[("test_n", i)] > thr for i in n_ids])), "threshold": thr}

    # summaries + paired comparisons vs M (descriptive)
    rng = np.random.default_rng(BOOT_SEED)
    res: dict = {"protocol": "exploratory/yolo_pcb4/PROTOCOL.md", "n_defects": len(a_ids), "n_test_normal": len(n_ids),
                 "size_tercile_px": [float(q1), float(q2)], "methods": {}, "unseen_type": unseen}
    for key, m in methods.items():
        hits = sum(m["a_flag"].values())
        lo, hi = clopper_pearson(hits, len(a_ids), LEVEL)
        by_size = {c: (sum(m["a_flag"][i] for i in a_ids if size_class[i] == c), sum(size_class[i] == c for i in a_ids))
                   for c in ("small", "medium", "large")}
        missed = sorted(i for i in a_ids if not m["a_flag"][i])
        entry = {"kind": m["kind"], "recall": hits / len(a_ids), "recall_x": hits, "recall_ci": [lo, hi],
                 "fpr_mean": float(np.mean(m["fpr_folds"])), "fpr_folds": m["fpr_folds"],
                 "auroc_mean": float(np.mean(m["auroc_folds"])), "auroc_folds": m["auroc_folds"],
                 "latency_median_ms": m["latency_ms"], "recall_by_size": by_size,
                 "missed": [{"image": i, "types": types[i], "mask_area_px": area[i]} for i in missed]}
        if "map50_folds" in m:
            entry["map50_folds"] = m["map50_folds"]
        if key != "m" and "m" in methods:
            ya = np.array([m["a_flag"][i] for i in a_ids], float)
            ma = np.array([methods["m"]["a_flag"][i] for i in a_ids], float)
            dn = m["n_flag_mean"] - methods["m"]["n_flag_mean"]
            entry["paired_vs_m"] = {"d_recall": float((ya - ma).mean()), "d_recall_ci": boot_ci(ya - ma, rng),
                                    "d_fpr": float(dn.mean()), "d_fpr_ci": boot_ci(dn, rng),
                                    "caught_only_by_this": int(((ya == 1) & (ma == 0)).sum()),
                                    "caught_only_by_m": int(((ya == 0) & (ma == 1)).sum())}
        res["methods"][key] = entry

    # site block (formatted strings; facts.py passes it through)
    def row(key: str) -> dict:
        e = res["methods"][key]
        lo, hi = e["recall_ci"]
        r = {"kind": e["kind"], "recall": pct(e["recall"]), "recall_frac": f"{e['recall_x']}/{len(a_ids)}",
             "recall_ci": f"{pct(lo)}–{pct(hi)}", "fpr": pct(e["fpr_mean"], 1),
             "fpr_range": "" if e["kind"] == "normal" else f"{pct(min(e['fpr_folds']), 1)}–{pct(max(e['fpr_folds']), 1)}",
             "auroc": f"{e['auroc_mean']:.3f}", "latency": f"{e['latency_median_ms']:.0f} ms",
             "label_key": {"m": "ref_m", "m_reg": "ref_m_reg"}.get(key, ""), "name": ARCHS.get(key, "")}
        if "paired_vs_m" in e:
            pv = e["paired_vs_m"]
            r["d_recall"] = pts(pv["d_recall"])
            r["d_recall_lo"], r["d_recall_hi"] = pts(pv["d_recall_ci"][0]), pts(pv["d_recall_ci"][1])
            r["d_fpr"] = pts(pv["d_fpr"])
            r["d_fpr_lo"], r["d_fpr_hi"] = pts(pv["d_fpr_ci"][0]), pts(pv["d_fpr_ci"][1])
        if "map50_folds" in e:
            r["map50"] = f"{np.mean(e['map50_folds']):.2f}"
            r["map50_range"] = f"{min(e['map50_folds']):.2f}–{max(e['map50_folds']):.2f}"
        r["small"] = pct(e["recall_by_size"]["small"][0] / e["recall_by_size"]["small"][1])
        return r

    order = [k for k in ("m", "m_reg", "yolov8s", "yolo11s") if k in res["methods"]]
    by = {k: row(k) for k in order}
    res["site"] = {
        "rows": [by[k] for k in order], "by": by,
        "size_rows": [{"label_key": by[k]["label_key"], "name": by[k]["name"],
                       "vals": [f"{pct(x / n)} ({x}/{n})" for x, n in (res['methods'][k]['recall_by_size'][c]
                                                                        for c in ('small', 'medium', 'large'))]}
                      for k in order],
        "unseen_rows": [{"type": t, "n": u["n"], "seen": f"{u['seen_hits']}/{u['n']}", "unseen": f"{u['unseen_hits']}/{u['n']}",
                         "normal_only": f"{u['m_hits']}/{u['n']}"} for t, u in unseen.items()],
        "unseen": {t: {"seen": f"{u['seen_hits']}/{u['n']}", "unseen": f"{u['unseen_hits']}/{u['n']}",
                       "normal_only": f"{u['m_hits']}/{u['n']}"} for t, u in unseen.items()},
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # results.md
    L = ["# Exploratory E1 results (PCB4) — descriptive, see PROTOCOL.md §8 before reading", "",
         "| Method | Recall (100 defects) | 95% CI | FPR test normals (mean; fold range) | AUROC | Latency median | mAP50 |",
         "|---|---|---|---|---|---|---|"]
    for k in order:
        r = by[k]
        L.append(f"| {r['name'] or k} | {r['recall']} ({r['recall_frac']}) | {r['recall_ci']} | {r['fpr']} {r['fpr_range']} | "
                 f"{r['auroc']} | {r['latency']} | {r.get('map50', '—')} |")
    L += ["", "Recall by defect size (mask-area terciles):", ""]
    for s in res["site"]["size_rows"]:
        L.append(f"- {s['name'] or s['label_key']}: " + " · ".join(s["vals"]))
    L += ["", "Paired vs M (normal-only, no registration), descriptive bootstrap 95% CI:", ""]
    for k in order[1:]:
        r = by[k]
        if "d_recall" in r:
            pv = res["methods"][k]["paired_vs_m"]
            L.append(f"- {r['name'] or k}: Δrecall {r['d_recall']} pts ({r['d_recall_lo']} to {r['d_recall_hi']}); "
                     f"ΔFPR {r['d_fpr']} pts ({r['d_fpr_lo']} to {r['d_fpr_hi']}); caught only by this {pv['caught_only_by_this']}, only by M {pv['caught_only_by_m']}")
    L += ["", "Unseen defect type (YOLO11s), single-label images of the held-out type:", ""]
    for t, u in unseen.items():
        L.append(f"- {t}: type seen (5-fold) {u['seen_hits']}/{u['n']}; type not in training {u['unseen_hits']}/{u['n']}; "
                 f"normal-only M {u['m_hits']}/{u['n']}")
    L += ["", "Missed defects per method: see results.json → methods.<name>.missed. Failure contact sheet: fig_failures.jpg."]
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # failure contact sheet: defects missed by YOLO11s (top row) and by M (bottom row), GT outline in cyan
    def thumb(img_id: str, title: str) -> Image.Image:
        im = Image.open(DATA / img_id).convert("RGB")
        mk = next(r["mask"] for r in anomalies if r["image"] == img_id)
        m = Image.open(DATA / mk).convert("L").point(lambda v: 255 if v > 0 else 0).resize(im.size)
        ring = ImageChops.subtract(m.filter(ImageFilter.MaxFilter(15)), m)  # cyan outline around the defect
        im.paste((0, 230, 255), mask=ring)
        im.thumbnail((300, 300))
        canvas = Image.new("RGB", (im.width, im.height + 18), "white")
        canvas.paste(im, (0, 18))
        ImageDraw.Draw(canvas).text((3, 3), title, fill=(20, 20, 20))
        return canvas

    rows_imgs = []
    for k in ("yolo11s", "m"):
        miss = res["methods"][k]["missed"][:6]
        rows_imgs.append([thumb(x["image"], f"{(ARCHS.get(k) or 'normal-only')} miss: {x['types']}, {x['mask_area_px']} px")
                          for x in miss])
    if any(rows_imgs):
        w = max(sum(t.width for t in r) for r in rows_imgs if r)
        h = sum(max((t.height for t in r), default=0) for r in rows_imgs)
        sheet = Image.new("RGB", (w, h), "white")
        y = 0
        for r in rows_imgs:
            x = 0
            for t in r:
                sheet.paste(t, (x, y))
                x += t.width
            y += max((t.height for t in r), default=0)
        sheet.save(OUT / "fig_failures.jpg", quality=85)
    sys.stdout.reconfigure(encoding="utf-8")
    print((OUT / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
