"""Five-fold construction and leakage audit for exploratory E1 (2026-10-10, after the experiment).

Uses only committed, path-free files, so it runs in a clean checkout without the dataset or the weights:
results/split.csv, results/scores_*.csv, audit/fold_manifest.csv (exported from the list files actually used for
training), audit/yolo_labels.csv and audit/dataset_identity.json.

    python exploratory/yolo_pcb4/audit/check_leakage.py

Writes exploratory/yolo_pcb4/audit/leakage_audit.json (per-run membership, counts and every check). Exit code 1 if a
check fails.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE  # optional output directory (tests)
RES = HERE.parent / "results"
CV = [f"cv_fold{k}" for k in range(1, 6)]
UNSEEN = {"unseen_missing": "missing", "unseen_extra": "extra"}


def read(p: Path) -> list[dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    split = read(RES / "split.csv")
    role = {r["image"]: r["role"] for r in split}
    types = {r["image"]: r["types"] for r in split}
    nbox = Counter(r["image"] for r in read(HERE / "yolo_labels.csv"))
    man = read(HERE / "fold_manifest.csv")
    ident = json.loads((HERE / "dataset_identity.json").read_text(encoding="utf-8"))
    part: dict[str, dict[str, str]] = defaultdict(dict)
    for r in man:
        assert r["image"] not in part[r["run"]], ("image listed twice in a run", r["run"], r["image"])
        part[r["run"]][r["image"]] = r["partition"]
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    calib = {i for i, r in role.items() if r == "calib"}
    test_n = {i for i, r in role.items() if r == "test_n"}
    fit = {i for i, r in role.items() if r == "fit"}
    defects = {i for i, r in role.items() if r.startswith("fold")}
    check("split: 1105 unique image ids", len(role) == len(split) == 1105, str(len(role)))
    check("split: no exact duplicate content (SHA-256) anywhere", len({r["sha256"] for r in split}) == len(split))
    check("dataset: no cross-role near-duplicate (dHash <= 10/256)", not ident["near_duplicate_cross_role_pairs"],
          f"nearest cross-role distance {ident['nearest_cross_role_hamming']['min']}")
    check("dataset: label files re-derive from masks", not ident["label_files_rederived_from_masks_mismatches"])
    check("dataset: every defect has >= 1 box", all(nbox[i] >= 1 for i in defects))

    runs = {}
    bg_sets = []
    for run in CV + list(UNSEEN):
        p = part[run]
        g = lambda name: {i for i, v in p.items() if v == name}  # noqa: E731
        tr, va, te = g("train"), g("internal_val"), g("test")
        bg = {i for i in tr | va if role[i] == "fit"}
        bg_sets.append(frozenset(bg))
        check(f"{run}: train/internal_val/test pairwise disjoint", not (tr & va or tr & te or va & te))
        check(f"{run}: CALIB and TEST-N never in train or internal_val", not ((tr | va) & (calib | test_n)))
        check(f"{run}: background normals come only from FIT", bg <= fit and all(role[i] in ("fit",) or i in defects for i in tr | va))
        if run in CV:
            k = int(run[-1])
            fold = {i for i in defects if role[i] == f"fold{k}"}
            check(f"{run}: test = exactly the 20 fold-{k} defects", te == fold, str(len(te)))
            check(f"{run}: train+internal_val defects = the other 80", (tr | va) & defects == defects - fold)
        else:
            t = UNSEEN[run]
            check(f"{run}: test = single-label '{t}' images", te == {i for i in defects if types[i] == t}, str(len(te)))
            check(f"{run}: no image carrying '{t}' in train or internal_val (checkpoint selection blind to it)",
                  not any(t in types[i].split(",") for i in (tr | va) & defects))
            excl = {i for i in defects if t in types[i].split(",") and types[i] != t}
            check(f"{run}: multi-label images with '{t}' excluded from train, val and test", not (excl & (tr | va | te)), str(len(excl)))
        runs[run] = {
            "train": {"defect_ids": sorted(tr & defects), "normal_ids": sorted(tr - defects)},
            "internal_val": {"defect_ids": sorted(va & defects), "normal_ids": sorted(va - defects)},
            "test": {"defect_ids": sorted(te)},
            "calib_normal_ids": "results/split.csv role=calib (identical 150 images in every run)",
            "test_normal_ids": "results/split.csv role=test_n (identical 255 images in every run)",
            "counts": {"train_defects": len(tr & defects), "train_normals": len(tr - defects),
                       "val_defects": len(va & defects), "val_normals": len(va - defects), "test_defects": len(te),
                       "calib_normals": len(calib), "test_normals": len(test_n),
                       "boxes_train": sum(nbox[i] for i in tr), "boxes_val": sum(nbox[i] for i in va),
                       "boxes_test": sum(nbox[i] for i in te),
                       "multi_type_train": sum("," in types[i] for i in tr & defects),
                       "multi_type_val": sum("," in types[i] for i in va & defects),
                       "multi_type_test": sum("," in types[i] for i in te)}}
    check("all runs: the same 160 background normals", len(set(bg_sets)) == 1 and len(bg_sets[0]) == 160)
    cover = Counter(i for run in CV for i, v in part[run].items() if v == "test")
    check("cv: every defect is a test image in exactly one fold", set(cover) == defects and set(cover.values()) == {1})

    # score files agree with the manifest; one out-of-fold prediction per defect per architecture
    for arch in ("yolov8s", "yolo11s"):
        seen = Counter()
        for run in CV:
            rows = read(RES / f"scores_{arch}_{run}.csv")
            st = defaultdict(set)
            for r in rows:
                st[r["set"]].add(r["image"])
            check(f"{arch} {run}: scored test set = manifest test set",
                  st["test"] == {i for i, v in part[run].items() if v == "test"})
            check(f"{arch} {run}: scored CALIB = the 150 CALIB, TEST-N = the 255 TEST-N, no FIT image",
                  st["calib"] == calib and st["test_n"] == test_n and not (set().union(*st.values()) & fit))
            seen.update(st["test"])
        check(f"{arch}: exactly one out-of-fold prediction per defect (100)", set(seen) == defects and set(seen.values()) == {1})
    for run, t in UNSEEN.items():
        rows = read(RES / f"scores_yolo11s_{run}.csv")
        check(f"yolo11s {run}: scored test set = manifest test set",
              {r["image"] for r in rows if r["set"] == "test"} == {i for i, v in part[run].items() if v == "test"})
    for m in ("m", "m_reg"):
        rows = read(RES / f"scores_{m}.csv")
        check(f"{m}: scores every non-FIT image once, no FIT image",
              sorted(r["image"] for r in rows) == sorted(set(role) - fit))

    out = {"note": "audit after the experiment; committed files only",
           "reuse": {"calib_normals": "the same 150 images set every threshold (5 folds x 2 architectures + 2 unseen runs + M + M-reg)",
                     "test_normals": "the same 255 images are scored by every model; YOLO pooled FPR = 1,275 evaluations of 255 images",
                     "background_normals": "the same 160 FIT normals in every YOLO run (128 train + 32 internal val, split differs by run seed)"},
           "label_use": {"permitted_training_labels": "box labels (from masks) of train and internal_val defects of each run",
                         "held_out_evaluation_labels": "image labels and masks of each run's test defects: used only for recall, "
                                                       "mAP50 (ultralytics val on the test list) and the post-hoc localisation check",
                         "thresholds": "CALIB normal scores only (no defect label)",
                         "checkpoint": "ultralytics best.pt by its fitness on internal_val (no test image)"},
           "runs": runs, "checks": [{"check": c, "ok": ok, "detail": d} for c, ok, d in checks],
           "all_ok": all(ok for _, ok, _ in checks)}
    (OUTDIR / "leakage_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for c, ok, d in checks:
        print("PASS" if ok else "FAIL", c, d)
    print(f"{sum(ok for _, ok, _ in checks)}/{len(checks)} checks pass")
    if not out["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
