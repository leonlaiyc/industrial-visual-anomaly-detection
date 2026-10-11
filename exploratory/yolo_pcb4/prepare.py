"""Exploratory E1 (PROTOCOL.md §3-§5): extract PCB4, fix the split, audit it, write YOLO datasets and the
normal-only reference config. Run with the project environment:

    .venv/Scripts/python.exe exploratory/yolo_pcb4/prepare.py

Deterministic (seed 20261010). Fails on any audit failure. Writes:
  exploratory/yolo_pcb4/results/split.csv, split_audit.json
  exploratory/yolo_pcb4/config_m.yaml (and config_m_reg.yaml only if the FIT-only check passes)
  artifacts/exploratory_yolo_pcb4/yolo/<run>/{data.yaml, train.txt, val.txt, test.txt}, labels/
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivad.align import foreground_mask  # noqa: E402
from ivad.config import sha256_file  # noqa: E402
from ivad.data import _safe_relative  # noqa: E402

SEED = 20261010
ARCHIVE = ROOT / "data" / "raw" / "VisA_20220922.tar"
DATA = ROOT / "data" / "visa"
OUT = HERE / "results"
ART = ROOT / "artifacts" / "exploratory_yolo_pcb4"
N_FIT, N_CALIB, N_FOLDS, N_BG = 600, 150, 5, 160
UNSEEN_TYPES = ("missing", "extra")
MIN_COMPONENT_PX = 4


def extract() -> list[dict[str, str]]:
    anno = DATA / "pcb4" / "image_anno.csv"
    if not anno.exists():
        n = 0
        with tarfile.open(ARCHIVE, mode="r|*") as tar:
            for m in tar:
                rel = _safe_relative(m.name, "pcb4") if m.isfile() else None
                if rel is None:
                    continue
                dst = DATA / Path(*rel.parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tar.extractfile(m).read())
                n += 1
        print(f"extracted {n} pcb4 files")
    with open(anno, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_split(anno: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(SEED)
    normals = sorted(r["image"] for r in anno if r["label"] == "normal")
    rng.shuffle(normals)
    rows = []
    for i, im in enumerate(normals):
        role = "fit" if i < N_FIT else "calib" if i < N_FIT + N_CALIB else "test_n"
        rows.append({"image": im, "label": "normal", "types": "", "mask": "", "role": role})
    defects = [r for r in anno if r["label"] != "normal"]
    by_type: dict[str, list[dict[str, str]]] = {}
    for r in sorted(defects, key=lambda r: (r["label"].split(",")[0], r["image"])):
        by_type.setdefault(r["label"].split(",")[0], []).append(r)
    dealt = []
    for t in sorted(by_type):
        grp = by_type[t]
        rng.shuffle(grp)
        dealt += grp
    for i, r in enumerate(dealt):
        rows.append({"image": r["image"], "label": "anomaly", "types": r["label"], "mask": r["mask"],
                     "role": f"fold{i % N_FOLDS + 1}"})
    return rows


def audit(rows: list[dict[str, str]]) -> dict:
    c = Counter(r["role"] for r in rows)
    hashes: dict[str, str] = {}
    dup = []
    for r in rows:
        r["sha256"] = sha256_file(DATA / r["image"])
        if r["sha256"] in hashes and hashes[r["sha256"]] != r["role"]:
            dup.append(r["image"])
        hashes.setdefault(r["sha256"], r["role"])
    imgs = [r["image"] for r in rows]
    a = {
        "counts": dict(sorted(c.items())),
        "disjoint": len(imgs) == len(set(imgs)),
        "cross_role_exact_duplicates": dup,
        "all_masks_present": all((DATA / r["mask"]).is_file() for r in rows if r["label"] == "anomaly"),
        "fold_sizes_ok": all(c[f"fold{k}"] == 100 // N_FOLDS for k in range(1, N_FOLDS + 1)),
        "fit_calib_ok": c["fit"] == N_FIT and c["calib"] == N_CALIB,
        "first_type_per_fold": {f"fold{k}": dict(Counter(r["types"].split(",")[0] for r in rows
                                                         if r["role"] == f"fold{k}")) for k in range(1, N_FOLDS + 1)},
    }
    a["ok"] = a["disjoint"] and not dup and a["all_masks_present"] and a["fold_sizes_ok"] and a["fit_calib_ok"]
    return a


def boxes(mask_path: Path) -> tuple[list[tuple[float, float, float, float]], int, tuple[int, int]]:
    m = np.array(Image.open(mask_path).convert("L")) > 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3), dtype=int))
    h, w = m.shape
    out = []
    for sl, idx in zip(ndimage.find_objects(lab), range(1, n + 1)):
        if sl is None or (lab[sl] == idx).sum() < MIN_COMPONENT_PX:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append(((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
    return out, int(m.sum()), (w, h)


def write_yolo_run(name: str, train_def: list[dict], bg: list[dict], test: list[dict], rng: random.Random) -> None:
    """Internal 20% validation split of (train_def + bg); test list is never trained on."""
    d = ART / "yolo" / name
    d.mkdir(parents=True, exist_ok=True)
    tdef, tbg = train_def[:], bg[:]
    rng.shuffle(tdef)
    rng.shuffle(tbg)
    nv_d, nv_b = round(0.2 * len(tdef)), round(0.2 * len(tbg))
    val, train = tdef[:nv_d] + tbg[:nv_b], tdef[nv_d:] + tbg[nv_b:]
    tr, va, te = ({r["image"] for r in s} for s in (train, val, test))
    assert not (tr & te) and not (va & te) and not (tr & va), name
    assert all(r["role"] not in ("calib", "test_n") for r in train + val), name
    lists = {"train": train, "val": val, "test": test}
    for k, rs in lists.items():
        (d / f"{k}.txt").write_text("".join(str((ART / "images" / r["image"]).resolve()) + "\n" for r in rs))
    (d / "data.yaml").write_text(yaml.safe_dump({
        "path": str(d.resolve()), "train": str((d / "train.txt").resolve()), "val": str((d / "val.txt").resolve()),
        "test": str((d / "test.txt").resolve()), "names": {0: "defect"}}))
    (d / "counts.json").write_text(json.dumps({k: {"defect": sum(r["label"] == "anomaly" for r in rs),
                                                   "normal": sum(r["label"] == "normal" for r in rs)}
                                               for k, rs in lists.items()}, indent=1))


def main() -> None:
    anno = extract()
    rows = make_split(anno)
    a = audit(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    # YOLO image tree: ultralytics maps .../images/... -> .../labels/..., so mirror images under ART/images
    sizes = Counter()
    for r in rows:
        src = DATA / r["image"]
        dst = ART / "images" / r["image"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        lab = ART / "labels" / Path(r["image"]).with_suffix(".txt")
        lab.parent.mkdir(parents=True, exist_ok=True)
        if r["label"] == "anomaly":
            bx, area, wh = boxes(DATA / r["mask"])
            r["mask_area_px"], r["n_boxes"] = str(area), str(len(bx))
            lab.write_text("".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in bx))
            sizes[wh] += 1
        else:
            r["mask_area_px"], r["n_boxes"] = "", "0"
            lab.write_text("")
            with Image.open(src) as im:
                sizes[im.size] += 1
    a["image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.items()}
    a["defects_without_boxes"] = [r["image"] for r in rows if r["label"] == "anomaly" and r["n_boxes"] == "0"]
    a["ok"] = a["ok"] and not a["defects_without_boxes"]
    fields = ["image", "label", "types", "role", "mask", "mask_area_px", "n_boxes", "sha256"]
    with open(OUT / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # YOLO runs: 5-fold CV (E1-E6) and unseen-type runs (E7); same 160 background normals throughout
    rng = random.Random(SEED)
    fit = [r for r in rows if r["role"] == "fit"]
    bg = rng.sample(fit, N_BG)
    defects = [r for r in rows if r["label"] == "anomaly"]
    for k in range(1, N_FOLDS + 1):
        test = [r for r in defects if r["role"] == f"fold{k}"]
        train = [r for r in defects if r["role"] != f"fold{k}"]
        write_yolo_run(f"cv_fold{k}", train, bg, test, random.Random(SEED + k))
    unseen = {}
    for t in UNSEEN_TYPES:
        test = [r for r in defects if r["types"] == t]
        train = [r for r in defects if t not in r["types"].split(",")]
        unseen[t] = {"test_single_label": len(test), "train_defects": len(train)}
        write_yolo_run(f"unseen_{t.replace(' ', '_')}", train, bg, test, random.Random(SEED + 100))
    a["unseen_type_runs"] = unseen
    a["background_normals"] = sorted(r["image"] for r in bg)

    # Normal-only reference configs (§5.2): size_hw from image aspect only
    (w0, h0), _ = sizes.most_common(1)[0]
    aspect = w0 / h0
    hh = 512 if abs(aspect - 1.5) <= 0.05 else int(round(768 / aspect / 32) * 32)
    a["reference_size_hw"] = [hh, 768]
    base = yaml.safe_load((ROOT / "config" / "pcb2_v1.yaml").read_text(encoding="utf-8"))
    m_split = OUT / "split_ivad.csv"
    with open(m_split, "w", newline="", encoding="utf-8") as f:  # ivad reader format; roles mapped
        wr = csv.DictWriter(f, fieldnames=["image", "label", "role"], lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({"image": r["image"], "label": r["label"], "role": r["role"]})

    def cfg_for(name: str, register: bool) -> dict:
        c = json.loads(json.dumps(base))
        c["run_name"] = f"Exploratory E1 normal-only reference {name} (PCB4)"
        c["paths"].update({"official_manifest": None, "split_manifest": str(m_split.relative_to(ROOT)).replace("\\", "/"),
                           "artifacts": f"artifacts/exploratory_yolo_pcb4/{name}",
                           "reports": f"artifacts/exploratory_yolo_pcb4/{name}/reports",
                           "inspection": f"artifacts/exploratory_yolo_pcb4/{name}/inspection",
                           "figures": f"artifacts/exploratory_yolo_pcb4/{name}/figures"})
        c["dataset"].update({"object": "pcb4", "n_fit": N_FIT, "n_calib": N_CALIB})
        c["preprocess"].update({"register": register, "crop_wh": None, "size_hw": [hh, 768]})
        return c

    hdr = "# GENERATED by exploratory/yolo_pcb4/prepare.py (PROTOCOL.md §5.2). Do not edit.\n"
    (HERE / "config_m.yaml").write_text(hdr + yaml.safe_dump(cfg_for("m", False), sort_keys=False), encoding="utf-8")

    # M-reg gate: FIT-only, original board-mask rule covers 5%-80% of the frame on >= 95% of FIT images
    fr = []
    for r in fit:
        with Image.open(DATA / r["image"]) as im:
            rgb = im.convert("RGB")
        try:
            fr.append(float(foreground_mask(rgb).mean()))  # same rule and scale as align.py
        except ValueError:
            fr.append(0.0)
    fr = np.array(fr)
    ok_frac = float(((fr >= 0.05) & (fr <= 0.80)).mean())
    a["m_reg_gate"] = {"fit_frac_in_range": round(ok_frac, 4), "median_mask_frac": round(float(np.median(fr)), 4),
                       "passes": ok_frac >= 0.95, "rule": "original board mask 5%-80% of frame on >=95% of FIT"}
    if ok_frac >= 0.95:
        (HERE / "config_m_reg.yaml").write_text(hdr + yaml.safe_dump(cfg_for("m_reg", True), sort_keys=False),
                                               encoding="utf-8")
    (OUT / "split_audit.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
    print(json.dumps({k: a[k] for k in ("counts", "ok", "image_sizes", "reference_size_hw", "m_reg_gate",
                                         "unseen_type_runs", "defects_without_boxes")}, indent=1))
    if not a["ok"]:
        raise SystemExit("split audit FAILED")


if __name__ == "__main__":
    main()
