"""Audit export (2026-10-10, after the experiment): turn the gitignored experiment inputs into small, path-free
files an external reviewer can check without the dataset or the weights. Changes no score, threshold or result.

Needs the development workspace (dataset under data/visa/pcb4, YOLO list and label files under
artifacts/exploratory_yolo_pcb4/). Project environment:

    .venv/Scripts/python.exe exploratory/yolo_pcb4/audit/build_manifests.py

Writes, in exploratory/yolo_pcb4/audit/:
  fold_manifest.csv       every image's partition in every YOLO run, read from the ACTUAL list files used for training
  yolo_labels.csv         every box label ACTUALLY used for training, plus a re-derivation check from the masks
  dataset_identity.json   image/mask geometry, exact and near-duplicate hashes across roles, official VisA splits
  fig_annotation_check.jpg  training box labels drawn on source images (VisA derivative, CC BY 4.0)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
E1 = HERE.parent
ROOT = E1.parents[1]
sys.path.insert(0, str(E1))
import prepare  # noqa: E402  (the experiment's own box rule, imported, not re-implemented)

RES = E1 / "results"
ART = ROOT / "artifacts" / "exploratory_yolo_pcb4"
DATA = ROOT / "data" / "visa"
ARCHIVE = ROOT / "data" / "raw" / "VisA_20220922.tar"
RUNS = [f"cv_fold{k}" for k in range(1, 6)] + ["unseen_missing", "unseen_extra"]
IMG_MARK = "/images/"  # list files hold absolute paths <ART>/images/<image id>; keep only the image id


def rel_id(abs_path: str) -> str:
    p = abs_path.strip().replace("\\", "/")
    i = p.index(IMG_MARK)
    return p[i + len(IMG_MARK):]


def dhash(path: Path, size: int = 16) -> int:
    """Difference hash of the grey image resized to (size+1) x size: resize-invariant near-duplicate identity."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    split = list(csv.DictReader(open(RES / "split.csv", newline="", encoding="utf-8")))
    by_id = {r["image"]: r for r in split}

    # 1. fold manifest from the actual list files
    rows = []
    for run in RUNS:
        d = ART / "yolo" / run
        member = {}
        for part, fn in (("train", "train.txt"), ("internal_val", "val.txt"), ("test", "test.txt")):
            for ln in (d / fn).read_text().splitlines():
                if ln.strip():
                    i = rel_id(ln)
                    assert i not in member, (run, i)
                    member[i] = part
        for r in split:
            part = member.get(r["image"])
            if part is None:
                part = {"calib": "calib", "test_n": "test_n"}.get(r["role"], "unused")
            else:
                assert r["role"] not in ("calib", "test_n"), (run, r["image"])
            rows.append({"run": run, "image": r["image"], "label": r["label"], "types": r["types"],
                         "split_role": r["role"], "partition": part})
    with open(HERE / "fold_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # 2. YOLO labels actually used, and an independent re-derivation from the masks with the experiment's rule
    lab_rows, mismatches = [], []
    for r in split:
        if r["label"] != "anomaly":
            t = (ART / "labels" / Path(r["image"]).with_suffix(".txt")).read_text()
            assert t == "", r["image"]  # normal images: empty label file (background)
            continue
        lp = ART / "labels" / Path(r["image"]).with_suffix(".txt")
        text = lp.read_text()
        used = [tuple(float(v) for v in ln.split()[1:]) for ln in text.splitlines() if ln.strip()]
        again, _, _ = prepare.boxes(DATA / r["mask"])
        if len(again) != len(used) or any(max(abs(a - b) for a, b in zip(u, g)) > 1e-6 for u, g in zip(used, again)):
            mismatches.append(r["image"])
        for j, (xc, yc, bw, bh) in enumerate(used):
            lab_rows.append({"image": r["image"], "box": j, "class": 0, "xc": f"{xc:.6f}", "yc": f"{yc:.6f}",
                             "w": f"{bw:.6f}", "h": f"{bh:.6f}",
                             "label_file_sha256": hashlib.sha256(text.encode()).hexdigest()})
    with open(HERE / "yolo_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lab_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(lab_rows)

    # 3. dataset identity: geometry, exact duplicates, near duplicates, official splits
    geom = Counter()
    mask_geom_mismatch = []
    sha_now = {}
    dh = {}
    for r in split:
        p = DATA / r["image"]
        sha_now[r["image"]] = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            geom[f"{im.size[0]}x{im.size[1]}"] += 1
            size = im.size
        if r["mask"]:
            with Image.open(DATA / r["mask"]) as m:
                if m.size != size:
                    mask_geom_mismatch.append(r["image"])
        dh[r["image"]] = dhash(p)
    sha_changed = [i for i in sha_now if sha_now[i] != by_id[i]["sha256"]]
    ids = list(dh)
    hs = np.array([[(dh[i] >> b) & 1 for b in range(256)] for i in ids], dtype=np.uint8)
    near = []  # cross-role pairs within Hamming distance <= 10 of 256 bits
    for a in range(len(ids)):
        d = (hs[a + 1:] != hs[a]).sum(1)
        for b in np.nonzero(d <= 10)[0]:
            j = a + 1 + int(b)
            if by_id[ids[a]]["role"] != by_id[ids[j]]["role"]:
                near.append({"a": ids[a], "a_role": by_id[ids[a]]["role"], "b": ids[j], "b_role": by_id[ids[j]]["role"],
                             "hamming": int(d[b])})
    min_cross = []
    for a in range(len(ids)):  # nearest cross-role neighbour distance distribution
        d = (hs != hs[a]).sum(1).astype(float)
        same = np.array([by_id[ids[j]]["role"] == by_id[ids[a]]["role"] for j in range(len(ids))])
        d[same] = np.inf
        min_cross.append(float(d.min()))
    official = {}
    with tarfile.open(ARCHIVE) as tar:
        for name in ("1cls", "2cls_highshot", "2cls_fewshot"):
            raw = tar.extractfile(f"split_csv/{name}.csv").read()
            rr = [x for x in csv.DictReader(io.StringIO(raw.decode("utf-8"))) if x["object"] == "pcb4"]
            c = Counter((x["split"], x["label"]) for x in rr)
            ours = defaultdict(Counter)
            for x in rr:
                ours[f"{x['split']}/{x['label']}"][by_id[x["image"]]["role"] if x["image"] in by_id else "absent"] += 1
            official[name] = {"upstream_sha256": hashlib.sha256(raw).hexdigest(), "pcb4_counts": {f"{s}/{l}": n for (s, l), n in sorted(c.items())},
                              "official_partition_to_e1_role": {k: dict(v) for k, v in sorted(ours.items())}}
    multi = [r for r in split if r["label"] == "anomaly" and "," in r["types"]]
    out = {
        "note": "built after the experiment for audit; reads inputs only",
        "n_images": len(split), "labels": dict(Counter(r["label"] for r in split)),
        "roles": dict(Counter(r["role"] for r in split)),
        "image_geometry": dict(geom), "mask_geometry_mismatches": mask_geom_mismatch,
        "sha256_recorded_vs_now_mismatches": sha_changed,
        "exact_duplicate_sha256_groups": [list(g) for g in _groups(sha_now).values() if len(g) > 1],
        "near_duplicate_rule": "dHash 16x16 (256 bits) of the grey image; pairs in DIFFERENT roles within Hamming <= 10",
        "near_duplicate_cross_role_pairs": near,
        "nearest_cross_role_hamming": {"min": min(min_cross), "p01": float(np.quantile(min_cross, 0.01)),
                                       "median": float(np.median(min_cross))},
        "multi_type_defect_images": len(multi),
        "type_label_counts": dict(Counter(r["types"] for r in split if r["label"] == "anomaly")),
        "boxes_per_defect_image": dict(sorted(Counter(int(r["n_boxes"]) for r in split if r["label"] == "anomaly").items())),
        "label_files_rederived_from_masks_mismatches": mismatches,
        "n_label_boxes_total": len(lab_rows),
        "official_visa_splits_pcb4": official,
    }
    (HERE / "dataset_identity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # 4. visual check of training annotations: 2 images per first-listed type, boxes (red) + mask outline (cyan)
    picks, seen = [], Counter()
    for r in sorted((r for r in split if r["label"] == "anomaly"), key=lambda r: r["image"]):
        t = r["types"].split(",")[0]
        if seen[t] < 2 or ("," in r["types"] and seen["multi"] < 2):
            seen[t] += 1
            seen["multi"] += "," in r["types"]
            picks.append(r)
    tiles = []
    for r in picks[:16]:
        im = Image.open(DATA / r["image"]).convert("RGB")
        m = Image.open(DATA / r["mask"]).convert("L").point(lambda v: 255 if v > 0 else 0)
        im.paste((0, 230, 255), mask=ImageChops.subtract(m.filter(ImageFilter.MaxFilter(9)), m))
        dr = ImageDraw.Draw(im)
        W, H = im.size
        for b in (x for x in lab_rows if x["image"] == r["image"]):
            xc, yc, bw, bh = (float(b[k]) for k in ("xc", "yc", "w", "h"))
            dr.rectangle([(xc - bw / 2) * W - 6, (yc - bh / 2) * H - 6, (xc + bw / 2) * W + 6, (yc + bh / 2) * H + 6],
                         outline=(255, 30, 30), width=5)
        im.thumbnail((340, 340))
        tile = Image.new("RGB", (340, im.height + 18), "white")
        tile.paste(im, (0, 18))
        ImageDraw.Draw(tile).text((3, 3), f"{Path(r['image']).name}: {r['types']} ({r['n_boxes']} box)", fill=(0, 0, 0))
        tiles.append(tile)
    cols = 4
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (340 * cols, th * ((len(tiles) + cols - 1) // cols) + 22), "white")
    for n, t in enumerate(tiles):
        sheet.paste(t, (340 * (n % cols), th * (n // cols)))
    ImageDraw.Draw(sheet).text((4, sheet.height - 18), "VisA PCB4 (Zou et al., ECCV 2022), CC BY 4.0. Red: YOLO training "
                               "box (drawn 6 px outside). Cyan: ground-truth mask outline.", fill=(0, 0, 0))
    sheet.save(HERE / "fig_annotation_check.jpg", quality=82)
    print(json.dumps({k: out[k] for k in ("image_geometry", "mask_geometry_mismatches", "sha256_recorded_vs_now_mismatches",
                                          "exact_duplicate_sha256_groups", "nearest_cross_role_hamming", "multi_type_defect_images",
                                          "label_files_rederived_from_masks_mismatches", "n_label_boxes_total")}, indent=1))
    print("near-duplicate cross-role pairs:", len(near))


def _groups(sha: dict[str, str]) -> dict[str, list[str]]:
    g: dict[str, list[str]] = defaultdict(list)
    for i, h in sha.items():
        g[h].append(i)
    return g


if __name__ == "__main__":
    main()
