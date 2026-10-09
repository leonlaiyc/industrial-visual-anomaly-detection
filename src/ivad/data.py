"""Dataset acquisition, frozen split creation and leakage audit for a VisA object.

Roles assigned to every image (one role per image, enforced by `audit_split`):

    fit       600 official-train normals  -> fit the Gaussian patch model
    calib     150 official-train normals  -> choose thresholds (no fitting)
    holdout   rest of the official-train normals (PCB1 154, PCB2 151)
              -> supplementary FPR estimate only
    test      100 normal + 100 anomaly    -> official VisA 1-class test split
"""
from __future__ import annotations

import csv
import io
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

from .config import resolve_path, sha256_file, write_json

ROLES = ("fit", "calib", "holdout", "test")


# --------------------------------------------------------------------------- #
# Official manifest
# --------------------------------------------------------------------------- #
def _upstream_manifest(cfg: dict[str, Any]) -> tuple[bytes, str]:
    """Raw `split_csv/1cls.csv`: from the official archive if it is present, else from the URL."""
    archive = resolve_path(cfg, "archive")
    if archive.exists():
        with tarfile.open(archive, mode="r|*") as tar:
            for m in tar:
                if m.isfile() and PurePosixPath(m.name).parts[-2:] == ("split_csv", "1cls.csv"):
                    src = tar.extractfile(m)
                    assert src is not None
                    return src.read(), f"{archive.name}:{m.name}"
    with urllib.request.urlopen(cfg["dataset"]["manifest_url"], timeout=60) as r:
        return r.read(), cfg["dataset"]["manifest_url"]


def fetch_official_manifest(cfg: dict[str, Any]) -> Path:
    """Write the official 1-class split rows for the configured object.

    Reads the upstream `split_csv/1cls.csv` (official archive, or the spot-diff
    URL), keeps rows for the object and records the upstream file hash for
    provenance. Skips if already present.
    """
    out = resolve_path(cfg, "official_manifest")
    if out.exists():
        return out
    obj = cfg["dataset"]["object"]
    raw, source = _upstream_manifest(cfg)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    keep = [r for r in rows if r["object"] == obj]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["object", "split", "label", "image", "mask"], lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
    import hashlib

    prov = {
        "source_url": cfg["dataset"]["manifest_url"],
        "read_from": source,
        "upstream_sha256": hashlib.sha256(raw).hexdigest(),
        "object": obj,
        "rows_kept": len(keep),
    }
    write_json(out.with_suffix(".provenance.json"), prov)
    return out


def read_official_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    c = Counter(f"{r['split']}/{r['label']}" for r in rows)
    return dict(sorted(c.items()))


# --------------------------------------------------------------------------- #
# Archive download and selective extraction
# --------------------------------------------------------------------------- #
def download_archive(cfg: dict[str, Any]) -> Path:
    """Download the official VisA tar (skips if the expected size is present)."""
    dst = resolve_path(cfg, "archive")
    expected = int(cfg["dataset"]["archive_bytes"])
    if dst.exists() and dst.stat().st_size == expected:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {cfg['dataset']['archive_url']} -> {dst} (~{expected / 1e9:.2f} GB)")
    urllib.request.urlretrieve(cfg["dataset"]["archive_url"], dst)
    if dst.stat().st_size != expected:
        raise RuntimeError(f"archive size {dst.stat().st_size} != expected {expected}")
    return dst


def _safe_relative(member_name: str, obj: str) -> PurePosixPath | None:
    """Map a tar member to '<obj>/...' or None; rejects absolute / '..' paths."""
    p = PurePosixPath(member_name)
    if p.is_absolute() or ".." in p.parts:
        return None
    if obj not in p.parts:
        return None
    return PurePosixPath(*p.parts[p.parts.index(obj):])


def extract_object(cfg: dict[str, Any]) -> Path:
    """Stream the tar once and extract only regular files under the object."""
    obj = cfg["dataset"]["object"]
    out_root = resolve_path(cfg, "data_dir")
    archive = resolve_path(cfg, "archive")
    n = 0
    with tarfile.open(archive, mode="r|*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            rel = _safe_relative(m.name, obj)
            if rel is None:
                continue
            dst = out_root / Path(*rel.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            with open(dst, "wb") as f:
                f.write(src.read())
            n += 1
    print(f"Extracted {n} files for '{obj}' into {out_root}")
    return out_root


def verify_files(cfg: dict[str, Any], rows: list[dict[str, str]]) -> None:
    root = resolve_path(cfg, "data_dir")
    missing = [r["image"] for r in rows if not (root / r["image"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images missing, e.g. {missing[:3]}")


# --------------------------------------------------------------------------- #
# Frozen split
# --------------------------------------------------------------------------- #
def make_split(
    official_rows: list[dict[str, str]],
    n_fit: int,
    n_calib: int,
    n_holdout: int,
    seed: int,
) -> list[dict[str, str]]:
    """Deterministically partition official-train normals into fit/calib/holdout.

    Images are sorted by path before permuting so the result depends only on the
    seed and the manifest contents, never on file-system or CSV row order.
    """
    train = sorted(r["image"] for r in official_rows if r["split"] == "train")
    if any(r["label"] != "normal" for r in official_rows if r["split"] == "train"):
        raise ValueError("official train split must contain only normal images")
    if len(train) != n_fit + n_calib + n_holdout:
        raise ValueError(
            f"train has {len(train)} images but split sizes sum to {n_fit + n_calib + n_holdout}"
        )
    perm = np.random.default_rng(seed).permutation(len(train))
    role_of = {}
    for i, idx in enumerate(perm):
        role_of[train[idx]] = "fit" if i < n_fit else "calib" if i < n_fit + n_calib else "holdout"

    out = []
    for r in sorted(official_rows, key=lambda r: (r["split"], r["label"], r["image"])):
        role = role_of[r["image"]] if r["split"] == "train" else "test"
        out.append(
            {"image": r["image"], "official_split": r["split"], "label": r["label"], "role": role}
        )
    return out


def add_hashes(split_rows: list[dict[str, str]], data_root: Path) -> None:
    for r in split_rows:
        r["sha256"] = sha256_file(data_root / r["image"])


def audit_split(
    split_rows: list[dict[str, str]], expected_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Raise ValueError on any leakage; return a summary dict when clean."""
    problems: list[str] = []
    paths = Counter(r["image"] for r in split_rows)
    dup_paths = [p for p, c in paths.items() if c > 1]
    if dup_paths:
        problems.append(f"{len(dup_paths)} image paths assigned more than once: {dup_paths[:3]}")

    for r in split_rows:
        if r["role"] not in ROLES:
            problems.append(f"unknown role {r['role']!r} for {r['image']}")
        if r["role"] in ("fit", "calib", "holdout") and (
            r["official_split"] != "train" or r["label"] != "normal"
        ):
            problems.append(f"{r['image']} in role {r['role']} is not an official-train normal")
        if r["role"] == "test" and r["official_split"] != "test":
            problems.append(f"{r['image']} in role test is not in the official test split")
        if r["official_split"] == "test" and r["role"] != "test":
            problems.append(f"official test image {r['image']} leaked into role {r['role']}")

    if all("sha256" in r for r in split_rows):
        roles_by_hash: dict[str, set[str]] = {}
        for r in split_rows:
            roles_by_hash.setdefault(r["sha256"], set()).add(r["role"])
        cross = {h: sorted(rs) for h, rs in roles_by_hash.items() if len(rs) > 1}
        if cross:
            problems.append(f"{len(cross)} identical file contents shared across roles")
        n_dup_within = sum(1 for r in split_rows) - len(roles_by_hash)
    else:
        n_dup_within = None

    counts = Counter(
        f"{r['role']}/{r['label']}" for r in split_rows
    )
    if expected_counts:
        for k, v in expected_counts.items():
            if counts.get(k, 0) != v:
                problems.append(f"count {k}={counts.get(k, 0)} expected {v}")

    if problems:
        raise ValueError("split audit FAILED:\n  " + "\n  ".join(problems))
    return {
        "status": "PASS",
        "counts": dict(sorted(counts.items())),
        "unique_paths": len(paths),
        "content_hash_checked": n_dup_within is not None,
        "identical_contents_total": n_dup_within,
    }


def write_split(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "official_split", "label", "role", "sha256"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in fields if k in rows[0]], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows: list[dict[str, str]], role: str, label: str | None = None) -> list[dict[str, str]]:
    return [r for r in rows if r["role"] == role and (label is None or r["label"] == label)]


def expected_counts(cfg: dict[str, Any]) -> dict[str, int]:
    d = cfg["dataset"]
    return {
        "fit/normal": d["n_fit"],
        "calib/normal": d["n_calib"],
        "holdout/normal": d["n_holdout"],
        "test/normal": d["n_test_normal"],
        "test/anomaly": d["n_test_anomaly"],
    }
