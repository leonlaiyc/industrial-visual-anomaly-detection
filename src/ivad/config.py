"""Configuration loading and fingerprinting.

The config file is the single place where model and protocol settings live.
Its canonical hash is written into the frozen operating point so evaluation can
refuse to run against a config that changed after the threshold was frozen.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

# Keys that may legitimately differ between machines without changing results.
_NON_SEMANTIC_KEYS = {("runtime", "device"), ("runtime", "num_workers"), ("runtime", "batch_size")}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"].get("root") or REPO_ROOT)


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve cfg['paths'][key] relative to the config's root directory."""
    root = project_root(cfg)
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else root / p


def semantic_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config subset that affects results (drops private and runtime-only keys)."""
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}
    out.get("paths", {}).pop("root", None)
    for section, key in _NON_SEMANTIC_KEYS:
        out.get(section, {}).pop(key, None)
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(semantic_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_text(path: str | Path) -> str:
    """Hash of a text file with line endings normalised to LF.

    Git may check text files out with CRLF on Windows; the frozen-operating-point
    check must not fail on a clean clone just because of line endings.
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    """UTF-8, LF line endings on every OS."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=2) + "\n")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
