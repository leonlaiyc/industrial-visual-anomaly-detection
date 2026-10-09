"""Split determinism, isolation and the leakage audit."""
import copy
from pathlib import Path

import pytest

from ivad.config import REPO_ROOT, load_config
from ivad.data import audit_split, expected_counts, make_split, read_split


def _manifest(n_train=30, n_test_n=5, n_test_a=5):
    rows = [{"object": "x", "split": "train", "label": "normal", "image": f"x/N/{i:04d}.JPG"}
            for i in range(n_train)]
    rows += [{"object": "x", "split": "test", "label": "normal", "image": f"x/TN/{i:03d}.JPG"}
             for i in range(n_test_n)]
    rows += [{"object": "x", "split": "test", "label": "anomaly", "image": f"x/A/{i:03d}.JPG"}
             for i in range(n_test_a)]
    return rows


EXP = {"fit/normal": 20, "calib/normal": 6, "holdout/normal": 4, "test/normal": 5, "test/anomaly": 5}


def test_split_counts_and_disjoint():
    s = make_split(_manifest(), 20, 6, 4, seed=1)
    audit = audit_split(s, EXP)
    assert audit["status"] == "PASS" and audit["unique_paths"] == 40
    roles = {r: {x["image"] for x in s if x["role"] == r} for r in ("fit", "calib", "holdout", "test")}
    for a in roles:
        for b in roles:
            if a < b:
                assert not roles[a] & roles[b]


def test_split_is_deterministic_and_order_independent():
    m = _manifest()
    a = make_split(m, 20, 6, 4, seed=1)
    b = make_split(list(reversed(m)), 20, 6, 4, seed=1)
    c = make_split(m, 20, 6, 4, seed=2)
    assert a == b
    assert a != c


def test_split_rejects_wrong_sizes_and_abnormal_train():
    with pytest.raises(ValueError):
        make_split(_manifest(), 20, 6, 5, seed=1)
    m = _manifest()
    m[0]["label"] = "anomaly"
    with pytest.raises(ValueError):
        make_split(m, 20, 6, 4, seed=1)


def test_audit_detects_test_leak():
    s = make_split(_manifest(), 20, 6, 4, seed=1)
    leaked = copy.deepcopy(s)
    t = next(r for r in leaked if r["role"] == "test")
    t["role"] = "fit"
    with pytest.raises(ValueError, match="leaked"):
        audit_split(leaked)


def test_audit_detects_duplicate_path_and_content():
    s = make_split(_manifest(), 20, 6, 4, seed=1)
    with pytest.raises(ValueError, match="more than once"):
        audit_split(s + [dict(s[0])])
    for i, r in enumerate(s):
        r["sha256"] = f"h{i}"
    fit = next(r for r in s if r["role"] == "fit")
    cal = next(r for r in s if r["role"] == "calib")
    cal["sha256"] = fit["sha256"]  # identical bytes under two names
    with pytest.raises(ValueError, match="identical file contents"):
        audit_split(s)


def test_audit_detects_count_mismatch():
    s = make_split(_manifest(), 20, 6, 4, seed=1)
    with pytest.raises(ValueError, match="count"):
        audit_split(s, {**EXP, "fit/normal": 21})


def test_committed_split_manifest_passes_audit():
    cfg = load_config()
    p = REPO_ROOT / cfg["paths"]["split_manifest"]
    if not p.exists():
        pytest.skip("split manifest not generated")
    rows = read_split(p)
    audit = audit_split(rows, expected_counts(cfg))
    assert audit["content_hash_checked"] and audit["unique_paths"] == 1104


def test_manifest_hash_ignores_line_endings(tmp_path):
    """A CRLF checkout of the split manifest must not invalidate the freeze."""
    from ivad.config import sha256_text

    a, b = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    a.write_bytes(b"image,role\nx.png,fit\n")
    b.write_bytes(b"image,role\r\nx.png,fit\r\n")
    assert sha256_text(a) == sha256_text(b)
