"""The PCB2 v1 and v2 configs isolate the single pre-registered change.

v1 (PCB1), PCB2-v1 and PCB2-v2 must share every model, preprocessing, calibration
and robustness setting; PCB2-v1 and PCB2-v2 may differ only in output paths, the
run name, the `scoring` section and v2's `comparison` settings.
"""
import copy

import pytest

from ivad.config import REPO_ROOT, config_hash, load_config

PCB1, V1, V2 = (REPO_ROOT / "config" / f for f in ("default.yaml", "pcb2_v1.yaml", "pcb2_v2.yaml"))
SHARED = ("seed", "preprocess", "model", "calibration", "robustness", "cost")


@pytest.mark.parametrize("path", [V1, V2])
def test_pcb2_configs_keep_v1_model_settings(path):
    a, b = load_config(PCB1), load_config(path)
    for k in SHARED:
        assert a[k] == b[k], k
    ev = dict(b["evaluation"])
    assert ev.pop("require_committed_freeze") is True
    assert ev == a["evaluation"]
    da, db = dict(a["dataset"]), dict(b["dataset"])
    assert (db.pop("object"), db.pop("n_holdout")) == ("pcb2", 151)
    da.pop("object"), da.pop("n_holdout")
    assert da == db


def test_v1_and_v2_differ_only_in_scoring():
    a, b = load_config(V1), load_config(V2)
    assert a["scoring"] == {"aggregation": "full_max"}
    assert b["scoring"]["aggregation"] == "roi_max"
    strip = []
    for c in (a, b):
        c = copy.deepcopy(c)
        for k in ("_config_path", "run_name", "scoring", "comparison"):
            c.pop(k, None)
        for k in ("artifacts", "reports", "figures"):
            c["paths"].pop(k)
        strip.append(c)
    assert strip[0] == strip[1]
    assert a["paths"]["split_manifest"] == b["paths"]["split_manifest"]
    assert a["paths"]["artifacts"] != b["paths"]["artifacts"]


def test_pcb1_v1_config_hash_unchanged():
    """The original PCB1 freeze stays valid: v2 work must not alter the v1 config hash."""
    assert config_hash(load_config(PCB1)) == "218f816a91c4932960e0476882fed6fa825919429cfca8ce31cfebce9650e918"
