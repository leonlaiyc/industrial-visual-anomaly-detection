"""Status of one evaluated run against the PROPOSED acceptance criteria.

Point estimates are compared with config/acceptance_criteria.yaml (unchanged
since v1). Confidence intervals are reported elsewhere, next to these
statuses; a status is never rounded into a pass.
"""
from __future__ import annotations

from typing import Any

import yaml

from .config import REPO_ROOT

ACCEPTANCE_FILE = REPO_ROOT / "config" / "acceptance_criteria.yaml"


def load_criteria() -> dict[str, Any]:
    return yaml.safe_load(ACCEPTANCE_FILE.read_text(encoding="utf-8"))


def acceptance_status(m: dict, rb: dict | None, pr: str, acc: dict) -> dict[str, Any]:
    """Per-criterion booleans, NO-GO triggers and the verdict (GO / NO-GO / INCONCLUSIVE).

    `rb` (robustness.json) may be None: the robustness criterion is then unevaluated and
    the verdict cannot be GO.
    """
    p, ng = acc["performance"], acc["no_go"]
    op = m["official_test"]["operating_points"][pr]
    hold = m["supplementary_holdout"]["operating_points"][pr]["fpr"]["value"]
    st: dict[str, Any] = {
        "auroc": m["official_test"]["auroc"]["value"] >= p["auroc_min"],
        "recall": op["recall"]["value"] >= p["recall_min"],
        "fpr_test": op["fpr"]["value"] <= p["fpr_max"],
        "fpr_holdout": hold <= p["fpr_max"],
        "latency": m["latency"]["median_ms"] <= p["latency_ms_max"],
        "robustness": None,
    }
    no_go = {
        "recall_below_floor": op["recall"]["value"] < ng["recall_floor"],
        "fpr_above_ceiling": max(op["fpr"]["value"], hold) > ng["fpr_ceiling"],
    }
    if rb is not None:
        # P5 is defined over the original perturbations; conditions added later for a specific
        # experiment carry `acceptance: false` in their spec and are reported, not scored.
        pert = [v for v in rb["perturbations"].values() if v.get("spec", {}).get("acceptance", True)]
        drop = max(-v["paired"]["delta_recall"]["value"] for v in pert)
        rise = max(v["paired"]["delta_fpr"]["value"] for v in pert)
        st["robustness"] = drop <= p["robustness_recall_drop_max"] and rise <= p["robustness_fpr_rise_max"]
        st["robustness_max_recall_drop"], st["robustness_max_fpr_rise"] = drop, rise
        no_go["robustness_collapse"] = drop > ng["robustness_recall_collapse"]
    st["no_go_triggers"] = [k for k, v in no_go.items() if v]
    keys = ("auroc", "recall", "fpr_test", "fpr_holdout", "latency", "robustness")
    st["verdict"] = ("NO-GO" if st["no_go_triggers"] else
                     "GO" if all(st[k] is True for k in keys) else "INCONCLUSIVE")
    return st
