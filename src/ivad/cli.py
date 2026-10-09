"""Command-line workflow.

    ivad prepare      official manifest + archive, extract the configured object, verify files
    ivad split        create the frozen fit/calib/holdout/test manifest + audit
    ivad inspect      contact sheet and alignment stats from FIT normals only
    ivad fit          fit the per-position Gaussian model on FIT normals
    ivad calibrate    score CALIB normals, choose and FREEZE thresholds
    ivad evaluate     official test + supplementary holdout (requires freeze)
    ivad robustness   perturbation study at the frozen threshold
    ivad report       figures, cost-sensitivity, sample outputs, results.md
    ivad roi          (v2 configs) ROI quality figure from FIT normals, after `fit`
    ivad compare      paired v1-vs-v2 comparison of two evaluated runs (--base, --new)
    ivad summary      README results block + one-page executive PDF from all runs
    ivad check        consistency checks across all runs, the PDF and the README
    ivad predict IMG  score one image and write a heat-map overlay
    ivad photostats   (PCB3 illumination experiment) FIT-only luminance reference values
    ivad illum-compare  paired baseline-vs-candidate illumination analysis (--base, --new)
    ivad fit-diagnostics  (PCB3, Addendum B2) FIT-only registration/ROI/photometric diagnostics (--base, --new)
    ivad all          prepare -> split -> fit -> calibrate -> evaluate -> robustness -> report

Every command accepts --config PATH (default: config/default.yaml = PCB1 v1).
The PCB2 study uses config/pcb2_v1.yaml and config/pcb2_v2.yaml; the PCB3 illumination
experiment uses config/pcb3_baseline.yaml and config/pcb3_illum.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config, resolve_path, write_json


def cmd_prepare(cfg) -> None:
    from . import data

    m = data.fetch_official_manifest(cfg)
    rows = data.read_official_manifest(m)
    print("official manifest counts:", data.manifest_counts(rows))
    root = resolve_path(cfg, "data_dir")
    if not all((root / r["image"]).exists() for r in rows):
        data.download_archive(cfg)
        data.extract_object(cfg)
    data.verify_files(cfg, rows)
    print(f"all {len(rows)} manifest images present under {root}")


def cmd_split(cfg) -> None:
    from . import data

    rows = data.read_official_manifest(resolve_path(cfg, "official_manifest"))
    d = cfg["dataset"]
    split = data.make_split(rows, d["n_fit"], d["n_calib"], d["n_holdout"], cfg["seed"])
    data.add_hashes(split, resolve_path(cfg, "data_dir"))
    audit = data.audit_split(split, data.expected_counts(cfg))
    out = resolve_path(cfg, "split_manifest")
    data.write_split(split, out)
    audit["seed"] = cfg["seed"]
    write_json(out.with_name(f"{out.stem}_audit.json"), audit)
    print(json.dumps(audit, indent=2))


def cmd_inspect(cfg) -> None:
    from .data import read_split
    from .inspect_data import run_inspection
    from .pipeline import mask_kwargs, pmap

    out = resolve_path(cfg, "inspection") if "inspection" in cfg["paths"] else resolve_path(cfg, "reports") / "inspection"
    rep = run_inspection(read_split(resolve_path(cfg, "split_manifest")), resolve_path(cfg, "data_dir"),
                         out, cfg["seed"],
                         crop_wh=cfg["preprocess"].get("crop_wh"), pmap=pmap(cfg), mask_kw=mask_kwargs(cfg))
    print(json.dumps(rep["registration_all_fit"], indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ivad", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "split", "inspect", "fit", "calibrate", "evaluate",
                 "robustness", "report", "roi", "summary", "check", "all", "photostats"):
        sub.add_parser(name)
    cp = sub.add_parser("compare")
    cp.add_argument("--base", default="config/pcb2_v1.yaml")
    cp.add_argument("--new", default="config/pcb2_v2.yaml")
    ip = sub.add_parser("illum-compare")
    ip.add_argument("--base", default="config/pcb3_baseline.yaml")
    ip.add_argument("--new", default="config/pcb3_illum.yaml")
    fd = sub.add_parser("fit-diagnostics")
    fd.add_argument("--base", default="config/pcb3_baseline.yaml")
    fd.add_argument("--new", default="config/pcb3_illum.yaml")
    pr = sub.add_parser("predict")
    pr.add_argument("image")
    pr.add_argument("--out", default=None, help="heat-map PNG path")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    from . import pipeline

    steps = {
        "prepare": cmd_prepare,
        "split": cmd_split,
        "inspect": cmd_inspect,
        "fit": pipeline.fit,
        "calibrate": pipeline.calibrate,
        "evaluate": pipeline.evaluate,
        "robustness": pipeline.robustness,
    }
    if args.cmd in steps:
        steps[args.cmd](cfg)
    elif args.cmd == "report":
        from .report import build_report
        build_report(cfg)
    elif args.cmd == "roi":
        from .report import roi_figure
        print(json.dumps(roi_figure(cfg), indent=2))
    elif args.cmd == "compare":
        from .compare import compare_runs
        compare_runs(load_config(args.base), load_config(args.new))
    elif args.cmd == "photostats":
        from .illum import photostats
        photostats(cfg)
    elif args.cmd == "illum-compare":
        from .illum import illum_compare
        illum_compare(load_config(args.base), load_config(args.new))
    elif args.cmd == "fit-diagnostics":
        from .fitdiag import run as fit_diagnostics
        fit_diagnostics(load_config(args.base), load_config(args.new))
    elif args.cmd == "summary":
        from .summary import build_summary
        build_summary()
    elif args.cmd == "check":
        from .consistency import run_checks
        return 0 if run_checks() else 1
    elif args.cmd == "predict":
        from .report import predict_one
        print(json.dumps(predict_one(cfg, args.image, args.out), indent=2))
    elif args.cmd == "all":
        from .report import build_report
        for name in ("prepare", "split", "fit", "calibrate", "evaluate", "robustness"):
            steps[name](cfg)
        build_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
