"""End-to-end steps: fit -> calibrate (freeze) -> evaluate -> robustness.

Protocol guards implemented here (see docs/evaluation_methodology.md):
  * fit reads only role=fit images; calibrate reads only role=calib images.
  * calibrate writes reports/frozen_operating_point.json containing the config
    hash, model-state hash and split-manifest hash.
  * evaluate / robustness refuse to run unless all three hashes still match,
    so the threshold cannot be silently re-tuned after test scores are seen.
  * with `evaluation.require_committed_freeze` (PCB2 runs), evaluate also
    refuses unless the config, split manifest, freeze file and src/ are
    committed to git and unchanged.

Image-score aggregation is `scoring.aggregation`: `full_max` (v1, the
default when the section is absent) or `roi_max` (v2, see roi.py).
An optional `preprocess.photometric` step (photometric.py, PCB3 illumination
experiment) runs first on every image, before registration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from . import calibration, metrics, photometric
from . import roi as roi_mod
from .align import Registrar
from .config import config_hash, project_root, resolve_path, sha256_file, sha256_text, write_json
from .data import read_split, select
from .features import (PatchFeatureExtractor, build_extractor, center_crop, check_batch, load_rgb,
                       to_tensor, weights_file, weights_info)
from .gaussian import CovarianceAccumulator, GaussianPatchModel, anomaly_map, image_score
from .robustness import apply_perturbation

Transform = Optional[Callable[[Image.Image], Image.Image]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_device(cfg: dict[str, Any]) -> torch.device:
    want = cfg.get("runtime", {}).get("device", "cuda")
    return torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def size_hw(cfg: dict[str, Any]) -> tuple[int, int]:
    s = cfg["preprocess"]["size_hw"]
    if not isinstance(s, (list, tuple)) or len(s) != 2:
        raise ValueError(f"preprocess.size_hw must be [H, W], got {s!r}")
    return int(s[0]), int(s[1])


def load_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    return read_split(resolve_path(cfg, "split_manifest"))


class Preprocessor:
    """register -> centred crop -> resize -> normalise. Shared by every split."""

    def __init__(self, cfg: dict[str, Any], registrar: Registrar | None) -> None:
        pp = cfg["preprocess"]
        if pp.get("register", False) and registrar is None:
            raise ValueError("config requires registration but no registrar was given")
        self.registrar = registrar if pp.get("register", False) else None
        self.crop_wh = pp.get("crop_wh")
        self.hw = size_hw(cfg)
        self.photometric = photometric.from_config(cfg)  # None unless configured
        self.mask_kw = mask_kwargs(cfg)

    def view(self, img: Image.Image) -> Image.Image:
        """The (corrected,) registered, cropped PIL image the model sees (before resize)."""
        if self.photometric is not None:
            img = self.photometric(img)
        if self.registrar is not None:
            img = self.registrar(img)
        return center_crop(img, self.crop_wh)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return to_tensor(self.view(img), self.hw)

    def with_mask(self, img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        """Model input plus the image's own board mask in the model frame (for the ROI)."""
        v = self.view(img)
        return to_tensor(v, self.hw), roi_mod.board_mask(v, self.hw, self.mask_kw)


def mask_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Board-mask options from `preprocess.board_mask` ({} = the original PCB1/PCB2 rule)."""
    bm = cfg["preprocess"].get("board_mask") or {}
    unknown = set(bm) - {"exclude_hue"}
    if unknown:
        raise ValueError(f"unknown preprocess.board_mask keys: {sorted(unknown)}")
    return {"exclude_hue": list(bm["exclude_hue"])} if bm.get("exclude_hue") is not None else {}


def pmap(cfg: dict[str, Any]) -> Callable:
    """Order-preserving map; threaded when runtime.num_workers > 0 (PIL releases the GIL)."""
    n = int(cfg.get("runtime", {}).get("num_workers", 0))
    if n <= 0:
        return map

    def _map(fn, items):
        items = list(items)
        with ThreadPoolExecutor(max_workers=n) as ex:
            for i in range(0, len(items), 64):  # bounded chunks keep memory flat
                yield from ex.map(fn, items[i:i + 64])
    return _map


def batches(cfg, paths: list[Path], pre: Preprocessor, perturb: Transform = None,
            with_masks: bool = False) -> Iterable:
    """Yield input batches (B, 3, H, W); with_masks: yield (batch, list of board masks)."""
    def one(p):
        im = load_rgb(p)
        im = perturb(im) if perturb is not None else im
        return pre.with_mask(im) if with_masks else (pre(im), None)
    bs = cfg["runtime"]["batch_size"]
    buf = []

    def emit():
        x = torch.stack([t for t, _ in buf])
        return (x, [m for _, m in buf]) if with_masks else x
    for item in pmap(cfg)(one, paths):
        buf.append(item)
        if len(buf) == bs:
            yield emit()
            buf = []
    if buf:
        yield emit()


def model_path(cfg) -> Path:
    return resolve_path(cfg, "artifacts") / "model.pt"


def freeze_path(cfg) -> Path:
    return resolve_path(cfg, "reports") / "frozen_operating_point.json"


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def write_scores(path: Path, rows: list[dict[str, str]], scores: np.ndarray,
                 info: dict[str, np.ndarray] | None = None) -> None:
    """Per-image scores; v2 runs add score_full, roi_coverage and roi_fallback columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    info = info or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image", "role", "label", "score"] + list(info))
        for i, (r, s) in enumerate(zip(rows, scores)):
            extra = [f"{int(v[i])}" if v.dtype == bool else f"{v[i]:.6f}" for v in info.values()]
            w.writerow([r["image"], r["role"], r["label"], f"{s:.6f}"] + extra)


def read_scores(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, np.array([float(r["score"]) for r in rows])


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit(cfg: dict[str, Any]) -> Path:
    set_determinism(cfg["seed"])
    device = get_device(cfg)
    hw = size_hw(cfg)
    rows = select(load_rows(cfg), "fit", "normal")
    if len(rows) != cfg["dataset"]["n_fit"]:
        raise ValueError(f"expected {cfg['dataset']['n_fit']} fit images, found {len(rows)}")
    root = resolve_path(cfg, "data_dir")
    paths = [root / r["image"] for r in rows]
    t0 = time.perf_counter()
    registrar = None
    if cfg["preprocess"].get("register", False):
        corr = photometric.from_config(cfg)  # the template must see what the model will see
        load = load_rgb if corr is None else (lambda q: corr(load_rgb(q)))
        registrar = Registrar.build_template(paths, load, pmap(cfg), mask_kwargs(cfg))  # FIT images only
    pre = Preprocessor(cfg, registrar)
    ext = build_extractor(cfg, device)
    acc = CovarianceAccumulator()
    use_roi = roi_mod.aggregation(cfg) == "roi_max"
    if use_roi and pre.registrar is None:
        raise ValueError("the board ROI requires registration (preprocess.register: true)")
    masks: list[np.ndarray] = []
    g1 = g2 = 0.0  # running grey-level moments of FIT inputs (ROI attached-structure rule)
    for b in batches(cfg, paths, pre, with_masks=use_roi):
        x, mk = b if use_roi else (b, [])
        check_batch(x, hw)
        acc.update(ext(x.to(device)))
        masks += mk
        if use_roi:
            g = roi_mod.gray_from_batch(x)
            g1, g2 = g1 + g.sum(0), g2 + (g * g).sum(0)
    m = cfg["model"]
    gm = GaussianPatchModel.from_accumulator(acc, m["cov_reg_rel"], m["cov_reg_eps"])
    state = {"gaussian": gm.state_dict(), "channel_idx": ext.channel_idx.cpu(),
             "template": None if registrar is None else torch.from_numpy(registrar.template),
             "config_hash": config_hash(cfg), "n_fit": acc.n,
             "fit_images": [r["image"] for r in rows]}
    if use_roi:  # v2: board ROI from FIT masks only, fixed rules (roi.py)
        rc = cfg["scoring"]["roi"]
        gray_std = None
        if "structure_k" in rc:  # Addendum A1
            n = len(masks)
            gray_std = (g2 / n - (g1 / n) ** 2).clamp_min(0).sqrt().numpy()
        info: dict[str, Any] = {}
        roi, _ = roi_mod.build_roi(masks, rc["vote_fraction"], rc["margin_px"], gray_std,
                                   rc.get("structure_k", 10.0), rc.get("structure_far_px", 48),
                                   rc.get("closing_px", 8), info=info)
        qc = {**roi_mod.roi_qc(roi, [roi_mod.coverage(mk, roi) for mk in masks], rc), "construction": info}
        state.update({"roi": torch.from_numpy(roi), "roi_qc": qc})
        print(f"fit: ROI area {qc['area_frac']:.3f}, FIT coverage pass {qc['fit_pass_frac']:.3f} -> {qc['status']}")
    out = model_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"fit: {acc.n} images, grid {gm.grid}, d={gm.mean.shape[1]}, "
          f"{time.perf_counter() - t0:.1f}s -> {out}")
    return out


@dataclass
class Detector:
    cfg: dict[str, Any]
    ext: PatchFeatureExtractor
    gm: GaussianPatchModel
    pre: Preprocessor
    roi: torch.Tensor | None = None  # (H, W) bool; None = v1 full-frame aggregation

    @property
    def device(self) -> torch.device:
        return self.gm.mean.device

    @torch.no_grad()
    def maps(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised batch (B, 3, H, W) -> smoothed anomaly maps (B, H, W)."""
        check_batch(x, self.pre.hw)
        d = self.gm.patch_distances(self.ext(x.to(self.device)))
        return anomaly_map(d, self.pre.hw, self.cfg["model"]["map_sigma"])

    def scores(self, amap: torch.Tensor, masks: list[np.ndarray] | None = None
               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Image scores (B,) from anomaly maps; v2 also returns score_full/roi_coverage/roi_fallback."""
        if self.roi is None:
            return image_score(amap), {}
        if masks is None or len(masks) != amap.shape[0]:
            raise ValueError("ROI scoring needs one board mask per image")
        roi_np = self.roi.cpu().numpy()
        cov = [roi_mod.coverage(m, roi_np) for m in masks]
        s, full, fb = roi_mod.aggregate(amap, self.roi, cov, self.cfg["scoring"]["roi"]["coverage_min"])
        return s, {"score_full": full, "roi_coverage": torch.tensor(cov, dtype=torch.float64),
                   "roi_fallback": fb}

    @torch.no_grad()
    def score_one(self, img: Image.Image) -> tuple[float, torch.Tensor, dict[str, Any]]:
        """End-to-end score of one PIL image: (score, anomaly map (H, W), extras)."""
        if self.roi is None:
            x, masks = self.pre(img), None
        else:
            x, mk = self.pre.with_mask(img)
            masks = [mk]
        amap = self.maps(x[None])
        s, info = self.scores(amap, masks)
        return float(s[0]), amap[0], {k: v[0].item() for k, v in info.items()}


def load_detector(cfg: dict[str, Any], device: torch.device | None = None) -> Detector:
    device = device or get_device(cfg)
    state = torch.load(model_path(cfg), map_location="cpu", weights_only=True)
    if state["config_hash"] != config_hash(cfg):
        raise RuntimeError("model.pt was fitted with a different config; re-run `ivad fit`")
    ext = build_extractor(cfg, device)
    if not torch.equal(ext.channel_idx.cpu(), state["channel_idx"]):
        raise RuntimeError("channel selection differs from the fitted model")
    tpl = state.get("template")
    registrar = Registrar(tpl.numpy(), mask_kwargs(cfg)) if tpl is not None else None
    roi = state.get("roi")
    if roi_mod.aggregation(cfg) == "roi_max":
        if roi is None:
            raise RuntimeError("config requires a board ROI but the fitted model has none; re-run `ivad fit`")
        rc = cfg["scoring"]["roi"]
        roi_mod.validate_roi(roi, size_hw(cfg), rc["min_area_frac"], rc["max_area_frac"])
    elif roi is not None:
        raise RuntimeError("fitted model carries a ROI but the config uses full-frame aggregation")
    return Detector(cfg, ext, GaussianPatchModel.from_state_dict(state["gaussian"]).to(device),
                    Preprocessor(cfg, registrar), roi)


@torch.no_grad()
def score_details(det: Detector, paths: list[Path], perturb: Transform = None, keep_maps: bool = False):
    """(scores (N,), maps (N, H, W) float16 or None, extras {name: (N,) array}; v1 extras are empty)."""
    use_roi = det.roi is not None
    scores, maps, extras = [], [], {}
    for b in batches(det.cfg, paths, det.pre, perturb, with_masks=use_roi):
        x, masks = b if use_roi else (b, None)
        amap = det.maps(x)
        s, info = det.scores(amap, masks)
        scores.append(s.cpu())
        for k, v in info.items():
            extras.setdefault(k, []).append(v.cpu())
        if keep_maps:
            maps.append(amap.half().cpu())
    s = torch.cat(scores).double().numpy()
    info = {k: torch.cat(v).numpy() for k, v in extras.items()}
    if "score_full" in info:
        info["score_full"] = info["score_full"].astype(np.float64)
    return s, (torch.cat(maps).numpy() if keep_maps else None), info


def score_images(det: Detector, paths: list[Path], perturb: Transform = None, keep_maps: bool = False):
    """Return image scores (N,) and optionally anomaly maps (N, H, W) as float16."""
    s, maps, _ = score_details(det, paths, perturb, keep_maps)
    return (s, maps) if keep_maps else s


def fallback_summary(info: dict[str, np.ndarray]) -> dict[str, Any] | None:
    if "roi_fallback" not in info:
        return None
    fb, cov = info["roi_fallback"].astype(bool), info["roi_coverage"]
    return {"n": int(len(fb)), "n_fallback": int(fb.sum()), "coverage_min": float(cov.min()),
            "coverage_median": float(np.median(cov))}


# --------------------------------------------------------------------------- #
# Calibrate and freeze
# --------------------------------------------------------------------------- #
def calibrate(cfg: dict[str, Any]) -> dict[str, Any]:
    set_determinism(cfg["seed"])
    device = get_device(cfg)
    rows = select(load_rows(cfg), "calib", "normal")
    if len(rows) != cfg["dataset"]["n_calib"]:
        raise ValueError(f"expected {cfg['dataset']['n_calib']} calib images, found {len(rows)}")
    det = load_detector(cfg, device)
    root = resolve_path(cfg, "data_dir")
    scores, _, info = score_details(det, [root / r["image"] for r in rows])
    reports = resolve_path(cfg, "reports")
    write_scores(reports / "scores_calib.csv", rows, scores, info)
    fb = fallback_summary(info)
    if fb is not None:
        limit = cfg["scoring"]["roi"]["calib_fallback_frac_max"]
        if fb["n_fallback"] > limit * fb["n"]:
            raise RuntimeError(f"ROI fail-safe triggered on {fb['n_fallback']}/{fb['n']} calibration normals "
                               f"(> {limit:.0%}): the ROI does not describe normal boards (protocol §8)")

    c = cfg["calibration"]
    thresholds = calibration.calibrate(scores, c["candidates"], c["level"])
    if c["primary"] not in thresholds:
        raise ValueError("primary threshold must be one of the candidates")
    wf = weights_file(cfg["model"]["backbone"]) if cfg["model"].get("pretrained", True) else None
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": "Threshold chosen from calibration normals only, before any official-test scoring.",
        "primary": c["primary"],
        "thresholds": thresholds,
        "calib_score_summary": {"n": len(scores), "min": float(scores.min()),
                                "median": float(np.median(scores)), "max": float(scores.max())},
        "config_hash": config_hash(cfg),
        "model_sha256": sha256_file(model_path(cfg)),
        "split_manifest_sha256": sha256_text(resolve_path(cfg, "split_manifest")),
        "pretrained_weights": {**weights_info(cfg["model"]["backbone"]),
                               "file_sha256": sha256_file(wf) if wf else None},
        "git_head_at_freeze": git_head(),
        "aggregation": roi_mod.aggregation(cfg),
    }
    if fb is not None:
        state = torch.load(model_path(cfg), map_location="cpu", weights_only=True)
        freeze["roi"] = {"calib_fallback": fb, "fit_qc": state["roi_qc"],
                         "mask_sha256": hashlib.sha256(state["roi"].numpy().tobytes()).hexdigest()}
    out = freeze_path(cfg)
    write_json(out, freeze)
    print(f"calibrate: primary {c['primary']} threshold = {thresholds[c['primary']]['threshold']:.4f} -> {out}")
    return freeze


def check_freeze(cfg: dict[str, Any]) -> dict[str, Any]:
    p = freeze_path(cfg)
    if not p.exists():
        raise RuntimeError("no frozen operating point: run `ivad calibrate` before evaluation")
    fz = json.loads(p.read_text(encoding="utf-8"))
    checks = {
        "config_hash": config_hash(cfg),
        "model_sha256": sha256_file(model_path(cfg)),
        "split_manifest_sha256": sha256_text(resolve_path(cfg, "split_manifest")),
    }
    bad = [k for k, v in checks.items() if fz[k] != v]
    if bad:
        raise RuntimeError(f"frozen operating point no longer matches: {bad}. "
                           "Changing the model or config after freezing invalidates the evaluation.")
    return fz


def check_committed(cfg: dict[str, Any]) -> str:
    """Refuse unless config, split manifest, freeze file and src/ are committed and unchanged.

    Returns the git HEAD. Enabled by `evaluation.require_committed_freeze` (PCB2 runs), so the
    exact frozen state that produced the test results can be recovered from git.
    """
    root = project_root(cfg).resolve()

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    files = [Path(cfg["_config_path"]), resolve_path(cfg, "split_manifest"), freeze_path(cfg)]
    for p in files:
        rel = p.resolve().relative_to(root).as_posix()
        if git("ls-files", "--error-unmatch", rel).returncode != 0:
            raise RuntimeError(f"{rel} is not committed; commit the frozen state before evaluation")
        if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            raise RuntimeError(f"{rel} differs from the committed version")
    if git("diff", "--quiet", "HEAD", "--", "src").returncode != 0:
        raise RuntimeError("src/ has uncommitted changes; evaluation code must be frozen")
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError("not a git repository")
    return head.stdout.strip()


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def _measure_latency(det, paths: list[Path], warmup: int) -> dict[str, Any]:
    """End-to-end per-image latency: decode + preprocess + forward + score, batch 1."""
    device = det.device
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for i, p in enumerate(paths[:warmup] + paths):
            sync()
            t0 = time.perf_counter()
            det.score_one(load_rgb(p))
            sync()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    return {"n_images": len(t), "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "definition": "decode JPEG + register + crop + resize + normalise + CNN forward + Mahalanobis + map + max"
                          + (" (ROI: + board mask + coverage)" if det.roi is not None else "")
                          + "; batch 1; after warm-up"}


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    fz = check_freeze(cfg)
    head = check_committed(cfg) if cfg["evaluation"].get("require_committed_freeze") else git_head()
    set_determinism(cfg["seed"])
    device = get_device(cfg)
    det = load_detector(cfg, device)
    root = resolve_path(cfg, "data_dir")
    reports = resolve_path(cfg, "reports")
    rows_all = load_rows(cfg)
    e = cfg["evaluation"]
    level = e["level"]

    results: dict[str, Any] = {"frozen_operating_point": {"primary": fz["primary"],
                               "frozen_at_utc": fz["frozen_at_utc"], "config_hash": fz["config_hash"]}}
    scored, fallbacks = {}, {}
    for role in ("test", "holdout"):
        rows = select(rows_all, role)
        s, _, info = score_details(det, [root / r["image"] for r in rows])
        write_scores(reports / f"scores_{role}.csv", rows, s, info)
        scored[role] = (rows, s)
        fallbacks[role] = fallback_summary(info)

    # Official benchmark: 100 normal + 100 anomaly
    rows, s = scored["test"]
    y = np.array([r["label"] == "anomaly" for r in rows])
    official = {
        "n_normal": int((~y).sum()), "n_anomaly": int(y.sum()),
        "auroc": metrics.bootstrap_auroc(y, s, e["n_bootstrap"], e["bootstrap_seed"], level),
        "operating_points": {name: metrics.operating_point(y, s, t["threshold"], level)
                             for name, t in fz["thresholds"].items()},
    }
    # Supplementary normal holdout: FPR only
    hrows, hs = scored["holdout"]
    holdout = {"n_normal": len(hs), "operating_points": {
        name: metrics.operating_point(np.zeros(len(hs), bool), hs, t["threshold"], level)
        for name, t in fz["thresholds"].items()}}
    # Descriptive pooled FPR over all 254 unseen normals (NOT the benchmark result)
    pooled_s = np.r_[s[~y], hs]
    pooled = {"n_normal": len(pooled_s), "note": "supplemental descriptive estimate; not the official benchmark",
              "operating_points": {name: metrics.operating_point(np.zeros(len(pooled_s), bool), pooled_s,
                                                                 t["threshold"], level)
                                   for name, t in fz["thresholds"].items()}}
    # Per-defect-type breakdown is unavailable at image level in the 1cls CSV;
    # we report score quantiles per class instead.
    results.update({
        "official_test": official,
        "supplementary_holdout": holdout,
        "pooled_unseen_normals": pooled,
        "score_quantiles": {
            "test_normal": np.quantile(s[~y], [0, .25, .5, .75, 1]).tolist(),
            "test_anomaly": np.quantile(s[y], [0, .25, .5, .75, 1]).tolist(),
            "holdout_normal": np.quantile(hs, [0, .25, .5, .75, 1]).tolist(),
        },
        "latency": _measure_latency(det, [root / r["image"] for r in rows], e["latency_warmup"]),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head_at_evaluation": head,
    })
    if fallbacks["test"] is not None:
        results["roi_fallback"] = fallbacks
    write_json(reports / "metrics.json", results)
    p = official["operating_points"][fz["primary"]]
    print(f"evaluate: AUROC {official['auroc']['value']:.4f}  recall {p['recall']['value']:.3f}  "
          f"FPR {p['fpr']['value']:.3f}  holdout FPR "
          f"{holdout['operating_points'][fz['primary']]['fpr']['value']:.3f}")
    return results


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def robustness(cfg: dict[str, Any]) -> dict[str, Any]:
    fz = check_freeze(cfg)
    set_determinism(cfg["seed"])
    device = get_device(cfg)
    det = load_detector(cfg, device)
    root = resolve_path(cfg, "data_dir")
    reports = resolve_path(cfg, "reports")
    t = fz["thresholds"][fz["primary"]]["threshold"]
    rb = cfg["robustness"]

    # Base scores come from `ivad evaluate` (same frozen model): test + holdout.
    trows, ts = read_scores(reports / "scores_test.csv")
    hrows, hs = read_scores(reports / "scores_holdout.csv")
    rows = trows + hrows
    base = np.r_[ts, hs]
    y = np.array([r["label"] == "anomaly" for r in rows])
    paths = [root / r["image"] for r in rows]

    out: dict[str, Any] = {"threshold": t, "primary": fz["primary"],
                           "images": {"anomaly": int(y.sum()), "normal": int((~y).sum())},
                           "normal_sources": "official test normals + supplementary holdout",
                           "perturbations": {}}
    per_image = {"image": [r["image"] for r in rows], "label": [r["label"] for r in rows], "base": base}
    for spec in rb["perturbations"]:
        s, _, info = score_details(det, paths, perturb=lambda im, sp=spec: apply_perturbation(im, sp))
        per_image[spec["name"]] = s
        fb, fp = base > t, s > t
        d = s - base
        out["perturbations"][spec["name"]] = {
            "spec": spec,
            "recall": metrics.rate_with_ci(int(fp[y].sum()), int(y.sum())),
            "fpr": metrics.rate_with_ci(int(fp[~y].sum()), int((~y).sum())),
            "paired": metrics.paired_bootstrap_delta(y, fb, fp, rb["n_bootstrap"], rb["bootstrap_seed"]),
            "score_change": {
                "normal_median_delta": float(np.median(d[~y])), "anomaly_median_delta": float(np.median(d[y])),
                "normal_frac_increased": float((d[~y] > 0).mean()),
                "anomaly_frac_increased": float((d[y] > 0).mean()),
            },
        }
        if fallback_summary(info) is not None:
            out["perturbations"][spec["name"]]["roi_fallback"] = fallback_summary(info)
        print(f"robustness {spec['name']}: recall {fp[y].mean():.3f} FPR {fp[~y].mean():.3f}")
    out["base"] = {"recall": metrics.rate_with_ci(int((base > t)[y].sum()), int(y.sum())),
                   "fpr": metrics.rate_with_ci(int((base > t)[~y].sum()), int((~y).sum()))}
    write_json(reports / "robustness.json", out)
    with open(reports / "robustness_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        keys = list(per_image)
        w.writerow(keys)
        for i in range(len(rows)):
            w.writerow([per_image[k][i] if isinstance(per_image[k][i], str) else f"{per_image[k][i]:.6f}"
                        for k in keys])
    return out
