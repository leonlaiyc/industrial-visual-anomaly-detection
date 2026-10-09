"""Global luminance affine standardisation (PCB3 illumination experiment, candidate only).

Pre-registered in docs/illumination_experiment_protocol.md §3. For a native RGB
image I:

    Y = luminance (PIL "L", ITU-R 601) over the whole frame;  mu = mean(Y), sigma = std(Y)
    I' = clip(round((I - mu) * sigma_ref / sigma + mu_ref), 0, 255)   (same gain on R, G, B)

mu_ref and sigma_ref are the medians of the per-image mu and sigma over the FIT
normals, computed once (`fit_reference`) and frozen in the config. One gain for
all channels keeps colour-type defects; global statistics are barely moved by
a local defect. The step runs before registration, so the board mask and the
features see the corrected image. Configs without `preprocess.photometric`
are unaffected (v1/v2 behaviour, identical config hash).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np
from PIL import Image

METHOD = "global_luma_affine"


def luma_stats(img: Image.Image) -> tuple[float, float]:
    """(mean, std) of the luminance over the whole frame."""
    y = np.asarray(img.convert("L"), dtype=np.float64)
    return float(y.mean()), float(y.std())


def standardise(img: Image.Image, ref_mean: float, ref_std: float) -> Image.Image:
    """Map the image's global luminance mean/std to (ref_mean, ref_std); one gain for R, G, B."""
    if ref_std <= 0:
        raise ValueError("ref_std must be positive")
    mu, sd = luma_stats(img)
    if sd < 1e-6:
        raise ValueError("image has no luminance variation; cannot standardise")
    gain = ref_std / sd
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    out = (a - mu) * gain + ref_mean
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def from_config(cfg: dict[str, Any]) -> Optional[Callable[[Image.Image], Image.Image]]:
    """The configured correction as an image -> image function, or None (baseline)."""
    ph = cfg.get("preprocess", {}).get("photometric")
    if not ph:
        return None
    if ph.get("method") != METHOD:
        raise ValueError(f"unknown photometric method {ph.get('method')!r}")
    if ph.get("ref_mean") is None or ph.get("ref_std") is None:
        raise ValueError("photometric correction needs frozen ref_mean and ref_std (run `ivad photostats`)")
    m, s = float(ph["ref_mean"]), float(ph["ref_std"])
    return lambda img: standardise(img, m, s)


def fit_reference(images: Iterable[Image.Image]) -> dict[str, Any]:
    """Reference values = medians of per-image luminance mean/std (FIT normals only)."""
    stats = np.array([luma_stats(im) for im in images])
    if len(stats) == 0:
        raise ValueError("need at least one image")
    mu, sd = stats[:, 0], stats[:, 1]

    def summ(v):
        return {"min": float(v.min()), "p05": float(np.quantile(v, .05)), "median": float(np.median(v)),
                "p95": float(np.quantile(v, .95)), "max": float(v.max()), "std": float(v.std())}
    return {"method": METHOD, "n_images": int(len(stats)),
            "ref_mean": float(np.median(mu)), "ref_std": float(np.median(sd)),
            "per_image_mean": summ(mu), "per_image_std": summ(sd)}
