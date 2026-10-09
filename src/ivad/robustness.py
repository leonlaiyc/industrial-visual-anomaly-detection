"""Controlled, modest image perturbations applied at native resolution.

Each perturbation mimics a plausible acquisition drift. They are applied to
the original image *before* the standard preprocessing so the model sees them
exactly as it would see a drifted camera frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_perturbation(img: Image.Image, spec: dict[str, Any]) -> Image.Image:
    kind = spec["kind"]
    if kind == "photometric":
        out = ImageEnhance.Brightness(img).enhance(float(spec["brightness"]))
        return ImageEnhance.Contrast(out).enhance(float(spec["contrast"]))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(spec["sigma_px"])))
    if kind == "shift":
        return shift_image(img, int(spec["dx_px"]), int(spec["dy_px"]))
    raise ValueError(f"unknown perturbation kind {kind!r}")


def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Translate content by (dx, dy) px (positive = right/down), replicating edges.

    Edge replication avoids introducing black borders, which would be an
    artificial anomaly rather than a camera-position change.
    """
    a = np.asarray(img)
    h, w = a.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        raise ValueError("shift larger than the image")
    pad = ((abs(dy), abs(dy)), (abs(dx), abs(dx))) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad, mode="edge")
    y0, x0 = abs(dy) - dy, abs(dx) - dx
    return Image.fromarray(np.ascontiguousarray(p[y0:y0 + h, x0:x0 + w]))
