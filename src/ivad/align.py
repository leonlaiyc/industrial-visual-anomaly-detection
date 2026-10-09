"""Board registration: undo placement translation, small tilt and 180° flips.

Why: inspection of FIT normals showed the PCB is not in a fixed pose —
~19% of boards are rotated 180°, centroids vary by about ±70 px and tilt by
about ±3° (see reports/inspection/). A per-position Gaussian model needs
position-consistent patches, so every image (all splits, identically) is
registered before feature extraction:

1. Foreground mask on a 1/4-scale copy: saturated (blue PCB) or bright
   (white sensors) pixels -> morphological clean-up -> largest component.
2. Centroid and principal-axis angle from the mask's image moments.
3. Rotate about the centroid by -angle and translate the centroid to the
   image centre (bilinear; uncovered borders filled with the median
   background colour).
4. Moments are ambiguous by 180°; choose 0° or 180° by normalised
   correlation with a canonical template = mean of registered FIT images.

The template is built from FIT images only and stored with the model.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

DOWN = 4  # mask / template computed at 1/4 resolution


def foreground_mask(img: Image.Image, sat_min: int = 90, val_min: int = 170) -> np.ndarray:
    """Board mask at 1/DOWN scale: saturated or bright pixels, cleaned, largest component."""
    w, h = img.size
    small = img.resize((w // DOWN, h // DOWN), Image.BILINEAR)
    hsv = np.asarray(small.convert("HSV"), dtype=np.uint8)
    m = (hsv[..., 1] > sat_min) | (hsv[..., 2] > val_min)
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3))
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("no foreground found; cannot register image")
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def board_pose(img: Image.Image) -> dict[str, float]:
    """Centroid (native px) and principal-axis angle (deg, in (-90, 90]) of the board."""
    m = foreground_mask(img)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, 1]
    angle = math.degrees(math.atan2(vy, vx))
    angle = (angle + 90.0) % 180.0 - 90.0  # principal axis is a line: fold to (-90, 90]
    return {"cx": (cx + 0.5) * DOWN - 0.5, "cy": (cy + 0.5) * DOWN - 0.5, "angle": angle,
            "area_frac": float(m.mean()), "axis_ratio": float(math.sqrt(evals[1] / max(evals[0], 1e-9)))}


def _background_fill(img: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(img)
    border = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(border, axis=0))


def warp(img: Image.Image, pose: dict[str, float], flip: bool, fill=None) -> Image.Image:
    """Rotate about the board centroid, move it to the image centre, optional 180°."""
    w, h = img.size
    a = math.radians(pose["angle"] + (180.0 if flip else 0.0))
    ox, oy = (w - 1) / 2, (h - 1) / 2
    # PIL's affine maps OUTPUT (x, y) -> INPUT; output = R(-a)(input - c) + o
    # so input = R(a)(output - o) + c
    ca, sa = math.cos(a), math.sin(a)
    coeffs = (ca, -sa, pose["cx"] - ca * ox + sa * oy,
              sa, ca, pose["cy"] - sa * ox - ca * oy)
    return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR,
                         fillcolor=_background_fill(img) if fill is None else fill)


def _gray_small(img: Image.Image) -> np.ndarray:
    w, h = img.size
    g = np.asarray(img.convert("L").resize((w // DOWN, h // DOWN), Image.BILINEAR), np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).mean())


class Registrar:
    """Holds the canonical template; `__call__` registers one PIL image."""

    def __init__(self, template: np.ndarray | None = None) -> None:
        self.template = template

    def register(self, img: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        if self.template is None:
            raise RuntimeError("template not built; call build_template first")
        pose = board_pose(img)
        cand = [warp(img, pose, flip=False), warp(img, pose, flip=True)]
        scores = [ncc(_gray_small(c), self.template) for c in cand]
        k = int(np.argmax(scores))
        return cand[k], {**pose, "flipped": bool(k), "ncc": scores[k], "ncc_other": scores[1 - k]}

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.register(img)[0]

    @classmethod
    def build_template(cls, paths: Sequence, load: Callable[[Any], Image.Image],
                       pmap: Callable = map) -> "Registrar":
        """Template from FIT images: seed orientation = first image (sorted order),
        then template = mean of all FIT images registered against that seed.

        `pmap` may be a parallel, order-preserving map; the sum is order-independent
        up to float round-off.
        """
        if len(paths) == 0:
            raise ValueError("need at least one image")
        first = load(paths[0])
        seed = cls(_gray_small(warp(first, board_pose(first), flip=False)))
        acc = sum(pmap(lambda p: _gray_small(seed(load(p))).astype(np.float64), paths))
        acc = (acc / len(paths)).astype(np.float32)
        return cls((acc - acc.mean()) / (acc.std() + 1e-6))
