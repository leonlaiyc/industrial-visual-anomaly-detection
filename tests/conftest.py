"""Synthetic 'board' images for fast, offline tests (no VisA download needed)."""
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw


def make_board(seed: int, size=(280, 200), anomaly=False, flip=False, dx=0, dy=0, tilt=0.0) -> Image.Image:
    """Blue board with two white 'sensors' and a pin header on a grey background."""
    rng = np.random.default_rng(seed)
    w, h = size
    canvas = Image.new("RGB", (w * 2, h * 2), (95, 95, 95))
    d = ImageDraw.Draw(canvas)
    cx, cy = w, h
    d.rectangle([cx - 80, cy - 35, cx + 80, cy + 35], fill=(30, 110, 190))
    for ox in (-45, 45):
        d.ellipse([cx + ox - 25, cy - 25, cx + ox + 25, cy + 25], fill=(235, 235, 235))
    for px in range(-12, 13, 8):  # pins above the board: breaks 180° symmetry
        d.rectangle([cx + px - 2, cy - 55, cx + px + 2, cy - 35], fill=(200, 200, 205))
    if anomaly:
        d.ellipse([cx - 10, cy + 10, cx + 2, cy + 22], fill=(220, 30, 30))
    img = canvas.rotate(tilt + (180 if flip else 0), resample=Image.BILINEAR, center=(cx, cy),
                        fillcolor=(95, 95, 95))
    img = img.crop((cx - w // 2 - dx, cy - h // 2 - dy, cx + w // 2 - dx, cy + h // 2 - dy))
    a = np.asarray(img).astype(np.float32) + rng.normal(0, 3, (h, w, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@pytest.fixture
def board():
    return make_board
