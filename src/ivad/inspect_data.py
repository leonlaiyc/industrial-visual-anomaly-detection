"""Pre-modelling inspection of ~20 normal FIT images (never calib/holdout/test).

Produces a contact sheet, mean/std images, a resize-loss comparison and
quantitative alignment estimates (phase correlation against the mean image).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .align import Registrar, board_pose
from .config import write_json
from .data import select
from .features import center_crop, load_rgb


def registration_analysis(fit: list[str], data_root: Path, out_dir: Path, chosen: list[str],
                          crop_wh, pmap=map, mask_kw=None) -> dict[str, Any]:
    """Pose statistics over ALL fit normals and the registered mean/std image.

    Uses only FIT images (unlabelled normals), so it cannot leak test information.
    """
    paths = [data_root / p for p in fit]
    poses = list(pmap(lambda p: board_pose(load_rgb(p), mask_kw), paths))
    reg = Registrar.build_template(paths, load_rgb, pmap, mask_kw)
    infos = list(pmap(lambda p: _reg_small(reg, p), paths))
    stack = np.stack([i[0] for i in infos])
    flips = np.array([i[1]["flipped"] for i in infos])
    margin = np.array([i[1]["ncc"] - i[1]["ncc_other"] for i in infos])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].imshow(stack.mean(0), cmap="gray"); axes[0].set_title(f"Mean of {len(paths)} REGISTERED fit normals")
    im1 = axes[1].imshow(stack.std(0), cmap="magma"); axes[1].set_title("Pixel std after registration")
    for a in axes:
        a.axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "registered_mean_std_fit_normals.png", dpi=110)
    plt.close(fig)

    # Registered + cropped views (what the model sees) for 5 inspected images
    show = chosen[:5]
    fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 4.6))
    for j, p in enumerate(show):
        im = load_rgb(data_root / p)
        axes[0, j].imshow(im); axes[0, j].set_title(f"{Path(p).name} raw", fontsize=8)
        axes[1, j].imshow(center_crop(reg(im), crop_wh)); axes[1, j].set_title("registered + crop", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "registered_crops.png", dpi=110)
    plt.close(fig)

    def pct(v):
        return {q: float(np.percentile(v, q)) for q in (0, 5, 50, 95, 100)}
    return {
        "n_images": len(paths),
        "native_size_wh": list(load_rgb(paths[0]).size),
        "centroid_x_px": pct([p["cx"] for p in poses]),
        "centroid_y_px": pct([p["cy"] for p in poses]),
        "tilt_deg": pct([p["angle"] for p in poses]),
        "board_area_fraction": pct([p["area_frac"] for p in poses]),
        "rotated_180_count": int(flips.sum()),
        "orientation_ncc_margin_min": float(margin.min()),
    }


def _reg_small(reg: Registrar, p: Path):
    im, info = reg.register(load_rgb(p))
    w, h = im.size
    return np.asarray(im.convert("L").resize((w // 4, h // 4), Image.BILINEAR), np.float32), info


def run_inspection(split_rows, data_root: Path, out_dir: Path, seed: int, n: int = 20,
                   candidate_widths=(256, 384, 512, 768), crop_wh=None, pmap=map,
                   mask_kw=None) -> dict[str, Any]:
    fit = sorted(r["image"] for r in select(split_rows, "fit"))
    rng = np.random.default_rng(seed)
    chosen = [fit[i] for i in sorted(rng.choice(len(fit), size=n, replace=False))]
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [Image.open(data_root / p).convert("RGB") for p in chosen]
    sizes = sorted({im.size for im in imgs})

    # Contact sheet
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))
    for ax, im, p in zip(axes.ravel(), imgs, chosen):
        ax.imshow(im)
        ax.set_title(Path(p).name, fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(imgs):]:
        ax.axis("off")
    fig.suptitle(f"VisA {Path(chosen[0]).parts[0].upper()} — {n} normal FIT images (seeded sample)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet_fit_normals.png", dpi=110)
    plt.close(fig)

    # Unregistered mean/std of the 20 images (ghosting = pose variation).
    # NOTE: an earlier version estimated shifts by phase correlation against this
    # mean; that is biased to zero because the mean contains each image. Pose is
    # now measured directly from the board mask (registration_analysis).
    w0, h0 = imgs[0].size
    small = (w0 // 4, h0 // 4)
    g = np.stack([np.asarray(im.convert("L").resize(small, Image.BILINEAR), np.float32) for im in imgs])
    mean_img = g.mean(0)
    brightness = g.mean(axis=(1, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].imshow(mean_img, cmap="gray"); axes[0].set_title("Mean of 20 normals"); axes[0].axis("off")
    im1 = axes[1].imshow(g.std(0), cmap="magma"); axes[1].set_title("Pixel std (bright = variable)"); axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_std_fit_normals.png", dpi=110)
    plt.close(fig)

    # Resize information loss: a native centre crop vs. the same region after
    # downsampling the whole image to each candidate width.
    im = imgs[0]
    W, H = im.size
    cw, ch = W // 6, H // 6
    box = (W // 2 - cw // 2, H // 2 - ch // 2, W // 2 + cw // 2, H // 2 + ch // 2)
    fig, axes = plt.subplots(1, len(candidate_widths) + 1, figsize=(3 * (len(candidate_widths) + 1), 3))
    axes[0].imshow(im.crop(box)); axes[0].set_title(f"native {W}x{H}", fontsize=8); axes[0].axis("off")
    for ax, cwid in zip(axes[1:], candidate_widths):
        s = cwid / W
        small_im = im.resize((cwid, round(H * s)), Image.BILINEAR)
        sb = tuple(int(round(v * s)) for v in box)
        ax.imshow(small_im.crop(sb).resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST))
        ax.set_title(f"width {cwid} (x{1 / s:.1f} down)", fontsize=8); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "resize_loss_crops.png", dpi=110)
    plt.close(fig)

    reg = registration_analysis(fit, data_root, out_dir, chosen, crop_wh, pmap, mask_kw)

    report = {
        "registration_all_fit": reg,
        "images_inspected": chosen,
        "role": "fit",
        "native_sizes_wh": [list(s) for s in sizes],
        "brightness_mean_0_255": {"min": float(brightness.min()), "max": float(brightness.max()),
                                   "std": float(brightness.std())},
    }
    write_json(out_dir / "inspection_stats.json", report)
    return report
