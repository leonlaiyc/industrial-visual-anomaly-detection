"""Image preprocessing and pretrained-CNN patch feature extraction.

Pipeline (identical for every split):

    RGB image --register (align.py)--> centred crop --resize (H, W)--> ImageNet normalisation
      --frozen torchvision ResNet--> intermediate maps of the configured layers
      --upsample deeper maps to the first layer's grid--> concatenate channels
      --keep a fixed random subset of d channels--> 3x3 local average
      --> patch features of shape (B, d, h, w)

Only torchvision components are used here; all statistics live in gaussian.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_SUPPORTED = ("resnet18", "resnet50", "wide_resnet50_2")
_STAGE_ORDER = ("layer1", "layer2", "layer3", "layer4")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def center_crop(img: Image.Image, crop_wh: Sequence[int] | None) -> Image.Image:
    """Centred crop (native px). After registration the board sits at the centre."""
    if not crop_wh:
        return img
    cw, ch = int(crop_wh[0]), int(crop_wh[1])
    w, h = img.size
    if cw > w or ch > h or cw <= 0 or ch <= 0:
        raise ValueError(f"crop {crop_wh} does not fit image {img.size}")
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def to_tensor(img: Image.Image, size_hw: Sequence[int]) -> torch.Tensor:
    """PIL RGB -> normalised float tensor (3, H, W). Bilinear (antialiased) resize."""
    if img.mode != "RGB":
        raise ValueError(f"expected RGB image, got mode {img.mode}")
    h, w = int(size_hw[0]), int(size_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid target size {size_hw}")
    arr = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def check_batch(x: torch.Tensor, size_hw: Sequence[int]) -> None:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected batch (B, 3, H, W), got {tuple(x.shape)}")
    if tuple(x.shape[2:]) != (int(size_hw[0]), int(size_hw[1])):
        raise ValueError(f"expected spatial size {tuple(size_hw)}, got {tuple(x.shape[2:])}")
    if not torch.isfinite(x).all():
        raise ValueError("input batch contains NaN or Inf")


# --------------------------------------------------------------------------- #
# Feature extractor
# --------------------------------------------------------------------------- #
class PatchFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        n_features: int,
        feature_seed: int,
        pool_kernel: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in _SUPPORTED:
            raise ValueError(f"backbone must be one of {_SUPPORTED}")
        if not layers or any(l not in _STAGE_ORDER for l in layers):
            raise ValueError(f"layers must be a non-empty subset of {_STAGE_ORDER}")
        self.layers = sorted(layers, key=_STAGE_ORDER.index)
        weights = "IMAGENET1K_V1" if pretrained else None
        net = torchvision.models.get_model(backbone, weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        last = _STAGE_ORDER.index(self.layers[-1])
        self.stages = nn.ModuleDict({n: getattr(net, n) for n in _STAGE_ORDER[: last + 1]})
        self.eval().requires_grad_(False)

        total = self._total_channels()
        if not 0 < n_features <= total:
            raise ValueError(f"n_features must be in (0, {total}]")
        g = torch.Generator().manual_seed(int(feature_seed))
        idx = torch.randperm(total, generator=g)[:n_features].sort().values
        self.register_buffer("channel_idx", idx, persistent=True)
        self.pool_kernel = int(pool_kernel)

    def _total_channels(self) -> int:
        with torch.no_grad():
            maps = self._raw_maps(torch.zeros(1, 3, 64, 64))
        return sum(m.shape[1] for m in maps)

    def _raw_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        out, h = [], self.stem(x)
        for name, stage in self.stages.items():
            h = stage(h)
            if name in self.layers:
                out.append(h)
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self._raw_maps(x)
        size = maps[0].shape[-2:]
        maps = [maps[0]] + [
            F.interpolate(m, size=size, mode="bilinear", align_corners=False) for m in maps[1:]
        ]
        feats = torch.cat(maps, dim=1).index_select(1, self.channel_idx)
        if self.pool_kernel > 1:
            k = self.pool_kernel
            feats = F.avg_pool2d(feats, k, stride=1, padding=k // 2, count_include_pad=False)
        return feats


def build_extractor(cfg: dict[str, Any], device: torch.device) -> PatchFeatureExtractor:
    m = cfg["model"]
    return PatchFeatureExtractor(
        backbone=m["backbone"],
        layers=m["layers"],
        n_features=m["n_features"],
        feature_seed=m["feature_seed"],
        pool_kernel=m.get("pool_kernel", 3),
        pretrained=m.get("pretrained", True),
    ).to(device)


def weights_file(backbone: str) -> Path | None:
    """Local cache path of the torchvision checkpoint (for provenance hashing)."""
    w = torchvision.models.get_model_weights(backbone).IMAGENET1K_V1
    p = Path(torch.hub.get_dir()) / "checkpoints" / Path(w.url).name
    return p if p.exists() else None


def weights_info(backbone: str) -> dict[str, Any]:
    w = torchvision.models.get_model_weights(backbone).IMAGENET1K_V1
    return {"enum": f"{type(w).__name__}.IMAGENET1K_V1", "url": w.url,
            "recipe": w.meta.get("recipe"), "torchvision": torchvision.__version__}
