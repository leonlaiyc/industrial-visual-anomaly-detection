"""Preprocessing, registration, perturbations and feature-extractor shapes."""
import numpy as np
import pytest
import torch
from PIL import Image

from ivad.align import Registrar, board_pose, ncc, _gray_small
from ivad.features import PatchFeatureExtractor, center_crop, check_batch, to_tensor
from ivad.robustness import apply_perturbation, shift_image


def test_to_tensor_shape_and_normalisation():
    img = Image.new("RGB", (40, 30), (124, 116, 104))  # ~ImageNet mean colour
    t = to_tensor(img, (16, 24))
    assert t.shape == (3, 16, 24) and t.abs().max() < 0.02
    with pytest.raises(ValueError):
        to_tensor(img.convert("L"), (16, 24))
    with pytest.raises(ValueError):
        to_tensor(img, (0, 24))


def test_check_batch_rejects_bad_inputs():
    check_batch(torch.zeros(2, 3, 8, 8), (8, 8))
    for bad in (torch.zeros(3, 8, 8), torch.zeros(1, 1, 8, 8), torch.zeros(1, 3, 8, 9)):
        with pytest.raises(ValueError):
            check_batch(bad, (8, 8))
    x = torch.zeros(1, 3, 8, 8)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        check_batch(x, (8, 8))


def test_center_crop():
    img = Image.new("RGB", (100, 80))
    assert center_crop(img, [60, 40]).size == (60, 40)
    assert center_crop(img, None).size == (100, 80)
    with pytest.raises(ValueError):
        center_crop(img, [120, 40])


def test_feature_extractor_shape_and_seeded_channels():
    a = PatchFeatureExtractor("resnet18", ["layer2", "layer3"], 20, feature_seed=5, pretrained=False)
    b = PatchFeatureExtractor("resnet18", ["layer3", "layer2"], 20, feature_seed=5, pretrained=False)
    c = PatchFeatureExtractor("resnet18", ["layer2", "layer3"], 20, feature_seed=6, pretrained=False)
    assert torch.equal(a.channel_idx, b.channel_idx) and not torch.equal(a.channel_idx, c.channel_idx)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 96)
    f = a(x)
    assert f.shape == (2, 20, 8, 12)  # stride 8 of layer2
    assert torch.equal(f, a(x))  # deterministic forward
    with pytest.raises(ValueError):
        PatchFeatureExtractor("resnet18", ["layer2"], 1000, 0, pretrained=False)
    with pytest.raises(ValueError):
        PatchFeatureExtractor("vgg16", ["layer2"], 10, 0, pretrained=False)


def test_board_pose_recovers_translation_and_tilt(board):
    ref = board_pose(board(0))
    p = board_pose(board(0, dx=20, dy=-10, tilt=3.0))
    assert p["cx"] - ref["cx"] == pytest.approx(20, abs=3)
    assert p["cy"] - ref["cy"] == pytest.approx(-10, abs=3)
    # PIL rotates counter-clockwise; image y points down -> angle is negative
    assert p["angle"] - ref["angle"] == pytest.approx(-3.0, abs=1.0)


def test_registration_undoes_flip_shift_and_tilt(board):
    imgs = [board(i, dx=int(5 * np.sin(i)), dy=int(4 * np.cos(i))) for i in range(6)]
    reg = Registrar.build_template(imgs, load=lambda im: im)
    canon = _gray_small(reg(board(10)))
    moved, info = reg.register(board(11, flip=True, dx=18, dy=-9, tilt=2.5))
    assert info["flipped"] is True
    assert ncc(_gray_small(moved), canon) > 0.85
    _, info2 = reg.register(board(12, dx=-15, dy=6, tilt=-2.0))
    assert info2["flipped"] is False


def test_shift_image_direction_and_edges():
    a = np.zeros((5, 6, 3), np.uint8)
    a[2, 2] = 255
    out = np.asarray(shift_image(Image.fromarray(a), dx=1, dy=-1))
    assert out[1, 3, 0] == 255 and out.sum() == 255 * 3
    with pytest.raises(ValueError):
        shift_image(Image.fromarray(a), dx=10, dy=0)


def test_perturbations_preserve_size_and_are_deterministic(board):
    img = board(0)
    for spec in ({"kind": "photometric", "brightness": 0.9, "contrast": 0.9},
                 {"kind": "blur", "sigma_px": 1.5}, {"kind": "shift", "dx_px": 4, "dy_px": 3}):
        a, b = apply_perturbation(img, spec), apply_perturbation(img, spec)
        assert a.size == img.size and np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        apply_perturbation(img, {"kind": "rain"})
