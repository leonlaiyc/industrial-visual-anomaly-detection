"""Board-ROI aggregation (v2): mask geometry, score aggregation, invalid masks, determinism."""
import numpy as np
import pytest
import torch

from ivad import roi as R
from ivad.align import Registrar
from ivad.features import center_crop

from .conftest import make_board

H, W = 40, 60


def rect(y0, y1, x0, x1, shape=(H, W)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


RC = {"vote_fraction": 0.5, "margin_px": 2, "coverage_min": 0.95, "fit_pass_frac_min": 0.99,
      "min_area_frac": 0.05, "max_area_frac": 0.9, "calib_fallback_frac_max": 0.05}


# --------------------------------------------------------------------------- #
# Mask geometry
# --------------------------------------------------------------------------- #
def test_build_roi_majority_vote_then_margin():
    masks = [rect(10, 30, 15, 45)] * 3 + [rect(10, 30, 15, 50)]  # 1 of 4 extends right: below 50% vote
    roi, freq = R.build_roi(masks, vote_fraction=0.5, margin_px=0)
    assert np.array_equal(roi, rect(10, 30, 15, 45))
    assert freq.max() == 1.0 and freq[20, 47] == 0.25
    roi2, _ = R.build_roi(masks, vote_fraction=0.5, margin_px=3)
    assert roi2[20, 47] and not roi2[20, 49]  # 3 px margin to the right of column 44
    assert roi2[7, 30] and not roi2[6, 30]  # and above row 10
    assert (roi2 >= roi).all()  # dilation only adds


def test_build_roi_keeps_largest_component_and_fills_holes():
    board = rect(10, 30, 10, 40)
    board[18:22, 20:24] = False  # e.g. a hole in the colour mask (dark component)
    speck = rect(2, 4, 55, 58)  # isolated background blob present in every mask
    roi, _ = R.build_roi([board | speck] * 5, 0.5, 0)
    assert roi[20, 22]  # hole filled
    assert not roi[3, 56]  # off-board blob dropped
    assert R.coverage(board, roi) == 1.0


def test_build_roi_rejects_bad_input():
    with pytest.raises(ValueError):
        R.build_roi([], 0.5, 2)
    with pytest.raises(ValueError):
        R.build_roi([rect(1, 5, 1, 5), rect(1, 5, 1, 5, shape=(H + 1, W))], 0.5, 2)
    with pytest.raises(ValueError):
        R.build_roi([rect(1, 5, 1, 5)], 0.0, 2)
    with pytest.raises(ValueError):
        R.build_roi([rect(1, 5, 1, 5)], 0.5, -1)


def test_coverage():
    roi = rect(10, 30, 10, 40)
    assert R.coverage(rect(12, 28, 12, 38), roi) == 1.0
    half = rect(10, 30, 30, 50)  # 10 of 20 columns inside
    assert R.coverage(half, roi) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="empty"):
        R.coverage(np.zeros((H, W), bool), roi)
    with pytest.raises(ValueError, match="shape"):
        R.coverage(rect(1, 3, 1, 3, shape=(H, W + 1)), roi)


def test_roi_aligns_with_model_frame_on_registered_boards():
    """The ROI built from registered synthetic boards (shifted, tilted, flipped) covers each board
    and excludes most background, in the same frame as the anomaly map."""
    imgs = [make_board(i, flip=i % 3 == 0, dx=(i % 5) * 4 - 8, dy=(i % 4) * 3 - 4, tilt=(i % 3) - 1.0)
            for i in range(12)]
    reg = Registrar.build_template(list(range(12)), lambda i: imgs[i])
    hw = (80, 120)
    masks = [R.board_mask(center_crop(reg(im), (240, 160)), hw) for im in imgs]
    assert all(m.shape == hw for m in masks)
    roi, _ = R.build_roi(masks, 0.5, 2)
    R.validate_roi(roi, hw, 0.05, 0.9)
    cov = [R.coverage(m, roi) for m in masks]
    assert min(cov) >= 0.95
    assert roi.mean() < 0.6  # background excluded
    # a deliberately mis-registered (unregistered, shifted) board is detected by low coverage
    bad = R.board_mask(center_crop(make_board(99, dx=40, dy=25), (240, 160)), hw)
    assert R.coverage(bad, roi) < 0.95


# --------------------------------------------------------------------------- #
# Invalid masks cannot silently produce scores
# --------------------------------------------------------------------------- #
def test_validate_roi_rejects_invalid_masks():
    good = rect(10, 30, 10, 40)
    R.validate_roi(good, (H, W), 0.05, 0.9)
    R.validate_roi(torch.from_numpy(good), (H, W))
    with pytest.raises(ValueError, match="empty"):
        R.validate_roi(np.zeros((H, W), bool), (H, W))
    with pytest.raises(ValueError, match="boolean"):
        R.validate_roi(good.astype(np.uint8), (H, W))
    with pytest.raises(ValueError, match="shape"):
        R.validate_roi(good, (H, W + 2))
    with pytest.raises(ValueError, match="connected"):
        R.validate_roi(good | rect(0, 2, 55, 58), (H, W))
    with pytest.raises(ValueError, match="area"):
        R.validate_roi(rect(0, H, 0, W), (H, W), 0.05, 0.9)


def test_roi_max_rejects_invalid_roi():
    amap = torch.rand(2, H, W)
    with pytest.raises(ValueError, match="empty"):
        R.roi_max(amap, torch.zeros(H, W, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        R.roi_max(amap, torch.ones(H, W))
    with pytest.raises(ValueError, match="match"):
        R.roi_max(amap, torch.ones(H, W + 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="match"):
        R.roi_max(amap[0], torch.ones(H, W, dtype=torch.bool))


def test_roi_qc_gates():
    roi = rect(10, 30, 10, 40)
    qc = R.roi_qc(roi, [1.0] * 99 + [0.97], RC)
    assert qc["status"] == "PASS" and qc["fit_pass_frac"] == 1.0
    with pytest.raises(ValueError, match="QC failed"):
        R.roi_qc(roi, [1.0] * 97 + [0.5] * 3, RC)  # 97% < 99% required
    with pytest.raises(ValueError, match="area"):
        R.roi_qc(rect(0, H, 0, W), [1.0] * 10, RC)


def test_aggregation_setting():
    assert R.aggregation({}) == "full_max"  # v1 configs have no scoring section
    assert R.aggregation({"scoring": {"aggregation": "roi_max"}}) == "roi_max"
    with pytest.raises(ValueError):
        R.aggregation({"scoring": {"aggregation": "mean"}})


# --------------------------------------------------------------------------- #
# Score aggregation
# --------------------------------------------------------------------------- #
def test_roi_max_ignores_background_peak():
    roi = torch.from_numpy(rect(10, 30, 10, 40))
    amap = torch.ones(3, H, W)
    amap[0, 2, 2] = 50.0  # background dust: v1 full max = 50, v2 ignores it
    amap[1, 20, 20] = 9.0  # on-board defect: both see it
    amap[2, 10, 10] = 7.0  # ROI corner pixel is inside
    full = amap.flatten(1).amax(1)
    v2 = R.roi_max(amap, roi)
    assert full.tolist() == [50.0, 9.0, 7.0]
    assert v2.tolist() == [1.0, 9.0, 7.0]
    assert (v2 <= full).all()


def test_aggregate_fallback_is_fail_safe():
    roi = torch.from_numpy(rect(10, 30, 10, 40))
    amap = torch.ones(3, H, W)
    amap[:, 2, 2] = 50.0
    amap[:, 20, 20] = 5.0
    s, full, fb = R.aggregate(amap, roi, [1.0, 0.95, 0.80], coverage_min=0.95)
    assert fb.tolist() == [False, False, True]  # 0.95 is accepted, 0.80 falls back
    assert s.tolist() == [5.0, 5.0, 50.0]
    assert torch.equal(full, torch.full((3,), 50.0))
    assert (s <= full).all() and (s >= R.roi_max(amap, roi)).all()


def test_aggregate_rejects_bad_coverage():
    roi = torch.from_numpy(rect(10, 30, 10, 40))
    amap = torch.rand(2, H, W)
    with pytest.raises(ValueError):
        R.aggregate(amap, roi, [1.0], 0.95)  # one value per image
    with pytest.raises(ValueError):
        R.aggregate(amap, roi, [1.0, float("nan")], 0.95)


def test_aggregation_is_deterministic():
    g = torch.Generator().manual_seed(0)
    amap = torch.rand(4, H, W, generator=g)
    roi = torch.from_numpy(rect(10, 30, 10, 40))
    cov = [1.0, 0.9, 0.99, 0.5]
    a = R.aggregate(amap, roi, cov, 0.95)
    b = R.aggregate(amap.clone(), roi.clone(), list(cov), 0.95)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    masks = [rect(10, 30, 10, 40)] * 3 + [rect(12, 31, 9, 41)]
    r1, f1 = R.build_roi(masks, 0.5, 2)
    r2, f2 = R.build_roi(list(reversed(masks)), 0.5, 2)  # order-independent
    assert np.array_equal(r1, r2) and np.array_equal(f1, f2)


# --------------------------------------------------------------------------- #
# Addendum A1: parts fixed to the board but missed by the colour mask
# --------------------------------------------------------------------------- #
def _pins_scene():
    core = rect(15, 35, 10, 50)  # board consensus from the colour mask
    std = np.full((H, W), 5.0) + np.random.default_rng(0).normal(0, 0.3, (H, W))
    std[0:15, 26:34] = 40.0  # header pins: above the board, touching it and the image border
    std[2:5, 2:5] = 40.0  # isolated high-variance blob on the cloth (not attached)
    std[15:35, 10:50] = 30.0  # board edges/features
    return core, std


def test_attached_structure_adds_pins_not_detached_blobs():
    core, std = _pins_scene()
    obj, st = R.attached_structure(core, std, k=10, far_px=6)
    assert obj[5, 30] and obj[0, 30]  # pins kept, up to the image border
    assert not obj[3, 3]  # detached blob ignored
    assert 4.5 < st["cloth_std_median"] < 5.5 and st["std_threshold"] < 30


def test_build_roi_with_structure_keeps_border_pins_and_margin():
    core, std = _pins_scene()
    info = {}
    roi, _ = R.build_roi([core] * 4, 0.5, 2, gray_std=std, structure_k=10, structure_far_px=6,
                         closing_px=3, info=info)
    assert roi[0, 30] and roi[5, 25]  # pins (incl. border row) and their 2 px margin
    assert not roi[3, 3] and not roi[5, 20]
    assert info["object_area_frac"] > info["core_area_frac"]
    R.validate_roi(roi, (H, W), 0.05, 0.9)
    roi0, _ = R.build_roi([core] * 4, 0.5, 2)  # original rule: pins outside the ROI
    assert not roi0[5, 30]


def test_gray_from_batch_matches_itu601():
    rgb = torch.tensor([30.0, 110.0, 190.0]).view(1, 3, 1, 1) / 255.0
    x = (rgb - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)) / torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    g = R.gray_from_batch(x)
    assert g.shape == (1, 1, 1)
    assert float(g) == pytest.approx(0.299 * 30 + 0.587 * 110 + 0.114 * 190, abs=1e-3)
