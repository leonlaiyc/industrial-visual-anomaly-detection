"""Numerical correctness of the in-project Gaussian / Mahalanobis code.

Reference implementations: numpy.cov and scipy.spatial.distance.mahalanobis.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

from ivad.gaussian import (CovarianceAccumulator, GaussianPatchModel, anomaly_map, image_score,
                           mahalanobis, regularize, whitening_matrices)


def _feats(rng, b, d, h, w, offset=0.0):
    return torch.from_numpy(rng.normal(size=(b, d, h, w)) + offset)


def _per_position(x):  # (N, d, h, w) -> (P, N, d)
    n, d, h, w = x.shape
    return x.permute(2, 3, 0, 1).reshape(h * w, n, d).numpy()


@pytest.mark.parametrize("offset", [0.0, 1e4])  # large offset probes cancellation
def test_streaming_moments_match_numpy(offset):
    rng = np.random.default_rng(0)
    x = _feats(rng, 37, 5, 3, 4, offset)
    acc = CovarianceAccumulator()
    for chunk in torch.split(x, [10, 1, 26]):  # uneven batches
        acc.update(chunk)
    mean, cov = acc.finalize()
    xp = _per_position(x)
    for p in range(xp.shape[0]):
        np.testing.assert_allclose(mean[p].numpy(), xp[p].mean(0), rtol=0, atol=1e-9 * max(1, offset))
        np.testing.assert_allclose(cov[p].numpy(), np.cov(xp[p], rowvar=False), rtol=1e-9, atol=1e-9)
    assert acc.n == 37


def test_mahalanobis_matches_scipy():
    rng = np.random.default_rng(1)
    fit = _feats(rng, 50, 6, 2, 3)
    acc = CovarianceAccumulator()
    acc.update(fit)
    mean, cov = acc.finalize()
    reg = regularize(cov, rel=0.01, eps=1e-6)
    W = whitening_matrices(reg)
    q = torch.from_numpy(rng.normal(size=(4, 6, 6)))  # (B, P, d) queries
    d_ours = mahalanobis(q, mean, W).numpy()
    for b in range(4):
        for p in range(6):
            vi = np.linalg.inv(reg[p].numpy())
            ref = scipy_mahalanobis(q[b, p].numpy(), mean[p].numpy(), vi)
            assert d_ours[b, p] == pytest.approx(ref, rel=1e-9)


def test_model_float32_scoring_close_to_float64_reference():
    rng = np.random.default_rng(2)
    fit = _feats(rng, 80, 8, 2, 2)
    acc = CovarianceAccumulator()
    acc.update(fit.float())
    gm = GaussianPatchModel.from_accumulator(acc, 0.01, 1e-6)
    q = _feats(rng, 3, 8, 2, 2).float()
    got = gm.patch_distances(q).numpy()  # (3, 2, 2)
    mean, cov = acc.finalize()
    reg = regularize(cov, 0.01, 1e-6).numpy()
    qp = _per_position(q.double())
    for p in range(4):
        vi = np.linalg.inv(reg[p])
        for b in range(3):
            ref = scipy_mahalanobis(qp[p, b], mean[p].numpy(), vi)
            assert got[b, p // 2, p % 2] == pytest.approx(ref, rel=1e-4)


def test_singular_covariance_is_regularised():
    # N < d and one constant channel: the sample covariance is singular.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 10, 1, 1))
    x[:, 3] = 5.0
    acc = CovarianceAccumulator()
    acc.update(torch.from_numpy(x))
    mean, cov = acc.finalize()
    assert np.linalg.matrix_rank(cov[0].numpy()) < 10
    W = whitening_matrices(regularize(cov, 0.01, 1e-6))
    assert torch.isfinite(W).all()
    d = mahalanobis(torch.from_numpy(rng.normal(size=(2, 1, 10))), mean, W)
    assert torch.isfinite(d).all() and (d > 0).all()


def test_all_zero_covariance_needs_eps():
    cov = torch.zeros(1, 3, 3, dtype=torch.float64)
    W = whitening_matrices(regularize(cov, rel=0.01, eps=1e-6))  # rel*0 + eps
    assert torch.isfinite(W).all()


def test_regularize_rejects_invalid():
    cov = torch.eye(2, dtype=torch.float64)[None]
    for rel, eps in [(0, 0), (-1, 0), (0.1, -1)]:
        with pytest.raises(ValueError):
            regularize(cov, rel, eps)


def test_non_positive_definite_raises():
    bad = -torch.eye(3, dtype=torch.float64)[None]
    with pytest.raises(torch.linalg.LinAlgError):
        whitening_matrices(bad)


def test_shape_validation():
    acc = CovarianceAccumulator()
    with pytest.raises(ValueError):
        acc.update(torch.zeros(2, 3, 4))  # not 4-D
    acc.update(torch.randn(3, 2, 2, 2))
    with pytest.raises(ValueError):
        acc.update(torch.randn(3, 2, 3, 2))  # grid changed
    with pytest.raises(ValueError):
        CovarianceAccumulator().finalize()  # no data
    with pytest.raises(ValueError):
        mahalanobis(torch.zeros(1, 2, 3), torch.zeros(2, 4), torch.zeros(2, 4, 4))
    gm = GaussianPatchModel(torch.zeros(4, 2), torch.eye(2).repeat(4, 1, 1), (2, 2), 1)
    with pytest.raises(ValueError):
        gm.patch_distances(torch.zeros(1, 2, 3, 3))


def test_identity_model_gives_euclidean_distance():
    gm = GaussianPatchModel(torch.zeros(1, 3), torch.eye(3)[None], (1, 1), 1)
    x = torch.tensor([3.0, 4.0, 0.0]).view(1, 3, 1, 1)
    assert gm.patch_distances(x).item() == pytest.approx(5.0)


def test_anomaly_map_and_image_score():
    d = torch.zeros(2, 4, 6)
    d[1, 2, 3] = 10.0
    m = anomaly_map(d, (16, 24), sigma=0)
    assert m.shape == (2, 16, 24)
    s = image_score(m)
    assert s[0] == 0 and s[1] > 0
    assert image_score(anomaly_map(d, (16, 24), sigma=2.0))[1] < s[1]  # smoothing lowers a spike


def test_state_dict_roundtrip():
    gm = GaussianPatchModel(torch.randn(4, 2), torch.randn(4, 2, 2), (2, 2), 9)
    gm2 = GaussianPatchModel.from_state_dict(gm.state_dict())
    assert torch.equal(gm.mean, gm2.mean) and gm2.grid == (2, 2) and gm2.n_fit == 9
