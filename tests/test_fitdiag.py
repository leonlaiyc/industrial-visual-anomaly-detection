"""Geometry helpers of the FIT-only diagnostics (Addendum B2)."""
import numpy as np
import pytest

from ivad.fitdiag import phase_shift, warp_discrepancy


def test_phase_shift_recovers_translation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 96))
    b = np.roll(np.roll(a, 5, axis=1), -3, axis=0)
    dx, dy = phase_shift(b, a)
    assert dx == pytest.approx(5, abs=0.2) and dy == pytest.approx(-3, abs=0.2)


def test_warp_discrepancy():
    p = {"cx": 100.0, "cy": 80.0, "angle": 0.0, "flipped": False}
    assert warp_discrepancy(p, p, (200, 160), (120, 80)) == 0
    q = {**p, "cx": 103.0, "cy": 84.0}  # pure translation: every pixel moves by 5 px
    assert warp_discrepancy(p, q, (200, 160), (120, 80)) == pytest.approx(5.0)
    r = {**p, "flipped": True}  # a 180° decision change moves pixels by twice their distance to the centre
    assert warp_discrepancy(p, r, (200, 160), (120, 80)) > 40
