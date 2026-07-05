"""
Unit tests for the proANUBIS geometry helpers.

These run without ROOT or any data file (pure NumPy), so CI can execute them on a
plain Python image in seconds -- no CERN software stack required.
"""
import numpy as np

from src.geometry import (
    delta_r_to_proanubis, wrap_dphi, PANUBIS_ETA, PANUBIS_PHI,
)


def test_delta_r_is_zero_at_detector_direction():
    assert delta_r_to_proanubis(PANUBIS_ETA, PANUBIS_PHI) == 0.0


def test_delta_r_matches_hand_calculation():
    # 0.1 away in eta, 0.2 away in phi -> sqrt(0.01 + 0.04)
    eta = PANUBIS_ETA + 0.1
    phi = PANUBIS_PHI + 0.2
    assert np.isclose(delta_r_to_proanubis(eta, phi), np.hypot(0.1, 0.2))


def test_phi_wraps_around_pi():
    # phi difference of ~2pi should wrap to ~0, not blow up
    assert np.isclose(wrap_dphi(2 * np.pi), 0.0, atol=1e-9)
    assert np.isclose(wrap_dphi(np.pi + 0.1), -(np.pi - 0.1))


def test_delta_r_vectorized():
    eta = np.array([PANUBIS_ETA, PANUBIS_ETA + 1.0])
    phi = np.array([PANUBIS_PHI, PANUBIS_PHI])
    out = delta_r_to_proanubis(eta, phi)
    assert out.shape == (2,)
    assert np.isclose(out[0], 0.0) and np.isclose(out[1], 1.0)
