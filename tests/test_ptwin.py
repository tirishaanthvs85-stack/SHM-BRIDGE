"""Unit tests for PTwin core and beam domain."""
import numpy as np
import pytest
from ptwin.core import SineRidgeDiff, auroc, dkw_threshold
import ptwin.beam as B


def test_beam_boundary_conditions():
    """Ensure beam deflection satisfies w(0) = w(L) = 0 exactly."""
    D = B.stiffness(sev=0.2, x0=0.5)
    w = B.deflection(D)
    assert np.isclose(w[0], 0.0, atol=1e-12)
    assert np.isclose(w[-1], 0.0, atol=1e-12)


def test_beam_campaign_shapes():
    """Verify campaign dictionary dimensions."""
    rng = np.random.default_rng(42)
    c = B.campaign(rng, sev=0.3, x0=0.4)
    assert c["y"].shape == (24,)
    assert c["w"].shape == (801,)
    assert c["D"].shape == (801,)
    assert c["sigma"] > 0


def test_sine_ridge_diff_exact():
    """Test SineRidgeDiff on known sine function."""
    z = np.linspace(0, 1, 26)[1:-1]
    y = np.sin(np.pi * z)
    est = SineRidgeDiff(L=1.0, K=20, lam=1e-10).fit(z, y)
    
    # Second derivative of sin(pi*z) is -pi^2 * sin(pi*z)
    z_eval = np.linspace(0.1, 0.9, 50)
    d2_pred = est.d2(z_eval)
    d2_true = -(np.pi ** 2) * np.sin(np.pi * z_eval)
    np.testing.assert_allclose(d2_pred, d2_true, rtol=1e-2, atol=1e-2)


def test_auroc_metric():
    """Verify AUROC calculation logic."""
    neg = [0.1, 0.2, 0.3]
    pos = [0.7, 0.8, 0.9]
    assert np.isclose(auroc(neg, pos), 1.0)
    
    # Inverted
    assert np.isclose(auroc(pos, neg), 0.0)


def test_dkw_threshold_validity():
    """Verify DKW threshold is valid and higher than median clean score."""
    clean_scores = np.random.exponential(scale=0.01, size=100)
    tau = dkw_threshold(clean_scores, alpha=0.05, delta=0.05)
    assert tau > np.median(clean_scores)
