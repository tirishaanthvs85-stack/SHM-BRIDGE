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


# ---- Phase 1: domains -------------------------------------------------------------------------
from ptwin.domains import DOMAINS
from ptwin.core import TikhonovDiff, eps_bound, hsu_quad_bound


@pytest.mark.parametrize("name", list(DOMAINS))
def test_domain_exact_solution_has_zero_ideal_residual(name):
    D = DOMAINS[name]()
    c = D.sample(np.random.default_rng(0), faulty=False)
    assert np.abs(D.ideal_resid(c)).max() <= 1e-6


@pytest.mark.parametrize("name", list(DOMAINS))
def test_domain_noiseless_estimator_residual_small(name):
    D = DOMAINS[name]()
    c = D.sample(np.random.default_rng(0), faulty=False)
    c["y"] = c["state"]
    s, _ = D.stat(c, D.lam_default)
    assert s < 0.02


@pytest.mark.parametrize("name", list(DOMAINS))
def test_domain_shapes(name):
    D = DOMAINS[name]()
    c = D.sample(np.random.default_rng(1), faulty=True)
    assert c["y"].shape == (D.m,) and c["state"].shape == (D.m,) and c["state_h"].shape == (D.m,)
    assert D.ideal_resid(c).shape == D.zf.shape


def test_tikhonov_derivative_of_quadratic():
    z = np.linspace(0, 1, 26)[1:-1]
    est = TikhonovDiff(lam=1e-12, order=2, anchors=[(0.0, 0.0, 1e6)]).fit(z, z ** 2)
    assert abs(est.d1(np.array([0.5]))[0] - 1.0) < 1e-2


def test_dkw_fpr_beam_fresh_healthy():
    D = DOMAINS["beam"]()
    rng = np.random.default_rng(5)
    sc = lambda n: np.array([D.stat(D.sample(rng, False), D.lam_default)[0] for _ in range(n)])
    tau = dkw_threshold(sc(100), alpha=0.05, delta=0.05)
    assert (sc(2000) > tau).mean() <= 0.05


def test_hsu_bound_covers_noise():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(50, 24))
    b = hsu_quad_bound(J, 0.1, delta=0.05)
    norms = np.array([np.linalg.norm(J @ rng.normal(0, 0.1, 24)) for _ in range(2000)])
    assert (norms <= b).mean() >= 0.95


# ---- Phases 2-5: twins -------------------------------------------------------------------------
import torch
from ptwin.models import Twin, gen_batch, to_t, d2
from ptwin import twin_gen


def test_gen_batch_bc_and_shapes():
    b = gen_batch(np.random.default_rng(0), 8)
    assert b["y"].shape == (8, 24) and b["D"].shape == (8, 801)
    assert np.allclose(b["state_h"][~b["faulty"]], b["state"][~b["faulty"]], atol=1e-9)


def test_beam_twin_hard_bcs_and_residual_path():
    net = Twin(skeleton=True)
    b = gen_batch(np.random.default_rng(1), 4)
    y, e = to_t(b)
    z = torch.tensor([[[0.0], [0.5], [1.0]]]).expand(4, -1, -1).clone().requires_grad_(True)
    u, dev = net(z, y, e)
    assert abs(float(u[:, 0].abs().max())) < 1e-6 and abs(float(u[:, 2].abs().max())) < 1e-6
    assert d2(u, z).shape == (4, 3, 1) and (dev >= 0).all() and (dev <= 0.7).all()


@pytest.mark.parametrize("domain", ["pipeline", "battery"])
def test_grid_twin_forward_and_loss(domain):
    gen = twin_gen.gen_pipe if domain == "pipeline" else twin_gen.gen_batt
    net = twin_gen.GridTwin(domain)
    b = gen(np.random.default_rng(0), 6)
    loss, _ = twin_gen.compute_loss(net, b, twin_gen.DEFAULT, "clean")
    assert torch.isfinite(loss)
    loss, _ = twin_gen.compute_loss(net, gen(np.random.default_rng(1), 6, faulty=True, env=np.full(6, 0.3 if domain == "pipeline" else 40.0)),
                                    {**twin_gen.DEFAULT, "lam_evade": 1.0}, "adaptive")
    assert torch.isfinite(loss)
