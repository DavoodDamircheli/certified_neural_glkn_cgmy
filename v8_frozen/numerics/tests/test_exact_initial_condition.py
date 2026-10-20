"""c_theta(0) = c_h^0 must hold at machine precision, for ANY weights (random,
untrained) and ANY inputs -- it is a structural property of the ansatz
c_theta(tau) = c_h^0 + tau * N_theta(...), not something training achieves.

Task 7 (working prompt), "Initial condition" assumption test: ||u_theta(0)
- u_0|| <= 1e-12. The MLP test below predates Task 7; the DeepONet/POD/
ProjectedFNO tests were added for it, extending the SAME adversarial
(deliberately large weights/basis values, not just typical random init)
stress test to all four CoefficientModel implementations -- CLAUDE.md Hard
Invariant #4 claims this holds structurally for ANY weights, a claim only
an adversarial trial actually exercises (tests/test_model_interface.py
already checks exact-IC with default-init weights on all four; that is a
weaker claim than "for ANY weights").
"""

from __future__ import annotations

import numpy as np
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.fno import FNO1d
from learning.mlp_coeff import MLPCoeffNet
from learning.pod import PODBasis, PODGalerkinROM
from learning.projected_fno import ProjectedFNO


def test_c_theta_at_tau_zero_equals_c0_random_weights_random_inputs():
    torch.manual_seed(0)
    mu_dim, psi_dim, batch = 6, 66, 8

    for trial in range(5):
        net = MLPCoeffNet(mu_dim=mu_dim, psi_dim=psi_dim, hidden_dims=(32, 32))
        # random (untrained) weights are the default init; re-randomize per trial
        for p in net.parameters():
            p.data = torch.randn_like(p) * 3.0  # deliberately large/unusual weights

        tau = torch.zeros(batch, 1, dtype=torch.float64)
        tau_norm = torch.rand(batch, 1, dtype=torch.float64) * 2 - 1  # arbitrary, tau=0 doesn't imply tau_norm=0
        mu_tilde = torch.rand(batch, mu_dim, dtype=torch.float64) * 2 - 1
        psi = torch.randn(batch, psi_dim, dtype=torch.float64)
        c0 = torch.randn(batch, psi_dim, dtype=torch.float64)

        net = net.double()
        c_theta_0 = net(tau, tau_norm, mu_tilde, psi, c0)
        max_err = (c_theta_0 - c0).abs().max().item()
        print(f"\ntrial {trial}: max |c_theta(0) - c0| = {max_err:.3e}")
        assert max_err == 0.0, (
            f"trial {trial}: c_theta(0) != c0 exactly (max err {max_err:.3e}); "
            f"the tau prefactor must be exactly zero at tau=0 regardless of N_theta's output"
        )


def test_c_theta_at_small_tau_is_close_to_c0():
    """Sanity check the OTHER direction: at small but nonzero tau, c_theta
    should be close to c0 (continuity), scaling with tau and ||N_theta||."""
    torch.manual_seed(1)
    net = MLPCoeffNet(mu_dim=6, psi_dim=10, hidden_dims=(16,)).double()
    batch = 4
    tau_norm = torch.zeros(batch, 1, dtype=torch.float64)
    mu_tilde = torch.zeros(batch, 6, dtype=torch.float64)
    psi = torch.zeros(batch, 10, dtype=torch.float64)
    c0 = torch.randn(batch, 10, dtype=torch.float64)

    n_theta = net.forward_n_theta(tau_norm, mu_tilde, psi)
    for tau_val in (1e-8, 1e-4, 1e-2):
        tau = torch.full((batch, 1), tau_val, dtype=torch.float64)
        c_theta = net(tau, tau_norm, mu_tilde, psi, c0)
        expected = c0 + tau_val * n_theta
        assert torch.allclose(c_theta, expected, atol=1e-12)


def test_deeponet_c_theta_at_tau_zero_equals_c0_adversarial_weights():
    """Same structural claim, same adversarial-weight stress, for
    DeepONetCoeffNet (c_theta(tau) = c0 + tau * DeepONet_theta(...) -- the
    SAME tau=0 multiplicative trick as MLPCoeffNet, see
    learning.deeponet_coeff's module docstring)."""
    torch.manual_seed(0)
    mu_dim, psi_dim, batch = 6, 66, 8

    for trial in range(5):
        net = DeepONetCoeffNet(
            mu_dim=mu_dim, psi_dim=psi_dim, latent_dim=16,
            branch_hidden=(16,), trunk_hidden=(16,),
        ).double()
        for p in net.parameters():
            p.data = torch.randn_like(p) * 3.0  # deliberately large/unusual weights

        tau = torch.zeros(batch, 1, dtype=torch.float64)
        tau_norm = torch.rand(batch, 1, dtype=torch.float64) * 2 - 1
        mu_tilde = torch.rand(batch, mu_dim, dtype=torch.float64) * 2 - 1
        psi = torch.randn(batch, psi_dim, dtype=torch.float64)
        c0 = torch.randn(batch, psi_dim, dtype=torch.float64)

        c_theta_0 = net(tau, tau_norm, mu_tilde, psi, c0)
        max_err = (c_theta_0 - c0).abs().max().item()
        assert max_err == 0.0, (
            f"trial {trial}: DeepONet c_theta(0) != c0 exactly (max err {max_err:.3e})"
        )


def test_projected_fno_c_theta_at_tau_zero_equals_c0_adversarial_weights():
    """Same structural claim for ProjectedFNO -- a DIFFERENT mechanism
    (c_theta(tau) = c0 + [Pi_h(u_FNO(tau)) - Pi_h(u_FNO(0))], a subtraction
    identity rather than a tau=0 multiplication -- see
    learning.projected_fno's module docstring), so worth testing
    separately even though both give exact cancellation in IEEE 754."""
    N, p = 16, 3
    x_min, x_max = float(np.log(100) - 4), float(np.log(100) + 4)
    ops = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    n_dof = ops.n_dof

    torch.manual_seed(2)
    for trial in range(3):
        fno = FNO1d(in_channels=1 + 1 + 6, out_channels=1, width=8, n_modes=4, n_layers=2).double()
        for p_ in fno.parameters():
            p_.data = torch.randn_like(p_) * 3.0
        net = ProjectedFNO(fno, Mesh(N=N, p=p, x_min=x_min, x_max=x_max), ops, mu_dim=6)

        rng = np.random.default_rng(trial)
        mu_tilde = rng.uniform(-1, 1, size=6)
        psi = rng.standard_normal(n_dof)
        c0 = rng.standard_normal(n_dof)
        time_grid = np.linspace(0.0, 0.5, 5)

        pred = net.predict_trajectory(mu_tilde, psi, c0, time_grid, T=0.5)
        max_err = np.max(np.abs(pred[0] - c0))
        assert max_err == 0.0, (
            f"trial {trial}: ProjectedFNO c_theta(0) != c0 exactly (max err {max_err:.3e})"
        )


def test_pod_galerkin_c_theta_at_tau_zero_equals_c0_adversarial_basis():
    """Same structural claim for PODGalerkinROM -- POD has no gradient-
    trained weights, so the adversarial stress here is a deliberately
    large/unusual POD basis (modes, mean) rather than large weights; the
    subtraction-identity mechanism (c0 + [g(tau) - g(0)]) doesn't require
    the modes to be a genuine orthonormal SVD basis for c_theta(0) = c0 to
    hold exactly, so an arbitrary large-magnitude 'basis' is a valid,
    stronger stress test."""
    N, p = 16, 3
    x_min, x_max = float(np.log(100) - 4), float(np.log(100) + 4)
    ops = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    n_dof = ops.n_dof

    from data.sampling import load_parameter_box
    from learning.normalization import stats_from_box

    stats = stats_from_box(load_parameter_box("configs/base.yaml"))

    rng = np.random.default_rng(4)
    n_modes = 5
    for trial in range(3):
        modes = rng.standard_normal((n_dof, n_modes)) * 50.0  # deliberately large, non-orthonormal
        mean = rng.standard_normal(n_dof) * 50.0
        basis = PODBasis(modes=modes, singular_values=np.ones(n_modes), mean=mean)
        rom = PODGalerkinROM(basis, ops, stats=stats)

        mu_tilde = rng.uniform(-1, 1, size=6)
        c0 = rng.standard_normal(n_dof)
        time_grid = np.linspace(0.0, 0.5, 5)

        pred = rom.predict_trajectory(mu_tilde, None, c0, time_grid, T=0.5)
        max_err = np.max(np.abs(pred[0] - c0))
        assert max_err == 0.0, (
            f"trial {trial}: PODGalerkinROM c_theta(0) != c0 exactly (max err {max_err:.3e})"
        )
