"""The torch (differentiable) CN residual must agree with the NumPy residual
(certification.residuals.compute_cn_residual) to tight tolerance on the
same inputs -- confirming learning.losses.torch_cn_residual really does
wrap the same operator code, not a reimplementation. Also checks gradients
flow correctly (finite, and match a numerical Jacobian-vector check via
torch.autograd.gradcheck on a small case).
"""

from __future__ import annotations

import numpy as np
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.residuals import compute_cn_residual, residual_norm
from learning.differentiable_operators import apply_A_batch, apply_M_batch, apply_M_inv_batch
from learning.losses import residual_loss, torch_cn_residual

torch.set_default_dtype(torch.float64)

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
PAYOFF = EuropeanPutPayoff(strike=100.0)


def _make_trajectory():
    mesh = Mesh(N=32, p=3)
    tg = TimeGrid(n_steps=20)
    traj = solve_cgmy(MU, PAYOFF, mesh, tg, mode="toeplitz")
    ops = OperatorCache(Mesh(N=traj.mesh_information.N, p=traj.mesh_information.p),
                         traj.mesh_information.x_min, traj.mesh_information.x_max)
    return traj, ops


def test_torch_residual_matches_numpy_residual_on_exact_trajectory():
    traj, ops = _make_trajectory()
    n = traj.solver_metadata.n_steps // 2

    r_numpy = compute_cn_residual(traj, MU, ops, n)

    c_n = torch.tensor(traj.coefficients[n]).unsqueeze(0)
    c_np1 = torch.tensor(traj.coefficients[n + 1]).unsqueeze(0)
    fbar = torch.tensor(traj.forcing_vectors[n]).unsqueeze(0)
    r_torch = torch_cn_residual(
        c_n, c_np1, fbar, traj.solver_metadata.dt, [MU], ops, traj.solver_metadata.bc_dofs
    )[0].numpy()

    max_diff = np.max(np.abs(r_torch - r_numpy))
    print(f"\nmax |r_torch - r_numpy| on exact trajectory (step {n}) = {max_diff:.3e}")
    assert max_diff < 1e-12


def test_torch_residual_matches_numpy_residual_on_perturbed_trajectory():
    """Same check but on a perturbed (nonzero-residual) state, to make sure
    agreement isn't a coincidence of the residual being ~0."""
    traj, ops = _make_trajectory()
    n = traj.solver_metadata.n_steps // 2
    rng = np.random.default_rng(0)

    import dataclasses
    coeffs = traj.coefficients.copy()
    coeffs[n + 1] = coeffs[n + 1] + 0.05 * np.linalg.norm(coeffs[n + 1]) * rng.standard_normal(ops.n_dof)
    traj_pert = dataclasses.replace(traj, coefficients=coeffs)

    r_numpy = compute_cn_residual(traj_pert, MU, ops, n)

    c_n = torch.tensor(traj_pert.coefficients[n]).unsqueeze(0)
    c_np1 = torch.tensor(traj_pert.coefficients[n + 1]).unsqueeze(0)
    fbar = torch.tensor(traj_pert.forcing_vectors[n]).unsqueeze(0)
    r_torch = torch_cn_residual(
        c_n, c_np1, fbar, traj_pert.solver_metadata.dt, [MU], ops, traj_pert.solver_metadata.bc_dofs
    )[0].numpy()

    baseline_norm = residual_norm(r_numpy, ops, traj.solver_metadata.bc_dofs)
    max_diff = np.max(np.abs(r_torch - r_numpy))
    print(f"\nperturbed residual norm = {baseline_norm:.3e}; max |r_torch - r_numpy| = {max_diff:.3e}")
    assert baseline_norm > 1e-3, "sanity: this residual should be clearly nonzero"
    assert max_diff < 1e-12


def test_residual_loss_gradient_matches_finite_difference():
    """torch_cn_residual/residual_loss must produce correct gradients w.r.t.
    c_theta -- checked against a central finite-difference estimate on a
    small mesh (not gradcheck's double-backward machinery, since apply_A's
    custom Function only implements a first-order VJP)."""
    N, p, x_min, x_max = 16, 3, -3.0, 3.0
    ops = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    n_dof = ops.n_dof
    dt = 0.01
    bc_dofs = ([], [])  # no BC masking complication for this isolated gradient check

    rng = np.random.default_rng(1)
    c_n = torch.tensor(rng.standard_normal(n_dof)).unsqueeze(0)
    c_np1 = torch.tensor(rng.standard_normal((1, n_dof)), requires_grad=True)
    fbar = torch.zeros(1, n_dof)

    def loss_fn(c_np1_):
        r = torch_cn_residual(c_n, c_np1_, fbar, dt, [MU], ops, bc_dofs)
        return residual_loss(r, ops)

    loss = loss_fn(c_np1)
    loss.backward()
    grad_analytic = c_np1.grad[0].detach().numpy().copy()

    eps = 1e-6
    grad_fd = np.zeros(n_dof)
    base = c_np1.detach().numpy()[0].copy()
    for i in range(n_dof):
        cp = base.copy(); cp[i] += eps
        cm = base.copy(); cm[i] -= eps
        lp = loss_fn(torch.tensor(cp).unsqueeze(0)).item()
        lm = loss_fn(torch.tensor(cm).unsqueeze(0)).item()
        grad_fd[i] = (lp - lm) / (2 * eps)

    max_diff = np.max(np.abs(grad_analytic - grad_fd))
    rel_diff = max_diff / (np.max(np.abs(grad_fd)) + 1e-12)
    print(f"\nmax |grad_analytic - grad_fd| = {max_diff:.3e}  (relative: {rel_diff:.3e})")
    assert rel_diff < 1e-4


def test_apply_operators_are_finite_and_differentiable():
    """Smoke test on the raw wrapped operators (apply_A_batch/apply_M_batch/
    apply_M_inv_batch): gradients must be finite and backward() must not raise."""
    N, p, x_min, x_max = 16, 3, -3.0, 3.0
    ops = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    n_dof = ops.n_dof
    v = torch.randn(3, n_dof, requires_grad=True)

    out_A = apply_A_batch(v, [MU, MU, MU], ops)
    out_M = apply_M_batch(v, ops)
    out_Minv = apply_M_inv_batch(v, ops)
    loss = out_A.sum() + out_M.sum() + out_Minv.sum()
    loss.backward()

    assert torch.isfinite(v.grad).all()
