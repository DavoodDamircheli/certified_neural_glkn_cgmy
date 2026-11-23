"""Differentiable (autograd-compatible) wrappers around
``cgmy.operators.OperatorCache``, for residual-based training losses
(Stage G). These WRAP the existing NumPy/SciPy apply_A/apply_AT/apply_M/
apply_M_inv (the single operator implementation, CLAUDE.md Hard Invariant
#1) via ``torch.autograd.Function`` -- no CGMY/Toeplitz numerics are
reimplemented in torch. mu is treated as a fixed, non-differentiable input
(training is w.r.t. network weights, not mu), so each Function only needs
a vector-Jacobian product with respect to its tensor input ``v``.

apply_A(mu, .) is linear in v, so its Jacobian is exactly A_h(mu) itself:
the VJP of a downstream gradient g is A_h(mu)^T @ g = apply_AT(mu, g) --
which is exactly why Stage A built apply_AT in the first place. M_h and
M_h^{-1} are symmetric, so they are self-adjoint (their own VJP).
"""

from __future__ import annotations

import numpy as np
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters


class _ApplyABatch(torch.autograd.Function):
    """Differentiable batched apply_A: row i of v uses mus[i]; all rows share ops
    (built from one common mesh -- see module docstring on why a fixed mesh
    is used for Stage G's training set, same reasoning as Stage F's POD)."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, mus: tuple[CGMYParameters, ...], ops: OperatorCache) -> torch.Tensor:
        ctx.mus = mus
        ctx.ops = ops
        v_np = v.detach().cpu().numpy()
        out_np = np.stack([ops.apply_A(mu, v_np[i]) for i, mu in enumerate(mus)])
        return torch.as_tensor(out_np, dtype=v.dtype, device=v.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        g_np = grad_output.detach().cpu().numpy()
        vjp_np = np.stack([ctx.ops.apply_AT(mu, g_np[i]) for i, mu in enumerate(ctx.mus)])
        grad_v = torch.as_tensor(vjp_np, dtype=grad_output.dtype, device=grad_output.device)
        return grad_v, None, None


def apply_A_batch(v: torch.Tensor, mus: list[CGMYParameters], ops: OperatorCache) -> torch.Tensor:
    """v: (batch, n_dof). Returns (batch, n_dof) with row i = A_h(mus[i]) @ v[i],
    differentiable w.r.t. v."""
    return _ApplyABatch.apply(v, tuple(mus), ops)


class _ApplyMBatch(torch.autograd.Function):
    """Differentiable batched apply_M (mu-independent; self-adjoint since M_h is symmetric)."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, ops: OperatorCache) -> torch.Tensor:
        ctx.ops = ops
        v_np = v.detach().cpu().numpy()
        out_np = np.stack([ops.apply_M(row) for row in v_np])
        return torch.as_tensor(out_np, dtype=v.dtype, device=v.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        g_np = grad_output.detach().cpu().numpy()
        vjp_np = np.stack([ctx.ops.apply_M(row) for row in g_np])
        return torch.as_tensor(vjp_np, dtype=grad_output.dtype, device=grad_output.device), None


def apply_M_batch(v: torch.Tensor, ops: OperatorCache) -> torch.Tensor:
    """v: (batch, n_dof). Returns (batch, n_dof) = M_h @ v[i], differentiable w.r.t. v."""
    return _ApplyMBatch.apply(v, ops)


class _ApplyMInvBatch(torch.autograd.Function):
    """Differentiable batched apply_M_inv (self-adjoint: M_h^{-1} is symmetric too).

    Used for the M_h^{-1}-weighted norm in L_res, not for the residual
    formula itself (which uses apply_M, not apply_M_inv).
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, ops: OperatorCache) -> torch.Tensor:
        ctx.ops = ops
        v_np = v.detach().cpu().numpy()
        out_np = np.stack([ops.apply_M_inv(row) for row in v_np])
        return torch.as_tensor(out_np, dtype=v.dtype, device=v.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        g_np = grad_output.detach().cpu().numpy()
        vjp_np = np.stack([ctx.ops.apply_M_inv(row) for row in g_np])
        return torch.as_tensor(vjp_np, dtype=grad_output.dtype, device=grad_output.device), None


def apply_M_inv_batch(v: torch.Tensor, ops: OperatorCache) -> torch.Tensor:
    """v: (batch, n_dof). Returns (batch, n_dof) = M_h^{-1} @ v[i], differentiable w.r.t. v."""
    return _ApplyMInvBatch.apply(v, ops)
