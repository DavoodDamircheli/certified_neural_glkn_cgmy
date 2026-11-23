import numpy as np
import scipy.sparse.linalg as spla
from scipy.integrate import fixed_quad

from cgmy_bspline.bspline_basis import phi, active_indices
from cgmy_bspline.boundary import put_payoff


def assemble_payoff_rhs(payoff_fn, grid_info, p, kink_x=None):
    """
    Assemble b^0_k = integral_{Omega} payoff_fn(x) * phi_{h,k}(x) dx for each active k.

    Integration is performed element-by-element within each basis function's support.
    When ``kink_x`` falls strictly inside an element, the integral is split there to
    avoid loss of accuracy from a kink in ``payoff_fn`` (e.g. the max(...,0) kink of
    a vanilla payoff at the strike). Pass ``kink_x=None`` for a smooth payoff.

    This is the payoff-generic core that ``assemble_put_rhs`` specializes; the
    quadrature scheme (element-by-element, kink-aware split, fixed_quad n=10) is
    unchanged from the original put-only implementation.
    """
    N = grid_info['N']
    h = grid_info['h']
    x_min = grid_info['x_min']

    ks = active_indices(N, p)
    b0 = np.zeros(N + p - 1)

    for i, k in enumerate(ks):
        j_lo = max(0, k)
        j_hi = min(N - 1, k + p - 1)

        val = 0.0
        for j in range(j_lo, j_hi + 1):
            a_elem = x_min + j * h
            b_elem = x_min + (j + 1) * h
            integrand = lambda x, _k=k: payoff_fn(x) * phi(x, _k, p, h, x_min)
            if kink_x is not None and a_elem < kink_x < b_elem:
                v1, _ = fixed_quad(integrand, a_elem, kink_x, n=10)
                v2, _ = fixed_quad(integrand, kink_x, b_elem, n=10)
                val += v1 + v2
            else:
                v, _ = fixed_quad(integrand, a_elem, b_elem, n=10)
                val += v

        b0[i] = val

    return b0


def assemble_put_rhs(params, grid_info, p):
    """
    European put specialization of ``assemble_payoff_rhs``: payoff_fn = put_payoff,
    kink at x = log(K). Behavior is unchanged from the original put-only
    implementation.
    """
    K = params.K
    return assemble_payoff_rhs(
        lambda x: put_payoff(x, K), grid_info, p, kink_x=np.log(K)
    )


def l2_project_generic(payoff_fn, grid_info, p, matrices_dict, kink_x=None):
    """
    Compute c^0 by solving M_h * c^0 = b^0 for an arbitrary terminal payoff.

    Parameters
    ----------
    payoff_fn     : callable, payoff_fn(x) -> value at log-price x
    grid_info     : dict from grid.make_grid
    p             : B-spline order
    matrices_dict : dict containing key 'M_h' (from matrices.operator_matrix)
    kink_x        : log-price location of a kink in payoff_fn, or None if smooth

    Returns
    -------
    c0 : numpy array of length N+p-1, the initial coefficient vector

    Note: this only projects the *terminal condition*. The Dirichlet boundary
    values applied during time-stepping in ``solver_cn.solve_cn`` encode the
    put's own analytic far-field asymptotics (V -> K*exp(-r*tau) deep ITM,
    V -> 0 deep OTM) and are NOT payoff-generic. Using this function with a
    non-put payoff_fn together with solve_cn's existing boundary handling
    will silently apply the wrong boundary condition for that payoff.
    """
    b0 = assemble_payoff_rhs(payoff_fn, grid_info, p, kink_x=kink_x)
    M_h = matrices_dict['M_h']
    return spla.spsolve(M_h, b0)


def l2_project_payoff(params, grid_info, p, matrices_dict):
    """
    Compute c^0 by solving M_h * c^0 = b^0.

    Parameters
    ----------
    params        : CGMYParams
    grid_info     : dict from grid.make_grid
    p             : B-spline order
    matrices_dict : dict containing key 'M_h' (from matrices.operator_matrix)

    Returns
    -------
    c0 : numpy array of length N+p-1, the initial coefficient vector
    """
    b0 = assemble_put_rhs(params, grid_info, p)
    M_h = matrices_dict['M_h']
    return spla.spsolve(M_h, b0)


def eval_solution(c, grid_info, p, x_eval):
    """
    Evaluate V_h(x) = sum_k c_k * phi_{h,k}(x) at each x in x_eval.

    Parameters
    ----------
    c         : coefficient vector, length N+p-1
    grid_info : dict from grid.make_grid
    p         : B-spline order
    x_eval    : numpy array of evaluation points

    Returns
    -------
    V : numpy array of same length as x_eval
    """
    h = grid_info['h']
    x_min = grid_info['x_min']
    N = grid_info['N']
    ks = active_indices(N, p)

    x_eval = np.asarray(x_eval, dtype=float)
    V = np.zeros_like(x_eval)
    for i, k in enumerate(ks):
        V += c[i] * phi(x_eval, k, p, h, x_min)
    return V
