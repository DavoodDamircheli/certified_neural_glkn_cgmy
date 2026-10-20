"""cgmy_cert: symbol-based exact assembly and residual certification for the
tempered-fractional CGMY spline Galerkin discretisation (v9 package).

Modules
-------
params      parameter box, query sets, scalar CGMY quantities, mesh conventions
symbols     periodised symbols, truncation tail bound, Toeplitz sections, free matrices
spectral    rigorous spectral lower bound, banded comparison norm, Theta
riesz       circulant-preconditioned PCG with the two-sided dual-norm bracket
direct      forcing/lifting, initial projection, exact-assembly CN march
certificate the certificate driver (banded / exact / two-stage modes)

Nothing in this package imports the frozen v8 workspace; the regression tests do.
"""

from .params import Query, Mesh, BOX, P_ORDER, X_WIDTH, X_MIN, mesh, box_vertices, sobol_queries
from .symbols import (bspline_ft_sq, alias_sum, tail_bound, symbols, toeplitz_section,
                      free_matrices, FreeSystem, ToeplitzOp, banded_from_toeplitz)

__version__ = "0.1.0"
