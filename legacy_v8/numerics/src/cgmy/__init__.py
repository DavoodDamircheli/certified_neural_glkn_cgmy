"""CGMY pricing model: the trusted Galerkin reference solver and its supporting operators.

This package wraps an existing, externally validated B-spline Galerkin solver
for the tempered-fractional CGMY pricing PIDE. Nothing in this package
reimplements the solver's numerics; it adapts the solver's inputs/outputs to
the interfaces used by ``certification``, ``learning``, and ``data``.
"""
