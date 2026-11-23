"""A posteriori certification of neural-operator predictions against the CGMY PIDE.

Implements stages 2-3 of the six-stage dependency chain:

    Galerkin solver -> discrete residual -> certificate -> neural model
    -> predict-certify-correct -> experiments

Given any candidate coefficient vector (from the reference solver, a neural
model, or a corrected prediction), this package computes a discrete
residual and a guaranteed, computable upper bound on the true error, using
only quantities available at prediction time (no reference solution
required).
"""
