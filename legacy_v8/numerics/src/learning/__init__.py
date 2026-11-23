"""Neural surrogate models for the CGMY parametric solution operator.

Stage 4 of the six-stage dependency chain: learn a fast approximation to
the map (CGMY parameters, contract data) -> B-spline coefficient vector,
trained on reference solves produced by ``cgmy.reference_solver`` and
stored via ``data.dataset``. Every architecture in this package must
ultimately produce (or be projected into, via ``projected_fno``) a
coefficient vector in the *same* discrete space the reference solver and
``certification`` package operate on.
"""
