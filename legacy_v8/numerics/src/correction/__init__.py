"""Predict-certify-correct loop: fixing up neural predictions that fail certification.

Stage 5 of the six-stage dependency chain. When
``certification.guaranteed_certificate.is_accepted`` rejects a neural
prediction, this package runs a cheap iterative correction seeded by that
prediction, rather than falling back to a full reference solve.
"""
