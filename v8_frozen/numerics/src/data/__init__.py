"""Parameter sampling, payoff families, and dataset construction.

Feeds stage 1 (reference solves) into stage 4 (neural model training): this
package samples the frozen CGMY/contract parameter box (``configs/base.yaml``),
resolves payoff definitions, and stores reference-solver outputs to HDF5 via
``common.io`` for consumption as training data.
"""
