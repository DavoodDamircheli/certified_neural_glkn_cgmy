"""Plan 2.2 done-when criteria (affine POD-Galerkin) and reference-file integrity."""
import json

import numpy as np
import pytest
from scipy.linalg import solve_triangular, norm

from conftest import ROOT, FULL
from cgmy_cert.params import Query
from cgmy_cert.eim import load_eim, sha256
from cgmy_cert.pod import AffineROM, pod_galerkin, mass_cholesky, boundary_couplings, allowance_vectors
from cgmy_cert.direct import reference_trajectory, mass_norm, free_forcing, initial_free_vector
from cgmy_cert.spectral import spectral_data
from cgmy_cert.certificate import certify

QS = [Query(0.4, 1.7, 6.3, 1.45, 0.03, T=0.7), Query(0.95, 1.1, 8.5, 1.78, 0.09, T=1.0),
      Query(0.06, 4.8, 2.2, 1.12, 0.011, T=0.15)]


@pytest.fixture(scope="module")
def eim64():
    return load_eim(64, nproc=8)


def _full_basis(fs):
    return solve_triangular(mass_cholesky(fs).T, np.eye(fs.n), lower=False)      # r = n_free = 62


def _rel(fs, Z, Zref):
    return (max(mass_norm(fs, Z[:, k] - Zref[:, k]) for k in range(Z.shape[1]))
            / max(mass_norm(fs, Zref[:, k]) for k in range(Z.shape[1])))


@pytest.mark.parametrize("q", QS)
def test_full_rank_reduced_equals_direct(q, eim64):
    """r = 62 (= n_free at N = 64): with A_I the reduced solution is the direct one to 1e-10; with A_Q to
    eps_Q level (eps_Q / ||A_I|| <= 1e-8, plan 2.1)."""
    fs, forc, tr = reference_trajectory(q, 64, 60, solver="lu")
    V = _full_basis(fs)
    assert _rel(fs, pod_galerkin(fs, forc, V, tr.Z[:, 0]), tr.Z) < 1e-10
    rom = AffineROM(V, eim64)
    sol = rom.solve(q, rom.project(tr.Z[:, 0]), 60)
    assert _rel(fs, V @ sol.alpha, tr.Z) < 1e-8


@pytest.mark.parametrize("q", QS)
def test_affine_forcing(q, eim64):
    """Affine reduced forcing equals V^T F (built with the full operator) to eps_Q level, and the boundary
    couplings built from Toeplitz columns reproduce direct.free_forcing exactly for A_I."""
    fs = reference_trajectory(q, 64, 60, solver="lu")[0]
    forc = free_forcing(fs, q.T, 60)
    uA = boundary_couplings(fs.A.t.real, fs.n, fs.mesh.p).sum(axis=0)
    assert np.max(np.abs(uA - forc.uA)) <= 1e-14 * np.max(np.abs(forc.uA))
    V = _full_basis(fs)
    rom = AffineROM(V, eim64)
    info = eim64.eps_Q(q)
    Fr, _, bL = rom.forcing(q, q.T, 60, info["ell"])
    sb = 0.5 * (bL[1:] + bL[:-1])
    bound = info["eps_Q"] * np.sqrt(fs.mesh.p - 1) * norm(V, 2) * sb.max()
    assert np.max(np.abs(Fr - V.T @ forc.F)) <= bound


def test_certificate_with_allowance_is_valid(eim64):
    """POD-Galerkin on A_Q (r = 24 from 20 trajectories), certified on (A_Q, f_Q) with the raw-vector allowance:
    the bound dominates the true error against the exact (A_I) reference."""
    train = [Query(0.2 + 0.04 * i, 1.0 + 0.2 * i, 2.5 + 0.3 * i, 1.15 + 0.03 * i, 0.02 + 0.004 * i, T=0.5)
             for i in range(20)]
    snaps = np.hstack([reference_trajectory(t, 64, 60, solver="lu")[2].Z for t in train])
    from cgmy_cert.pod import pod_basis
    for q in QS:
        fs, forc, tr = reference_trajectory(q, 64, 60, solver="lu")
        V, _ = pod_basis(snaps, fs, 24)
        rom = AffineROM(V, eim64)
        z0 = initial_free_vector(q, fs.mesh)
        sol = rom.solve(q, rom.project(z0), 60)
        Z = V @ sol.alpha
        fs_Q, info = eim64.affine_free_system(q)
        spec = spectral_data(q, 64)
        aS, aX, raw, free = allowance_vectors(Z, sol.bL, 3, info["eps_Q"], spec.alpha_lb, spec.x_min)
        assert np.all(raw >= free)
        forc_Q = free_forcing(fs_Q, q.T, 60)
        for mode in ("banded", "exact"):
            c = certify(fs_Q, forc_Q, Z, z0, 1e-3, mode=mode, spec=spec, allowance=aS, allowance_X=aX)
            err = max(mass_norm(fs, Z[:, k] - tr.Z[:, k]) for k in range(61))
            assert err <= c.bound, (mode, err, c.bound)


# ---------------------------------------------------------------- stored references (plan 2.2)

DATA = ROOT / "data"
MAN = ROOT / "results" / "MANIFEST.json"


@pytest.mark.skipif(not MAN.exists(), reason="references not generated (scripts/gen_references.py)")
def test_reference_files():
    man = json.loads(MAN.read_text())["files"]
    names = ["references_N512_NT240_seed0_800.npz", "references_N512_NT240_seed0_4000.npz",
             "references_N512_NT240_validation30.npz"]
    for nm in names:
        assert nm in man and (DATA / nm).exists()
    d8 = np.load(DATA / names[0], mmap_mode="r"); d4 = np.load(DATA / names[1], mmap_mode="r")
    assert list(d8["query_ids"]) == list(d4["query_ids"][:800])
    assert d4["Z"].shape == (4000, 510, 241) and d4["Z"].dtype == np.float64
    k = [0, 399, 799] if not FULL else range(800)
    for i in k:
        assert np.array_equal(d8["Z"][i], d4["Z"][i])
    if FULL:
        for nm in names:
            assert sha256(DATA / nm) == man[nm]["sha256"], nm
