"""Instrumentation proving the free workflow makes no hidden legacy call
(roadmap §2 table row "Full matrix fails but free matrix passes"; gate G1).

Inside ``forbid_legacy_calls()``:

- legacy entry points raise ``LegacyCallError``: the raw residual
  (``certification.residuals.compute_all_residuals``/``compute_cn_residual``),
  the raw Krylov correction (``correction.krylov.correct_trajectory``), the
  legacy pipeline (``pipeline.predict_certify_correct.predict_certify_correct``)
  and the mass indicator;
- raw full-space S_h actions (``OperatorCache.apply_S``) raise -- the free
  operator builds S_I from raw A/A^T actions via inject/restrict and never needs
  them;
- raw ``OperatorCache.apply_M_inv`` raises except inside the two labelled scopes
  that legitimately use the raw L2 projection: the model input encoding
  (``data.payoff_family.european_put``) and the discrete initial target
  (``correction.krylov.compute_initial_coefficients``, the trusted c^0);
- the generic certification functions (stability constants, kappa, Riesz PCG)
  are wrapped to log the operator class/dimension they receive and to raise if
  handed a raw ``OperatorCache``.

The returned log is evidence, written into the G1 traces.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from cgmy.operators import OperatorCache
from certification.constrained import FreeOperatorCache


class LegacyCallError(RuntimeError):
    pass


_FORBIDDEN = (
    ("certification.residuals", "compute_all_residuals"),
    ("certification.residuals", "compute_cn_residual"),
    ("correction.krylov", "correct_trajectory"),
    ("pipeline.predict_certify_correct", "predict_certify_correct"),
    ("pipeline.predict_certify_correct", "compute_all_residuals"),
    ("pipeline.predict_certify_correct", "correct_trajectory"),
    ("certification.mass_indicator", "eta_mass_indicator"),
)
# (module, attribute, index of the ops argument) for generic certification functions.
_AUDITED = (
    ("certification.free_problem", "compute_stability_constants", 1),
    ("certification.free_problem", "kappa_operator_norm", 1),
    ("certification.guaranteed_certificate", "solve_riesz_pcg", 2),
)
_RAW_MINV_SCOPES = (
    ("data.payoff_family", "european_put", "model input encoding psi (raw L2 projection)"),
    ("correction.krylov", "compute_initial_coefficients", "discrete initial target c_h^0 (trusted convention)"),
)


@contextlib.contextmanager
def forbid_legacy_calls() -> Iterator[list[dict[str, Any]]]:
    import importlib

    log: list[dict[str, Any]] = []
    saved: list[tuple[Any, str, Any]] = []
    scope: list[str] = []

    def patch(obj: Any, name: str, new: Any) -> None:
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)

    for mod_name, attr in _FORBIDDEN:
        mod = importlib.import_module(mod_name)

        def _raise(*a: Any, _n: str = f"{mod_name}.{attr}", **k: Any) -> Any:
            log.append({"call": _n, "result": "BLOCKED"})
            raise LegacyCallError(f"legacy call {_n} inside the free workflow")

        patch(mod, attr, _raise)

    for mod_name, attr, idx in _AUDITED:
        mod = importlib.import_module(mod_name)
        orig = getattr(mod, attr)

        def _wrapped(*a: Any, _orig: Any = orig, _n: str = f"{mod_name}.{attr}", _i: int = idx, **k: Any) -> Any:
            ops = a[_i] if len(a) > _i else k.get("ops")
            entry = {"call": _n, "ops_class": type(ops).__name__, "dim": getattr(ops, "n_dof", None)}
            log.append(entry)
            if isinstance(ops, OperatorCache) or not isinstance(ops, FreeOperatorCache):
                entry["result"] = "BLOCKED"
                raise LegacyCallError(f"{_n} received {type(ops).__name__}, not FreeOperatorCache")
            return _orig(*a, **k)

        patch(mod, attr, _wrapped)

    orig_S = OperatorCache.apply_S
    orig_Minv = OperatorCache.apply_M_inv

    def _raw_S(self: OperatorCache, *a: Any, **k: Any) -> Any:
        log.append({"call": "OperatorCache.apply_S (raw full S_h)", "result": "BLOCKED"})
        raise LegacyCallError("raw full-space S_h action inside the free workflow")

    def _raw_Minv(self: OperatorCache, *a: Any, **k: Any) -> Any:
        if not scope:
            log.append({"call": "OperatorCache.apply_M_inv (raw M_h^-1)", "result": "BLOCKED"})
            raise LegacyCallError("raw M_h^{-1} outside the labelled raw-projection scopes")
        log.append({"call": "OperatorCache.apply_M_inv (raw M_h^-1)", "scope": scope[-1], "result": "allowed"})
        return orig_Minv(self, *a, **k)

    patch(OperatorCache, "apply_S", _raw_S)
    patch(OperatorCache, "apply_M_inv", _raw_Minv)

    for mod_name, attr, label in _RAW_MINV_SCOPES:
        mod = importlib.import_module(mod_name)
        orig = getattr(mod, attr)

        def _scoped(*a: Any, _orig: Any = orig, _label: str = label, **k: Any) -> Any:
            scope.append(_label)
            try:
                return _orig(*a, **k)
            finally:
                scope.pop()

        patch(mod, attr, _scoped)

    try:
        yield log
    finally:
        for obj, name, orig in reversed(saved):
            setattr(obj, name, orig)


def summarize(log: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_events": len(log), "blocked": [e for e in log if e.get("result") == "BLOCKED"]}
    counts: dict[str, int] = {}
    for e in log:
        key = f"{e['call']} [{e.get('ops_class', e.get('scope', ''))}, dim={e.get('dim', '')}]"
        counts[key] = counts.get(key, 0) + 1
    out["calls"] = counts
    return out
