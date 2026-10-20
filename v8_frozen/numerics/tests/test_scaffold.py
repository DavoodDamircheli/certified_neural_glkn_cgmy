"""Import smoke tests for the repository scaffold.

These tests only check that every module in the pipeline is importable and
documented. They do not exercise any numerics, which have not been
implemented yet.
"""

import importlib

import pytest

MODULES = [
    "cgmy.reference_solver",
    "cgmy.operators",
    "cgmy.lifting",
    "cgmy.splines",
    "cgmy.dirichlet_bc",
    "certification.residuals",
    "certification.mass_indicator",
    "certification.riesz_solver",
    "certification.guaranteed_certificate",
    "certification.constants",
    "learning.mlp_coeff",
    "learning.deeponet_coeff",
    "learning.pod",
    "learning.fno",
    "learning.projected_fno",
    "learning.losses",
    "data.sampling",
    "data.payoff_family",
    "data.dataset",
    "correction.krylov",
    "common.io",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_and_is_documented(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} is missing a module docstring"
