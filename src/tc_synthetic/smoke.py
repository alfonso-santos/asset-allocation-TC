"""Modulo de smoke tests del toolbox de datos sinteticos."""

import numpy as np

from tc_synthetic.diagnostics import (
    compute_basic_diagnostics,
    compute_correlation_diagnostics,
    compute_state_conditioned_diagnostics,
)
from tc_synthetic.generator import (
    generate_static_scenario_data,
    generate_two_state_scenario_data,
)
from tc_synthetic.special_generators import generate_special_structure_data
from tc_synthetic.specs import (
    CopulaSpec,
    MarginalSpec,
    ScenarioSpec,
    StateProcessSpec,
    StructureSpec,
)

__all__ = [
    "run_static_smoke",
    "run_dynamic_smoke",
    "run_special_smoke",
]


def run_static_smoke(seed: int = 123) -> dict[str, object]:
    """Ejecuta un recorrido minimo del generador estatico con diagnosticos basicos."""
    rng = np.random.default_rng(seed)
    n_obs = 500
    n_assets = 4
    structure = StructureSpec(
        kind="block",
        params={
            "block_sizes": [2, 2],
            "rho_within": 0.6,
            "rho_between": 0.1,
        },
    )
    copula = CopulaSpec(kind="gaussian")
    marginal = MarginalSpec(kind="student_t", params={"df": 5.0})

    x = generate_static_scenario_data(n_obs, n_assets, structure, copula, marginal, rng)
    basic = compute_basic_diagnostics(x)
    corr = compute_correlation_diagnostics(x)
    return {
        "x": x,
        "basic_diagnostics": basic,
        "correlation_diagnostics": corr,
    }


def run_dynamic_smoke(seed: int = 123) -> dict[str, object]:
    """Ejecuta un recorrido minimo del generador dinamico con diagnosticos por estado."""
    rng = np.random.default_rng(seed)
    n_obs = 500
    n_assets = 4
    calm_spec = ScenarioSpec(
        name="calm",
        n_obs=n_obs,
        n_assets=n_assets,
        structure=StructureSpec(kind="equicorrelation", params={"rho": 0.2}),
        copula=CopulaSpec(kind="gaussian"),
        marginal=MarginalSpec(kind="gaussian"),
    )
    crisis_spec = ScenarioSpec(
        name="crisis",
        n_obs=n_obs,
        n_assets=n_assets,
        structure=StructureSpec(kind="equicorrelation", params={"rho": 0.7}),
        copula=CopulaSpec(kind="gaussian"),
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.97, 0.03], [0.05, 0.95]], dtype=float),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(calm_spec, crisis_spec, state_spec, rng)
    basic = compute_basic_diagnostics(x)
    state_diag = compute_state_conditioned_diagnostics(x, states)
    return {
        "x": x,
        "states": states,
        "basic_diagnostics": basic,
        "state_conditioned_diagnostics": state_diag,
    }


def run_special_smoke(seed: int = 123) -> dict[str, object]:
    """Ejecuta un recorrido minimo del generador especial con diagnosticos basicos."""
    rng = np.random.default_rng(seed)
    n_obs = 500
    n_assets = 4
    structure = StructureSpec(
        kind="nonlinear_redundancy",
        params={"group_sizes": [2, 2]},
    )
    marginal = MarginalSpec(kind="gaussian")

    x = generate_special_structure_data(n_obs, n_assets, structure, marginal, rng)
    basic = compute_basic_diagnostics(x)
    corr = compute_correlation_diagnostics(x)
    return {
        "x": x,
        "basic_diagnostics": basic,
        "correlation_diagnostics": corr,
    }
