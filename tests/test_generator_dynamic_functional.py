"""Pruebas funcionales para el generador dinamico de ``tc_synthetic``."""

import numpy as np
import pytest
from scipy.stats import norm

from tc_synthetic.generator import generate_two_state_scenario_data
from tc_synthetic.specs import CopulaSpec, MarginalSpec, ScenarioSpec, StateProcessSpec, StructureSpec


N_OBS = 40_000
N_ASSETS = 3



def _empirical_corr(x: np.ndarray) -> np.ndarray:
    """Calcula la matriz de correlacion empirica por columnas."""
    return np.corrcoef(x, rowvar=False)



def _empirical_kurtosis(x: np.ndarray) -> float:
    """Calcula la curtosis de Pearson de una muestra."""
    mean = np.mean(x)
    var = np.var(x)
    return float(np.mean((x - mean) ** 4) / (var**2))



def _gaussianize_sample_ranks(x: np.ndarray) -> np.ndarray:
    """Gaussianiza cada columna via uniforms empiricos construidos por ranks."""
    n_obs = x.shape[0]
    columns: list[np.ndarray] = []
    for column_index in range(x.shape[1]):
        order = np.argsort(x[:, column_index], kind="mergesort")
        ranks = np.empty(n_obs, dtype=float)
        ranks[order] = np.arange(1, n_obs + 1, dtype=float)
        uniforms = (ranks - 0.5) / n_obs
        columns.append(norm.ppf(uniforms))
    return np.column_stack(columns).astype(float, copy=False)



def _make_scenario(
    *,
    name: str,
    n_obs: int = N_OBS,
    n_assets: int = N_ASSETS,
    rho: float = 0.4,
    marginal: MarginalSpec | None = None,
    copula: CopulaSpec | None = None,
) -> ScenarioSpec:
    """Construye un ``ScenarioSpec`` pequeno y limpio para los tests dinamicos."""
    return ScenarioSpec(
        name=name,
        n_assets=n_assets,
        n_obs=n_obs,
        structure=StructureSpec(kind="equicorrelation", params={"rho": rho}),
        copula=copula or CopulaSpec(kind="gaussian"),
        marginal=marginal or MarginalSpec(kind="gaussian"),
    )



def _mean_off_diagonal(correlation: np.ndarray) -> float:
    """Calcula el promedio de los terminos fuera de la diagonal."""
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    return float(np.mean(correlation[mask]))



def test_dynamic_generator_returns_valid_shapes_and_binary_states() -> None:
    """Verifica shapes, soporte de estados y finitud de la salida dinamica."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.97, 0.03], [0.03, 0.97]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )

    assert x.shape == (N_OBS, N_ASSETS)
    assert states.shape == (N_OBS,)
    assert set(np.unique(states)).issubset({0, 1})
    assert np.all(np.isfinite(x))



def test_different_marginals_across_states_are_reflected_in_state_conditioned_subsamples() -> None:
    """Verifica que la cola pesada del estado crisis aparece en la submuestra condicionada."""
    calm_spec = _make_scenario(name="calm", rho=0.4, marginal=MarginalSpec(kind="gaussian"))
    crisis_spec = _make_scenario(
        name="crisis",
        rho=0.4,
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.97, 0.03], [0.03, 0.97]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )
    x_calm = x[states == 0, 0]
    x_crisis = x[states == 1, 0]

    assert x_calm.size > 500
    assert x_crisis.size > 500
    assert _empirical_kurtosis(x_crisis) > _empirical_kurtosis(x_calm)



def test_different_dependence_across_states_is_reflected_in_state_conditioned_correlations() -> None:
    """Verifica que la dependencia empirica aumenta en el estado crisis."""
    calm_spec = _make_scenario(name="calm", rho=0.2, marginal=MarginalSpec(kind="gaussian"))
    crisis_spec = _make_scenario(name="crisis", rho=0.8, marginal=MarginalSpec(kind="gaussian"))
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.97, 0.03], [0.03, 0.97]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )
    calm_corr = _empirical_corr(x[states == 0])
    crisis_corr = _empirical_corr(x[states == 1])

    assert _mean_off_diagonal(crisis_corr) > _mean_off_diagonal(calm_corr) + 0.2



def test_gaussianized_state_conditioned_dependence_is_preserved_under_non_gaussian_marginals() -> None:
    """Verifica la diferencia de dependencia entre estados bajo marginales no gaussianas."""
    student_t = MarginalSpec(kind="student_t", params={"df": 5.0})
    calm_spec = _make_scenario(name="calm", rho=0.2, marginal=student_t)
    crisis_spec = _make_scenario(name="crisis", rho=0.7, marginal=student_t)
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.97, 0.03], [0.03, 0.97]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )
    calm_corr = _empirical_corr(_gaussianize_sample_ranks(x[states == 0]))
    crisis_corr = _empirical_corr(_gaussianize_sample_ranks(x[states == 1]))

    assert _mean_off_diagonal(crisis_corr) > _mean_off_diagonal(calm_corr) + 0.15



def test_dynamic_generator_is_reproducible() -> None:
    """Verifica reproducibilidad exacta del generador dinamico."""
    calm_spec = _make_scenario(name="calm", rho=0.2)
    crisis_spec = _make_scenario(
        name="crisis",
        rho=0.7,
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.97, 0.03], [0.03, 0.97]]),
            "initial_state": 0,
        },
        enabled=True,
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    x_a, states_a = generate_two_state_scenario_data(calm_spec, crisis_spec, state_spec, rng_a)
    x_b, states_b = generate_two_state_scenario_data(calm_spec, crisis_spec, state_spec, rng_b)

    assert np.array_equal(x_a, x_b)
    assert np.array_equal(states_a, states_b)



def test_identity_transition_with_initial_state_zero_uses_only_calm_scenario() -> None:
    """Verifica que la identidad con estado inicial calm mantiene solo el escenario calm."""
    calm_spec = _make_scenario(name="calm", marginal=MarginalSpec(kind="gaussian"))
    crisis_spec = _make_scenario(
        name="crisis",
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.eye(2, dtype=float),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )

    assert np.array_equal(np.unique(states), np.array([0]))
    assert _empirical_kurtosis(x[:, 0]) < 3.5



def test_identity_transition_with_initial_state_one_uses_only_crisis_scenario() -> None:
    """Verifica que la identidad con estado inicial crisis mantiene solo el escenario crisis."""
    calm_spec = _make_scenario(name="calm", marginal=MarginalSpec(kind="gaussian"))
    crisis_spec = _make_scenario(
        name="crisis",
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.eye(2, dtype=float),
            "initial_state": 1,
        },
        enabled=True,
    )

    x, states = generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )

    assert np.array_equal(np.unique(states), np.array([1]))
    assert _empirical_kurtosis(x[:, 0]) > 3.5
