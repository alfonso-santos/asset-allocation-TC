"""Pruebas funcionales para el generador estatico de ``tc_synthetic``."""

import numpy as np
import pytest
from scipy.stats import norm

from tc_synthetic.generator import generate_static_scenario_data
from tc_synthetic.specs import CopulaSpec, MarginalSpec, StructureSpec


N_OBS = 40_000



def _empirical_corr(x: np.ndarray) -> np.ndarray:
    """Calcula la matriz de correlacion empirica por columnas."""
    return np.corrcoef(x, rowvar=False)



def _empirical_skew(x: np.ndarray) -> float:
    """Calcula la asimetria muestral estandarizada."""
    mean = np.mean(x)
    std = np.sqrt(np.var(x))
    return float(np.mean((x - mean) ** 3) / (std**3))



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



def test_static_generator_with_independence_copula_has_near_zero_empirical_dependence() -> None:
    """Verifica dependencia empirica pequena bajo copula de independencia."""
    rng = np.random.default_rng(123)

    sample = generate_static_scenario_data(
        N_OBS,
        3,
        StructureSpec(kind="equicorrelation", params={"rho": 0.6}),
        CopulaSpec(kind="independence"),
        MarginalSpec(kind="gaussian"),
        rng,
    )
    empirical_corr = _empirical_corr(sample)
    off_diagonal = empirical_corr[~np.eye(empirical_corr.shape[0], dtype=bool)]

    assert np.all(np.abs(off_diagonal) < 0.03)



def test_static_generator_with_gaussian_copula_and_gaussian_marginal_matches_target_correlation() -> None:
    """Verifica que la integracion gaussian+gaussian reproduce la correlacion objetivo."""
    rng = np.random.default_rng(123)
    target = np.array(
        [
            [1.0, 0.6, 0.6],
            [0.6, 1.0, 0.6],
            [0.6, 0.6, 1.0],
        ]
    )

    sample = generate_static_scenario_data(
        N_OBS,
        3,
        StructureSpec(kind="equicorrelation", params={"rho": 0.6}),
        CopulaSpec(kind="gaussian"),
        MarginalSpec(kind="gaussian"),
        rng,
    )
    empirical_corr = _empirical_corr(sample)

    assert np.allclose(empirical_corr, target, atol=0.05)



def test_static_generator_with_gaussian_copula_and_student_t_marginal_preserves_heavier_tails() -> None:
    """Verifica colas mas pesadas y varianza razonable con marginal Student-t."""
    rng = np.random.default_rng(123)

    sample = generate_static_scenario_data(
        N_OBS,
        3,
        StructureSpec(kind="equicorrelation", params={"rho": 0.4}),
        CopulaSpec(kind="gaussian"),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        rng,
    )

    assert sample.shape == (N_OBS, 3)
    assert _empirical_kurtosis(sample[:, 0]) > 3.5
    assert abs(np.var(sample[:, 0]) - 1.0) < 0.08



def test_static_generator_with_gaussian_copula_and_skew_normal_marginal_preserves_positive_skew() -> None:
    """Verifica asimetria positiva con marginal skew-normal."""
    rng = np.random.default_rng(123)

    sample = generate_static_scenario_data(
        N_OBS,
        3,
        StructureSpec(kind="equicorrelation", params={"rho": 0.4}),
        CopulaSpec(kind="gaussian"),
        MarginalSpec(kind="skew_normal", params={"shape": 5.0}),
        rng,
    )

    assert _empirical_skew(sample[:, 0]) > 0.15



def test_gaussianized_rank_correlation_of_generated_sample_is_close_to_target_under_non_gaussian_marginals() -> None:
    """Verifica que la dependencia subyacente se conserva tras gaussianizar por ranks."""
    rng = np.random.default_rng(123)
    target = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )

    sample = generate_static_scenario_data(
        N_OBS,
        3,
        StructureSpec(kind="equicorrelation", params={"rho": 0.5}),
        CopulaSpec(kind="gaussian"),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        rng,
    )
    gaussianized = _gaussianize_sample_ranks(sample)
    empirical_corr = _empirical_corr(gaussianized)

    assert np.allclose(empirical_corr, target, atol=0.06)



def test_static_generator_is_reproducible() -> None:
    """Verifica reproducibilidad exacta con la misma seed y los mismos specs."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.4})
    copula = CopulaSpec(kind="gaussian")
    marginal = MarginalSpec(kind="student_t", params={"df": 5.0})

    sample_a = generate_static_scenario_data(N_OBS, 3, structure, copula, marginal, rng_a)
    sample_b = generate_static_scenario_data(N_OBS, 3, structure, copula, marginal, rng_b)

    assert np.array_equal(sample_a, sample_b)



@pytest.mark.parametrize(
    ("copula", "marginal"),
    [
        (CopulaSpec(kind="gaussian"), MarginalSpec(kind="gaussian")),
        (CopulaSpec(kind="gaussian"), MarginalSpec(kind="student_t", params={"df": 5.0})),
        (CopulaSpec(kind="independence"), MarginalSpec(kind="gaussian")),
    ],
)
def test_no_nan_or_inf_in_static_generated_sample(
    copula: CopulaSpec,
    marginal: MarginalSpec,
) -> None:
    """Verifica finitud total en varias combinaciones del generador estatico."""
    rng = np.random.default_rng(123)

    sample = generate_static_scenario_data(
        N_OBS,
        3,
        StructureSpec(kind="equicorrelation", params={"rho": 0.4}),
        copula,
        marginal,
        rng,
    )

    assert np.all(np.isfinite(sample))
