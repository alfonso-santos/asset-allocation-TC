"""Pruebas funcionales para las copulas de ``tc_synthetic``."""

import numpy as np
import pytest
from scipy.stats import norm

from tc_synthetic.copulas import (
    gaussian_latent_to_uniform,
    sample_clayton_copula,
    sample_gaussian_copula,
    sample_grouped_t_copula,
    sample_independence_copula,
    sample_t_copula,
)


N_OBS = 40_000



def _gaussianize_uniforms(u: np.ndarray) -> np.ndarray:
    """Convierte uniforms a normales estandar mediante ``norm.ppf``."""
    return norm.ppf(u).astype(float, copy=False)



def _empirical_corr(x: np.ndarray) -> np.ndarray:
    """Calcula la matriz de correlacion empirica por columnas."""
    return np.corrcoef(x, rowvar=False)



def _lower_tail_coincidence(u: np.ndarray, q: float = 0.05) -> float:
    """Calcula ``P(U2 <= q | U1 <= q)`` para una muestra bivariante."""
    mask = u[:, 0] <= q
    assert mask.sum() > 0
    return float(np.mean(u[mask, 1] <= q))



def _upper_tail_coincidence(u: np.ndarray, q: float = 0.95) -> float:
    """Calcula ``P(U2 >= q | U1 >= q)`` para una muestra bivariante."""
    mask = u[:, 0] >= q
    assert mask.sum() > 0
    return float(np.mean(u[mask, 1] >= q))



def test_gaussian_latent_to_uniform_preserves_shape_and_open_unit_bounds() -> None:
    """Verifica shape y cotas estrictas para la transformacion gaussiana a uniforms."""
    rng = np.random.default_rng(123)
    latent = rng.standard_normal(size=(2_000, 3))

    result = gaussian_latent_to_uniform(latent)

    assert result.shape == latent.shape
    assert np.all(result > 0.0)
    assert np.all(result < 1.0)



def test_independence_copula_has_near_zero_empirical_correlations() -> None:
    """Verifica que la copula de independencia induce correlaciones pequenas."""
    rng = np.random.default_rng(123)

    sample = sample_independence_copula(N_OBS, 3, rng)
    empirical_corr = _empirical_corr(_gaussianize_uniforms(sample))
    off_diagonal = empirical_corr[~np.eye(empirical_corr.shape[0], dtype=bool)]

    assert np.all(np.abs(off_diagonal) < 0.03)



def test_gaussian_copula_matches_target_correlation_approximately() -> None:
    """Verifica que la copula gaussiana reproduce aproximadamente la correlacion objetivo."""
    rng = np.random.default_rng(123)
    target_corr = np.array(
        [
            [1.0, 0.6, 0.2],
            [0.6, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )

    sample = sample_gaussian_copula(N_OBS, target_corr, rng)
    empirical_corr = _empirical_corr(_gaussianize_uniforms(sample))

    assert np.allclose(empirical_corr, target_corr, atol=0.05)



def test_t_copula_matches_target_correlation_approximately() -> None:
    """Verifica que la t copula reproduce aproximadamente la correlacion objetivo."""
    rng = np.random.default_rng(123)
    target_corr = np.array(
        [
            [1.0, 0.6, 0.2],
            [0.6, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )

    sample = sample_t_copula(N_OBS, target_corr, 5.0, rng)
    empirical_corr = _empirical_corr(_gaussianize_uniforms(sample))

    assert np.allclose(empirical_corr, target_corr, atol=0.06)



def test_t_copula_has_stronger_joint_tail_behavior_than_gaussian_copula() -> None:
    """Verifica que la t copula muestra mayor coincidencia conjunta en las colas."""
    correlation = np.array([[1.0, 0.7], [0.7, 1.0]])
    gaussian_sample = sample_gaussian_copula(N_OBS, correlation, np.random.default_rng(123))
    t_sample = sample_t_copula(N_OBS, correlation, 5.0, np.random.default_rng(123))

    lower_gaussian = _lower_tail_coincidence(gaussian_sample)
    upper_gaussian = _upper_tail_coincidence(gaussian_sample)
    lower_t = _lower_tail_coincidence(t_sample)
    upper_t = _upper_tail_coincidence(t_sample)

    assert lower_t > lower_gaussian
    assert upper_t > upper_gaussian



def test_clayton_copula_shows_stronger_lower_tail_than_upper_tail_dependence() -> None:
    """Verifica que Clayton tiene dependencia mas marcada en la cola inferior."""
    rng = np.random.default_rng(123)

    sample = sample_clayton_copula(N_OBS, 2, 2.0, rng)
    lower = _lower_tail_coincidence(sample)
    upper = _upper_tail_coincidence(sample)

    assert lower > upper
    assert lower > 0.12



def test_grouped_t_copula_returns_valid_uniforms_with_positive_dependence_structure() -> None:
    """Verifica shape, cotas y dependencia positiva basica en grouped-t."""
    rng = np.random.default_rng(123)
    correlation = np.array(
        [
            [1.0, 0.5, 0.2, 0.1],
            [0.5, 1.0, 0.15, 0.1],
            [0.2, 0.15, 1.0, 0.45],
            [0.1, 0.1, 0.45, 1.0],
        ]
    )
    group_assignments = np.array([0, 0, 1, 1])
    group_dfs = {0: 4.0, 1: 10.0}

    sample = sample_grouped_t_copula(N_OBS, correlation, group_assignments, group_dfs, rng)
    empirical_corr = _empirical_corr(_gaussianize_uniforms(sample))

    assert sample.shape == (N_OBS, 4)
    assert np.all(sample > 0.0)
    assert np.all(sample < 1.0)
    assert np.allclose(np.diag(empirical_corr), 1.0, atol=1e-12)
    assert empirical_corr[0, 1] > 0.2
    assert empirical_corr[2, 3] > 0.2



def test_no_nan_or_inf_in_any_copula_sample() -> None:
    """Verifica finitud total en muestras de todas las copulas soportadas."""
    gaussian_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    grouped_corr = np.array(
        [
            [1.0, 0.4, 0.2, 0.1],
            [0.4, 1.0, 0.15, 0.1],
            [0.2, 0.15, 1.0, 0.35],
            [0.1, 0.1, 0.35, 1.0],
        ]
    )
    samples = [
        sample_independence_copula(1_000, 3, np.random.default_rng(123)),
        sample_gaussian_copula(1_000, gaussian_corr, np.random.default_rng(123)),
        sample_t_copula(1_000, gaussian_corr, 5.0, np.random.default_rng(123)),
        sample_grouped_t_copula(
            1_000,
            grouped_corr,
            np.array([0, 0, 1, 1]),
            {0: 4.0, 1: 10.0},
            np.random.default_rng(123),
        ),
        sample_clayton_copula(1_000, 2, 2.0, np.random.default_rng(123)),
    ]

    for sample in samples:
        assert np.all(np.isfinite(sample))
