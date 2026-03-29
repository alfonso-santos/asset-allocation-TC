"""Pruebas funcionales para las marginales de ``tc_synthetic``."""

import numpy as np
import pytest

from tc_synthetic.marginals import (
    apply_standard_normal_inverse_cdf,
    apply_standardized_skew_normal_inverse_cdf,
    apply_standardized_student_t_inverse_cdf,
    sample_heterogeneous_marginals,
    sample_standard_normal_marginal,
    sample_standardized_skew_normal_marginal,
    sample_standardized_student_t_marginal,
)
from tc_synthetic.specs import MarginalSpec


N_OBS = 50_000
MEAN_TOL = 0.02
VAR_TOL = 0.05



def _sample_open_unit_uniforms(rng: np.random.Generator, size: int) -> np.ndarray:
    """Genera uniforms estrictamente en ``(0, 1)``."""
    return rng.uniform(
        low=np.nextafter(0.0, 1.0),
        high=np.nextafter(1.0, 0.0),
        size=size,
    )



def _assert_standardized(sample: np.ndarray) -> None:
    """Verifica media y varianza cercanas a las de una variable estandarizada."""
    mean = np.mean(sample)
    var = np.var(sample)
    assert abs(mean) < MEAN_TOL
    assert abs(var - 1.0) < VAR_TOL



def _empirical_kurtosis(sample: np.ndarray) -> float:
    """Calcula la curtosis empirica de Pearson."""
    mean = np.mean(sample)
    var = np.var(sample)
    centered = sample - mean
    return np.mean(centered**4) / (var**2)



def _empirical_skew(sample: np.ndarray) -> float:
    """Calcula la asimetria empirica estandarizada."""
    mean = np.mean(sample)
    std = np.sqrt(np.var(sample))
    centered = sample - mean
    return np.mean(centered**3) / (std**3)



def test_gaussian_marginal_has_standardized_moments() -> None:
    """Verifica que la marginal gaussiana tiene media y varianza coherentes."""
    rng = np.random.default_rng(123)

    sample = sample_standard_normal_marginal(N_OBS, rng)

    _assert_standardized(sample)



def test_standardized_student_t_marginal_has_unit_variance_and_heavier_tails_than_normal() -> None:
    """Verifica que la Student-t estandarizada conserva varianza unitaria y colas pesadas."""
    rng = np.random.default_rng(123)

    student_sample = sample_standardized_student_t_marginal(N_OBS, 5.0, rng)
    normal_sample = sample_standard_normal_marginal(N_OBS, rng)

    _assert_standardized(student_sample)
    assert _empirical_kurtosis(student_sample) > _empirical_kurtosis(normal_sample)



def test_standardized_skew_normal_marginal_has_unit_variance_and_positive_skew() -> None:
    """Verifica que la skew-normal estandarizada mantiene asimetria positiva."""
    rng = np.random.default_rng(123)

    sample = sample_standardized_skew_normal_marginal(N_OBS, 5.0, rng)

    _assert_standardized(sample)
    assert _empirical_skew(sample) > 0.2



def test_gaussian_inverse_cdf_produces_standard_normal_moments() -> None:
    """Verifica que la inversa gaussiana aplicada a uniforms produce una normal estandar."""
    rng = np.random.default_rng(123)
    uniforms = _sample_open_unit_uniforms(rng, N_OBS)

    sample = apply_standard_normal_inverse_cdf(uniforms)

    _assert_standardized(sample)



def test_student_t_inverse_cdf_produces_unit_variance_and_heavier_tails_than_normal() -> None:
    """Verifica que la inversa Student-t produce colas mas pesadas que la normal."""
    rng = np.random.default_rng(123)
    uniforms = _sample_open_unit_uniforms(rng, N_OBS)

    student_sample = apply_standardized_student_t_inverse_cdf(uniforms, 5.0)
    normal_sample = apply_standard_normal_inverse_cdf(uniforms)

    assert abs(np.var(student_sample) - 1.0) < VAR_TOL
    assert _empirical_kurtosis(student_sample) > _empirical_kurtosis(normal_sample)



def test_heterogeneous_marginals_produce_standardized_columns() -> None:
    """Verifica que cada columna heterogenea mantiene media y varianza coherentes."""
    rng = np.random.default_rng(123)
    specs = [
        MarginalSpec(kind="gaussian"),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        MarginalSpec(kind="skew_normal", params={"shape": 5.0}),
    ]

    sample = sample_heterogeneous_marginals(N_OBS, specs, rng)

    assert sample.shape == (N_OBS, 3)
    for column_index in range(sample.shape[1]):
        _assert_standardized(sample[:, column_index])



def test_marginal_generators_and_inverse_transforms_do_not_produce_nan_or_inf() -> None:
    """Verifica robustez basica: ninguna marginal produce ``NaN`` o infinitos."""
    rng = np.random.default_rng(123)
    uniforms = _sample_open_unit_uniforms(rng, N_OBS)

    gaussian_sample = sample_standard_normal_marginal(N_OBS, rng)
    student_sample = sample_standardized_student_t_marginal(N_OBS, 5.0, rng)
    skew_sample = sample_standardized_skew_normal_marginal(N_OBS, 5.0, rng)
    gaussian_inverse = apply_standard_normal_inverse_cdf(uniforms)
    student_inverse = apply_standardized_student_t_inverse_cdf(uniforms, 5.0)
    skew_inverse = apply_standardized_skew_normal_inverse_cdf(uniforms, 5.0)
    heterogeneous = sample_heterogeneous_marginals(
        N_OBS,
        [
            MarginalSpec(kind="gaussian"),
            MarginalSpec(kind="student_t", params={"df": 5.0}),
            MarginalSpec(kind="skew_normal", params={"shape": 5.0}),
        ],
        rng,
    )

    for sample in [
        gaussian_sample,
        student_sample,
        skew_sample,
        gaussian_inverse,
        student_inverse,
        skew_inverse,
        heterogeneous,
    ]:
        assert np.all(np.isfinite(sample))
