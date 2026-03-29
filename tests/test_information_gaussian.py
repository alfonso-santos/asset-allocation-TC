"""Pruebas para las funciones gaussianizadas de ``tc_synthetic.information``."""

import numpy as np

from tc_synthetic.information import (
    compute_excess_total_correlation,
    compute_gaussian_total_correlation,
    compute_gaussianized_correlation,
)
from tc_synthetic.special_generators import generate_nonlinear_redundancy_data
from tc_synthetic.specs import MarginalSpec, StructureSpec


def test_gaussianized_correlation_is_symmetric_with_unit_diagonal() -> None:
    """Verifica simetria y diagonal unitaria de la correlacion gaussianizada."""
    x = np.random.default_rng(123).standard_normal((400, 3))

    correlation = compute_gaussianized_correlation(x)

    assert correlation.shape == (3, 3)
    np.testing.assert_allclose(correlation, correlation.T)
    np.testing.assert_allclose(np.diag(correlation), np.ones(3), atol=1e-6)


def test_gaussian_total_correlation_is_near_zero_under_independence() -> None:
    """Verifica que la TC gaussiana es pequena para variables independientes."""
    x = np.random.default_rng(123).standard_normal((400, 3))

    tc_gauss = compute_gaussian_total_correlation(x)

    assert tc_gauss < 0.05


def test_gaussian_total_correlation_increases_with_correlation() -> None:
    """Verifica que la TC gaussiana aumenta cuando las columnas se correlacionan."""
    rng = np.random.default_rng(123)
    x_independent = rng.standard_normal((400, 3))
    covariance = np.array(
        [
            [1.0, 0.7, 0.7],
            [0.7, 1.0, 0.7],
            [0.7, 0.7, 1.0],
        ]
    )
    x_correlated = rng.multivariate_normal(np.zeros(3), covariance, size=400)

    tc_independent = compute_gaussian_total_correlation(x_independent)
    tc_correlated = compute_gaussian_total_correlation(x_correlated)

    assert tc_correlated > tc_independent


def test_excess_total_correlation_is_near_zero_for_correlated_gaussian_data() -> None:
    """Verifica que el exceso es pequeno en datos gaussianos correlacionados."""
    rng = np.random.default_rng(123)
    covariance = np.array(
        [
            [1.0, 0.7, 0.7],
            [0.7, 1.0, 0.7],
            [0.7, 0.7, 1.0],
        ]
    )
    x = rng.multivariate_normal(np.zeros(3), covariance, size=400)

    xi = compute_excess_total_correlation(x, max_layers=50)

    assert abs(xi) < 0.1


def test_excess_total_correlation_is_positive_for_nonlinear_redundancy() -> None:
    """Verifica que el exceso es positivo cuando hay dependencia no gaussiana."""
    x = generate_nonlinear_redundancy_data(
        400,
        4,
        StructureSpec(kind="nonlinear_redundancy", params={"group_sizes": [2, 2]}),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )

    xi = compute_excess_total_correlation(x, max_layers=50)

    assert xi > 0.0


def test_gaussian_information_functions_are_reproducible() -> None:
    """Verifica reproducibilidad exacta para el mismo input."""
    x = np.random.default_rng(123).standard_normal((400, 3))

    tc_a = compute_gaussian_total_correlation(x)
    tc_b = compute_gaussian_total_correlation(x)

    assert tc_a == tc_b
