"""Pruebas funcionales para los wrappers de informacion basados en RBIG."""

import numpy as np
import pytest

pytest.importorskip("rbig")

from tc_synthetic.information import (
    estimate_rbig_joint_entropy,
    estimate_rbig_mutual_information,
    estimate_rbig_total_correlation,
)


def test_rbig_total_correlation_is_near_zero_for_independence() -> None:
    """Verifica que la TC estimada es pequena para variables independientes."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((500, 3))

    tc = estimate_rbig_total_correlation(x, max_layers=50)

    assert tc < 0.1


def test_rbig_total_correlation_increases_with_correlation() -> None:
    """Verifica que la TC aumenta al introducir dependencia entre columnas."""
    rng = np.random.default_rng(123)
    x_indep = rng.standard_normal((500, 3))
    correlation = np.array(
        [
            [1.0, 0.7, 0.7],
            [0.7, 1.0, 0.7],
            [0.7, 0.7, 1.0],
        ]
    )
    x_corr = rng.multivariate_normal(np.zeros(3), correlation, size=500)

    tc_indep = estimate_rbig_total_correlation(x_indep, max_layers=50)
    tc_corr = estimate_rbig_total_correlation(x_corr, max_layers=50)

    assert tc_corr > tc_indep


def test_rbig_joint_entropy_returns_finite_float() -> None:
    """Verifica que la entropia conjunta devuelve un escalar finito."""
    rng = np.random.default_rng(123)
    correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    x = rng.multivariate_normal(np.zeros(2), correlation, size=500)

    h_joint = estimate_rbig_joint_entropy(x, max_layers=50)

    assert isinstance(h_joint, float)
    assert np.isfinite(h_joint)


def test_rbig_mutual_information_is_near_zero_for_independence() -> None:
    """Verifica que la MI estimada es pequena para bloques independientes."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((500, 1))
    y = rng.standard_normal((500, 1))

    mi = estimate_rbig_mutual_information(x, y, max_layers=50)

    assert mi < 0.1


def test_rbig_mutual_information_is_positive_under_dependence() -> None:
    """Verifica que la MI aumenta cuando los bloques comparten senal."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((500, 1))
    y = x + 0.2 * rng.standard_normal((500, 1))

    mi = estimate_rbig_mutual_information(x, y, max_layers=50)

    assert mi > 0.1


def test_rbig_estimators_are_reproducible_for_same_input() -> None:
    """Verifica reproducibilidad para el mismo input y los mismos parametros."""
    x = np.random.default_rng(123).standard_normal((500, 3))

    tc_a = estimate_rbig_total_correlation(x, max_layers=50)
    tc_b = estimate_rbig_total_correlation(x, max_layers=50)

    assert tc_a == tc_b
