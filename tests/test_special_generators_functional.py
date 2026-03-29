"""Pruebas funcionales para los generadores especiales de ``tc_synthetic``."""

import numpy as np
import pytest

from tc_synthetic.special_generators import (
    generate_nonlinear_redundancy_data,
    generate_special_structure_data,
)
from tc_synthetic.specs import MarginalSpec, StructureSpec


N_OBS = 40_000


def _make_structure() -> StructureSpec:
    """Construye una estructura simple de redundancia no lineal para tests."""
    return StructureSpec(
        kind="nonlinear_redundancy",
        params={"group_sizes": [2, 2]},
    )


def _empirical_corr(x: np.ndarray) -> np.ndarray:
    """Calcula la matriz de correlacion empirica por columnas."""
    return np.corrcoef(x, rowvar=False)


def _quadratic_dependence_score(x: np.ndarray, y: np.ndarray) -> float:
    """Mide dependencia no lineal simple via la correlacion entre ``x**2`` e ``y``."""
    return float(abs(np.corrcoef(x**2, y)[0, 1]))


def _mean_off_diagonal_block(corr: np.ndarray, indices: list[int]) -> float:
    """Calcula el promedio fuera de la diagonal dentro de un bloque."""
    values: list[float] = []
    for i in indices:
        for j in indices:
            if i != j:
                values.append(float(corr[i, j]))
    return float(np.mean(values))


def _mean_cross_block_abs_corr(corr: np.ndarray, left: list[int], right: list[int]) -> float:
    """Calcula el promedio del valor absoluto de las correlaciones entre dos bloques."""
    values: list[float] = []
    for i in left:
        for j in right:
            values.append(float(abs(corr[i, j])))
    return float(np.mean(values))


def _empirical_kurtosis(x: np.ndarray) -> float:
    """Calcula la curtosis de Pearson de una muestra."""
    mean = np.mean(x)
    var = np.var(x)
    return float(np.mean((x - mean) ** 4) / (var**2))


def test_nonlinear_redundancy_generator_returns_valid_finite_sample() -> None:
    """Verifica shape, dtype y finitud del generador no lineal."""
    sample = generate_nonlinear_redundancy_data(
        N_OBS,
        4,
        _make_structure(),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )

    assert sample.shape == (N_OBS, 4)
    assert np.issubdtype(sample.dtype, np.floating)
    assert np.all(np.isfinite(sample))


def test_within_group_dependence_is_stronger_than_cross_group_dependence() -> None:
    """Verifica que, en promedio, la dependencia intra-grupo supera la cruzada."""
    sample = generate_nonlinear_redundancy_data(
        N_OBS,
        4,
        _make_structure(),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )
    corr = _empirical_corr(sample)
    within_left = abs(_mean_off_diagonal_block(corr, [0, 1]))
    within_right = abs(_mean_off_diagonal_block(corr, [2, 3]))
    cross = _mean_cross_block_abs_corr(corr, [0, 1], [2, 3])

    assert np.mean([within_left, within_right]) > cross


def test_nonlinear_dependence_score_is_materially_positive_within_a_group() -> None:
    """Verifica que el score cuadratico intra-grupo es claramente positivo."""
    sample = generate_nonlinear_redundancy_data(
        N_OBS,
        4,
        _make_structure(),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )
    score = _quadratic_dependence_score(sample[:, 0], sample[:, 1])

    assert score > 0.20


def test_nonlinear_dependence_score_within_group_exceeds_cross_group_pair() -> None:
    """Verifica que la dependencia no lineal intra-grupo supera a la cruzada."""
    sample = generate_nonlinear_redundancy_data(
        N_OBS,
        4,
        _make_structure(),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )
    score_within = _quadratic_dependence_score(sample[:, 0], sample[:, 1])
    score_cross = _quadratic_dependence_score(sample[:, 0], sample[:, 2])

    assert score_within > score_cross + 0.05


def test_marginal_transformation_yields_standardized_like_columns_under_gaussian_marginal() -> None:
    """Verifica medias y varianzas razonables con marginal gaussiana."""
    sample = generate_nonlinear_redundancy_data(
        N_OBS,
        4,
        _make_structure(),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )

    for column_index in range(sample.shape[1]):
        mean = np.mean(sample[:, column_index])
        var = np.var(sample[:, column_index])
        assert abs(mean) < 0.03
        assert abs(var - 1.0) < 0.08


def test_student_t_marginal_preserves_heavier_tails_in_special_generator() -> None:
    """Verifica curtosis elevada con marginal Student-t en el generador especial."""
    sample = generate_nonlinear_redundancy_data(
        N_OBS,
        4,
        _make_structure(),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        np.random.default_rng(123),
    )

    assert _empirical_kurtosis(sample[:, 0]) > 3.5


def test_dispatcher_special_generator_matches_direct_nonlinear_generator() -> None:
    """Verifica igualdad exacta entre dispatcher y generador directo."""
    structure = _make_structure()
    marginal = MarginalSpec(kind="gaussian")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    dispatched = generate_special_structure_data(N_OBS, 4, structure, marginal, rng_a)
    direct = generate_nonlinear_redundancy_data(N_OBS, 4, structure, marginal, rng_b)

    assert np.array_equal(dispatched, direct)


def test_nonlinear_redundancy_generator_is_reproducible() -> None:
    """Verifica reproducibilidad exacta del generador directo."""
    structure = _make_structure()
    marginal = MarginalSpec(kind="gaussian")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = generate_nonlinear_redundancy_data(N_OBS, 4, structure, marginal, rng_a)
    sample_b = generate_nonlinear_redundancy_data(N_OBS, 4, structure, marginal, rng_b)

    assert np.array_equal(sample_a, sample_b)
