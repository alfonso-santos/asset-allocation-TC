"""Pruebas funcionales para estructuras dentro del generador estatico."""

import numpy as np
import pytest

from tc_synthetic.generator import generate_static_scenario_data
from tc_synthetic.specs import CopulaSpec, MarginalSpec, StructureSpec


N_OBS = 40_000
GAUSSIAN_COPULA = CopulaSpec(kind="gaussian")
GAUSSIAN_MARGINAL = MarginalSpec(kind="gaussian")


def _empirical_corr(x: np.ndarray) -> np.ndarray:
    """Calcula la matriz de correlacion empirica por columnas."""
    return np.corrcoef(x, rowvar=False)


def _mean_block_corr(corr: np.ndarray, indices: list[int]) -> float:
    """Calcula la media de correlaciones dentro de un bloque sin diagonal."""
    values: list[float] = []
    for i in indices:
        for j in indices:
            if i != j:
                values.append(float(corr[i, j]))
    return float(np.mean(values))


def _mean_cross_corr(corr: np.ndarray, block_a: list[int], block_b: list[int]) -> float:
    """Calcula la media de correlaciones entre dos bloques."""
    values: list[float] = []
    for i in block_a:
        for j in block_b:
            values.append(float(corr[i, j]))
    return float(np.mean(values))


def test_block_structure_has_stronger_within_block_than_between_block_dependence() -> None:
    """Verifica el patron relativo esperado en la estructura por bloques."""
    sample = generate_static_scenario_data(
        N_OBS,
        4,
        StructureSpec(
            kind="block",
            params={
                "block_sizes": [2, 2],
                "rho_within": 0.7,
                "rho_between": 0.1,
            },
        ),
        GAUSSIAN_COPULA,
        GAUSSIAN_MARGINAL,
        np.random.default_rng(123),
    )
    corr = _empirical_corr(sample)
    within_left = _mean_block_corr(corr, [0, 1])
    within_right = _mean_block_corr(corr, [2, 3])
    between = _mean_cross_corr(corr, [0, 1], [2, 3])

    assert within_left > between + 0.2
    assert within_right > between + 0.2


def test_near_duplicates_structure_has_very_high_intra_group_correlation() -> None:
    """Verifica correlaciones intra-grupo muy altas en near-duplicates."""
    sample = generate_static_scenario_data(
        N_OBS,
        4,
        StructureSpec(
            kind="near_duplicates",
            params={
                "group_sizes": [2, 2],
                "rho_duplicate": 0.95,
                "rho_background": 0.1,
            },
        ),
        GAUSSIAN_COPULA,
        GAUSSIAN_MARGINAL,
        np.random.default_rng(123),
    )
    corr = _empirical_corr(sample)
    within_left = _mean_block_corr(corr, [0, 1])
    within_right = _mean_block_corr(corr, [2, 3])
    cross = _mean_cross_corr(corr, [0, 1], [2, 3])

    assert within_left > 0.85
    assert within_right > 0.85
    assert cross < within_left
    assert cross < within_right


def test_factor_structure_matches_implied_relative_dependence_pattern() -> None:
    """Verifica el patron relativo esperado de una estructura factorial simple."""
    loadings = np.array([[0.8], [0.8], [0.2], [0.2]])
    sample = generate_static_scenario_data(
        N_OBS,
        4,
        StructureSpec(kind="factor", params={"loadings": loadings}),
        GAUSSIAN_COPULA,
        GAUSSIAN_MARGINAL,
        np.random.default_rng(123),
    )
    corr = _empirical_corr(sample)

    assert corr[0, 1] > 0.5
    assert corr[2, 3] > 0.02
    assert corr[0, 1] > corr[0, 2] + 0.2


@pytest.mark.parametrize(
    "structure",
    [
        StructureSpec(
            kind="block",
            params={
                "block_sizes": [2, 2],
                "rho_within": 0.7,
                "rho_between": 0.1,
            },
        ),
        StructureSpec(
            kind="near_duplicates",
            params={
                "group_sizes": [2, 2],
                "rho_duplicate": 0.95,
                "rho_background": 0.1,
            },
        ),
        StructureSpec(
            kind="factor",
            params={"loadings": np.array([[0.8], [0.8], [0.2], [0.2]])},
        ),
    ],
)
def test_structures_do_not_introduce_nan_or_inf(structure: StructureSpec) -> None:
    """Verifica finitud total en varias estructuras lineales soportadas."""
    sample = generate_static_scenario_data(
        N_OBS,
        4,
        structure,
        GAUSSIAN_COPULA,
        GAUSSIAN_MARGINAL,
        np.random.default_rng(123),
    )

    assert np.all(np.isfinite(sample))


def test_structure_generation_is_reproducible() -> None:
    """Verifica reproducibilidad exacta para una estructura por bloques."""
    structure = StructureSpec(
        kind="block",
        params={
            "block_sizes": [2, 2],
            "rho_within": 0.7,
            "rho_between": 0.1,
        },
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = generate_static_scenario_data(
        N_OBS,
        4,
        structure,
        GAUSSIAN_COPULA,
        GAUSSIAN_MARGINAL,
        rng_a,
    )
    sample_b = generate_static_scenario_data(
        N_OBS,
        4,
        structure,
        GAUSSIAN_COPULA,
        GAUSSIAN_MARGINAL,
        rng_b,
    )

    assert np.array_equal(sample_a, sample_b)
