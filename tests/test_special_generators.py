"""Pruebas para los generadores especiales de ``tc_synthetic``."""

import numpy as np
import pytest

from tc_synthetic.special_generators import generate_nonlinear_redundancy_data
from tc_synthetic.specs import MarginalSpec, StructureSpec



def _make_structure(group_sizes: list[int] | None = None) -> StructureSpec:
    """Construye una estructura minima de redundancia no lineal para tests."""
    return StructureSpec(
        kind="nonlinear_redundancy",
        params={"group_sizes": group_sizes or [2, 2]},
    )



def test_generate_nonlinear_redundancy_data_returns_expected_shape_dtype_and_finite_values() -> None:
    """Verifica el caso valido basico del generador especial."""
    result = generate_nonlinear_redundancy_data(
        8,
        4,
        _make_structure([2, 2]),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )

    assert result.shape == (8, 4)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.all(np.isfinite(result))



def test_generate_nonlinear_redundancy_data_is_reproducible_for_same_seed() -> None:
    """Verifica reproducibilidad con la misma seed y los mismos parametros."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    structure = _make_structure([2, 2])
    marginal = MarginalSpec(kind="gaussian")

    sample_a = generate_nonlinear_redundancy_data(8, 4, structure, marginal, rng_a)
    sample_b = generate_nonlinear_redundancy_data(8, 4, structure, marginal, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_generate_nonlinear_redundancy_data_raises_for_invalid_structure_kind() -> None:
    """Verifica que el generador especial exige ``nonlinear_redundancy``."""
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.3})

    with pytest.raises(
        ValueError,
        match="special generator requires a nonlinear_redundancy structure",
    ):
        generate_nonlinear_redundancy_data(
            8,
            4,
            structure,
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_nonlinear_redundancy_data_raises_when_group_sizes_is_missing() -> None:
    """Verifica que ``group_sizes`` es obligatorio."""
    structure = StructureSpec(kind="nonlinear_redundancy")

    with pytest.raises(
        ValueError,
        match="nonlinear_redundancy structure requires 'group_sizes'",
    ):
        generate_nonlinear_redundancy_data(
            8,
            4,
            structure,
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_nonlinear_redundancy_data_raises_when_group_sizes_sum_does_not_match_n_assets() -> None:
    """Verifica que la suma de ``group_sizes`` debe coincidir con ``n_assets``."""
    with pytest.raises(ValueError, match="sum of group_sizes must match n_assets"):
        generate_nonlinear_redundancy_data(
            8,
            4,
            _make_structure([2, 1]),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_generate_nonlinear_redundancy_data_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        generate_nonlinear_redundancy_data(
            8,
            4,
            _make_structure([2, 2]),
            MarginalSpec(kind="gaussian"),
            rng,
        )



def test_generate_nonlinear_redundancy_data_supports_non_gaussian_marginals() -> None:
    """Verifica que el generador especial tambien funciona con marginal Student-t."""
    result = generate_nonlinear_redundancy_data(
        8,
        4,
        _make_structure([2, 2]),
        MarginalSpec(kind="student_t", params={"df": 7.0}),
        np.random.default_rng(123),
    )

    assert result.shape == (8, 4)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.all(np.isfinite(result))

# === Added in Step 8.2 ===

from tc_synthetic.special_generators import generate_special_structure_data



def test_generate_special_structure_data_returns_expected_shape_dtype_and_finite_values() -> None:
    """Verifica el caso valido basico del dispatcher especial."""
    result = generate_special_structure_data(
        8,
        4,
        _make_structure([2, 2]),
        MarginalSpec(kind="gaussian"),
        np.random.default_rng(123),
    )

    assert result.shape == (8, 4)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.all(np.isfinite(result))



def test_generate_special_structure_data_is_reproducible_for_same_seed() -> None:
    """Verifica reproducibilidad con la misma seed y los mismos parametros."""
    structure = _make_structure([2, 2])
    marginal = MarginalSpec(kind="gaussian")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = generate_special_structure_data(8, 4, structure, marginal, rng_a)
    sample_b = generate_special_structure_data(8, 4, structure, marginal, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_generate_special_structure_data_raises_for_unsupported_structure_kind() -> None:
    """Verifica el error exacto para kinds especiales no soportados."""
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.3})

    with pytest.raises(
        ValueError,
        match="unsupported special structure kind: equicorrelation",
    ):
        generate_special_structure_data(
            8,
            4,
            structure,
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_generate_special_structure_data_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que el dispatcher exige un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        generate_special_structure_data(
            8,
            4,
            _make_structure([2, 2]),
            MarginalSpec(kind="gaussian"),
            rng,
        )



def test_generate_special_structure_data_matches_nonlinear_redundancy_generator_exactly() -> None:
    """Verifica que el dispatcher delega exactamente en el generador concreto."""
    structure = _make_structure([2, 2])
    marginal = MarginalSpec(kind="student_t", params={"df": 7.0})
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = generate_special_structure_data(8, 4, structure, marginal, rng_a)
    expected = generate_nonlinear_redundancy_data(8, 4, structure, marginal, rng_b)

    assert np.array_equal(result, expected)
