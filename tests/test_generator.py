"""Pruebas minimas para el generador de ``tc_synthetic``."""

import numpy as np
import pytest

import tc_synthetic.generator as generator_module
from tc_synthetic.copulas import (
    sample_clayton_copula,
    sample_gaussian_copula,
    sample_grouped_t_copula,
    sample_independence_copula,
    sample_t_copula,
)
from tc_synthetic.generator import (
    apply_marginal_transform,
    resolve_structure_correlation,
    sample_uniform_from_copula,
)
from tc_synthetic.marginals import (
    apply_standard_normal_inverse_cdf,
    apply_standardized_skew_normal_inverse_cdf,
    apply_standardized_student_t_inverse_cdf,
)
from tc_synthetic.specs import CopulaSpec, MarginalSpec, ScenarioSpec, StateProcessSpec, StructureSpec
from tc_synthetic.structures import (
    build_block_correlation_matrix,
    build_equicorrelation_matrix,
    build_factor_correlation_matrix,
    build_near_duplicate_correlation_matrix,
)


# === Added in Step 7.1 ===

def test_resolve_structure_correlation_resolves_equicorrelation() -> None:
    """Verifica que la estructura equicorrelacionada se resuelve correctamente."""
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.3})

    result = resolve_structure_correlation(structure, 4)
    expected = build_equicorrelation_matrix(4, 0.3)

    assert np.array_equal(result, expected)



def test_resolve_structure_correlation_resolves_block() -> None:
    """Verifica que la estructura por bloques se resuelve correctamente."""
    structure = StructureSpec(
        kind="block",
        params={
            "block_sizes": [2, 2],
            "rho_within": 0.6,
            "rho_between": 0.1,
        },
    )

    result = resolve_structure_correlation(structure, 4)
    expected = build_block_correlation_matrix([2, 2], 0.6, 0.1)

    assert np.array_equal(result, expected)



def test_resolve_structure_correlation_resolves_near_duplicates() -> None:
    """Verifica que la estructura near-duplicates se resuelve correctamente."""
    structure = StructureSpec(
        kind="near_duplicates",
        params={
            "group_sizes": [2, 1],
            "rho_duplicate": 0.95,
            "rho_background": 0.1,
        },
    )

    result = resolve_structure_correlation(structure, 3)
    expected = build_near_duplicate_correlation_matrix([2, 1], 0.95, 0.1)

    assert np.array_equal(result, expected)



def test_resolve_structure_correlation_resolves_factor() -> None:
    """Verifica que la estructura factor se resuelve correctamente."""
    loadings = np.array(
        [
            [0.3, 0.1],
            [0.2, 0.4],
            [0.1, 0.2],
        ]
    )
    structure = StructureSpec(kind="factor", params={"loadings": loadings})

    result = resolve_structure_correlation(structure, 3)
    expected = build_factor_correlation_matrix(loadings)

    assert np.array_equal(result, expected)



def test_resolve_structure_correlation_raises_for_invalid_structure() -> None:
    """Verifica que ``structure`` debe ser un ``StructureSpec``."""
    with pytest.raises(TypeError, match="structure must be a StructureSpec"):
        resolve_structure_correlation("x", 3)



def test_resolve_structure_correlation_raises_for_invalid_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.3})

    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        resolve_structure_correlation(structure, 0)



def test_resolve_structure_correlation_raises_for_missing_equicorrelation_rho() -> None:
    """Verifica que equicorrelation requiere ``rho``."""
    structure = StructureSpec(kind="equicorrelation")

    with pytest.raises(ValueError, match="equicorrelation structure requires 'rho'"):
        resolve_structure_correlation(structure, 3)



def test_resolve_structure_correlation_raises_for_missing_block_parameters() -> None:
    """Verifica que block requiere todos sus parametros esenciales."""
    structure = StructureSpec(kind="block", params={"rho_within": 0.6, "rho_between": 0.1})

    with pytest.raises(ValueError, match="block structure requires 'block_sizes'"):
        resolve_structure_correlation(structure, 4)



def test_resolve_structure_correlation_raises_for_missing_near_duplicates_parameters() -> None:
    """Verifica que near-duplicates requiere sus parametros esenciales."""
    structure = StructureSpec(kind="near_duplicates", params={"group_sizes": [2, 1]})

    with pytest.raises(ValueError, match="near_duplicates structure requires 'rho_duplicate'"):
        resolve_structure_correlation(structure, 3)



def test_resolve_structure_correlation_raises_for_missing_factor_loadings() -> None:
    """Verifica que factor requiere ``loadings``."""
    structure = StructureSpec(kind="factor")

    with pytest.raises(ValueError, match="factor structure requires 'loadings'"):
        resolve_structure_correlation(structure, 3)



def test_resolve_structure_correlation_raises_for_nonlinear_redundancy() -> None:
    """Verifica que nonlinear_redundancy no define una matriz lineal."""
    structure = StructureSpec(kind="nonlinear_redundancy")

    with pytest.raises(
        ValueError,
        match="nonlinear_redundancy does not define a linear correlation matrix",
    ):
        resolve_structure_correlation(structure, 3)



def test_resolve_structure_correlation_raises_for_unknown_kind() -> None:
    """Verifica que un kind desconocido produce un error claro."""
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.3})
    structure.kind = "unknown"

    with pytest.raises(ValueError, match="unsupported structure kind: unknown"):
        resolve_structure_correlation(structure, 3)


# === Added in Step 7.2 ===

def test_sample_uniform_from_copula_resolves_independence() -> None:
    """Verifica que la copula de independencia se resuelve correctamente."""
    copula = CopulaSpec(kind="independence")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_uniform_from_copula(copula, 5, 3, rng_a)
    expected = sample_independence_copula(5, 3, rng_b)

    assert np.array_equal(result, expected)



def test_sample_uniform_from_copula_resolves_gaussian() -> None:
    """Verifica que la copula gaussiana se resuelve correctamente."""
    copula = CopulaSpec(kind="gaussian")
    correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_uniform_from_copula(copula, 5, 2, rng_a, correlation=correlation)
    expected = sample_gaussian_copula(5, correlation, rng_b)

    assert np.array_equal(result, expected)



def test_sample_uniform_from_copula_resolves_t() -> None:
    """Verifica que la copula t se resuelve correctamente."""
    copula = CopulaSpec(kind="t", params={"df": 5.0})
    correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_uniform_from_copula(copula, 5, 2, rng_a, correlation=correlation)
    expected = sample_t_copula(5, correlation, 5.0, rng_b)

    assert np.array_equal(result, expected)



def test_sample_uniform_from_copula_resolves_grouped_t() -> None:
    """Verifica que la copula grouped-t se resuelve correctamente."""
    copula = CopulaSpec(
        kind="grouped_t",
        params={
            "group_assignments": np.array([0, 0, 1]),
            "group_dfs": {0: 5.0, 1: 10.0},
        },
    )
    correlation = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.25],
            [0.1, 0.25, 1.0],
        ]
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_uniform_from_copula(copula, 5, 3, rng_a, correlation=correlation)
    expected = sample_grouped_t_copula(
        5,
        correlation,
        np.array([0, 0, 1]),
        {0: 5.0, 1: 10.0},
        rng_b,
    )

    assert np.array_equal(result, expected)



def test_sample_uniform_from_copula_resolves_clayton() -> None:
    """Verifica que la copula Clayton se resuelve correctamente."""
    copula = CopulaSpec(kind="clayton", params={"theta": 2.0})
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_uniform_from_copula(copula, 5, 3, rng_a)
    expected = sample_clayton_copula(5, 3, 2.0, rng_b)

    assert np.array_equal(result, expected)



def test_sample_uniform_from_copula_raises_for_invalid_copula() -> None:
    """Verifica que ``copula`` debe ser un ``CopulaSpec``."""
    with pytest.raises(TypeError, match="copula must be a CopulaSpec"):
        sample_uniform_from_copula("x", 5, 3, np.random.default_rng(123))



def test_sample_uniform_from_copula_raises_for_invalid_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    copula = CopulaSpec(kind="independence")

    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_uniform_from_copula(copula, 0, 3, np.random.default_rng(123))



def test_sample_uniform_from_copula_raises_for_invalid_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    copula = CopulaSpec(kind="independence")

    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        sample_uniform_from_copula(copula, 5, 0, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_uniform_from_copula_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    copula = CopulaSpec(kind="independence")

    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_uniform_from_copula(copula, 5, 3, rng)



def test_sample_uniform_from_copula_raises_when_gaussian_missing_correlation() -> None:
    """Verifica que gaussian requiere correlacion."""
    copula = CopulaSpec(kind="gaussian")

    with pytest.raises(ValueError, match="gaussian copula requires 'correlation'"):
        sample_uniform_from_copula(copula, 5, 2, np.random.default_rng(123))



def test_sample_uniform_from_copula_raises_when_t_missing_correlation() -> None:
    """Verifica que t requiere correlacion."""
    copula = CopulaSpec(kind="t", params={"df": 5.0})

    with pytest.raises(ValueError, match="t copula requires 'correlation'"):
        sample_uniform_from_copula(copula, 5, 2, np.random.default_rng(123))



def test_sample_uniform_from_copula_raises_when_t_missing_df() -> None:
    """Verifica que t requiere ``df``."""
    copula = CopulaSpec(kind="t")
    correlation = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="t copula requires 'df'"):
        sample_uniform_from_copula(copula, 5, 2, np.random.default_rng(123), correlation=correlation)



def test_sample_uniform_from_copula_raises_when_grouped_t_missing_correlation() -> None:
    """Verifica que grouped_t requiere correlacion."""
    copula = CopulaSpec(
        kind="grouped_t",
        params={
            "group_assignments": np.array([0, 0, 1]),
            "group_dfs": {0: 5.0, 1: 10.0},
        },
    )

    with pytest.raises(ValueError, match="grouped_t copula requires 'correlation'"):
        sample_uniform_from_copula(copula, 5, 3, np.random.default_rng(123))



def test_sample_uniform_from_copula_raises_when_grouped_t_missing_group_assignments() -> None:
    """Verifica que grouped_t requiere ``group_assignments``."""
    copula = CopulaSpec(kind="grouped_t", params={"group_dfs": {0: 5.0}})
    correlation = np.eye(1, dtype=float)

    with pytest.raises(ValueError, match="grouped_t copula requires 'group_assignments'"):
        sample_uniform_from_copula(copula, 5, 1, np.random.default_rng(123), correlation=correlation)



def test_sample_uniform_from_copula_raises_when_grouped_t_missing_group_dfs() -> None:
    """Verifica que grouped_t requiere ``group_dfs``."""
    copula = CopulaSpec(kind="grouped_t", params={"group_assignments": np.array([0])})
    correlation = np.eye(1, dtype=float)

    with pytest.raises(ValueError, match="grouped_t copula requires 'group_dfs'"):
        sample_uniform_from_copula(copula, 5, 1, np.random.default_rng(123), correlation=correlation)



def test_sample_uniform_from_copula_raises_when_clayton_missing_theta() -> None:
    """Verifica que Clayton requiere ``theta``."""
    copula = CopulaSpec(kind="clayton")

    with pytest.raises(ValueError, match="clayton copula requires 'theta'"):
        sample_uniform_from_copula(copula, 5, 3, np.random.default_rng(123))



def test_sample_uniform_from_copula_raises_for_shape_mismatch() -> None:
    """Verifica que la muestra resuelta debe coincidir con ``(n_obs, n_assets)``."""
    copula = CopulaSpec(kind="gaussian")
    correlation = np.eye(3, dtype=float)

    with pytest.raises(
        ValueError,
        match=r"resolved copula sample shape must match \(n_obs, n_assets\)",
    ):
        sample_uniform_from_copula(copula, 5, 2, np.random.default_rng(123), correlation=correlation)



def test_sample_uniform_from_copula_raises_for_unknown_kind() -> None:
    """Verifica que un kind desconocido produce un error claro."""
    copula = CopulaSpec(kind="independence")
    copula.kind = "unknown"

    with pytest.raises(ValueError, match="unsupported copula kind: unknown"):
        sample_uniform_from_copula(copula, 5, 3, np.random.default_rng(123))


# === Added in Step 7.3 ===

def test_apply_marginal_transform_resolves_gaussian() -> None:
    """Verifica que la marginal gaussiana se resuelve correctamente."""
    marginal = MarginalSpec(kind="gaussian")
    u = np.array([0.1, 0.5, 0.9])

    result = apply_marginal_transform(marginal, u)
    expected = apply_standard_normal_inverse_cdf(u)

    assert np.array_equal(result, expected)



def test_apply_marginal_transform_resolves_student_t() -> None:
    """Verifica que la marginal Student-t se resuelve correctamente."""
    marginal = MarginalSpec(kind="student_t", params={"df": 5.0})
    u = np.array([[0.1, 0.5], [0.9, 0.75]])

    result = apply_marginal_transform(marginal, u)
    expected = apply_standardized_student_t_inverse_cdf(u, 5.0)

    assert np.array_equal(result, expected)



def test_apply_marginal_transform_resolves_skew_normal() -> None:
    """Verifica que la marginal skew-normal se resuelve correctamente."""
    marginal = MarginalSpec(kind="skew_normal", params={"shape": 4.0})
    u = np.array([[0.1, 0.5], [0.9, 0.75]])

    result = apply_marginal_transform(marginal, u)
    expected = apply_standardized_skew_normal_inverse_cdf(u, 4.0)

    assert np.array_equal(result, expected)



def test_apply_marginal_transform_resolves_heterogeneous() -> None:
    """Verifica que la marginal heterogenea se resuelve columna a columna."""
    marginal = MarginalSpec(
        kind="heterogeneous",
        params={
            "specs": [
                MarginalSpec(kind="gaussian"),
                MarginalSpec(kind="student_t", params={"df": 5.0}),
                MarginalSpec(kind="skew_normal", params={"shape": 4.0}),
            ]
        },
    )
    u = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )

    result = apply_marginal_transform(marginal, u)
    expected = np.column_stack(
        [
            apply_standard_normal_inverse_cdf(u[:, 0]),
            apply_standardized_student_t_inverse_cdf(u[:, 1], 5.0),
            apply_standardized_skew_normal_inverse_cdf(u[:, 2], 4.0),
        ]
    )

    assert np.array_equal(result, expected)



def test_apply_marginal_transform_raises_for_invalid_marginal() -> None:
    """Verifica que ``marginal`` debe ser un ``MarginalSpec``."""
    with pytest.raises(TypeError, match="marginal must be a MarginalSpec"):
        apply_marginal_transform("x", np.array([0.1, 0.5]))



def test_apply_marginal_transform_raises_for_non_array_u() -> None:
    """Verifica que ``u`` debe ser un ``ndarray``."""
    marginal = MarginalSpec(kind="gaussian")

    with pytest.raises(TypeError, match="u must be a numpy.ndarray"):
        apply_marginal_transform(marginal, [0.1, 0.5])



def test_apply_marginal_transform_raises_for_invalid_u_dimension() -> None:
    """Verifica que ``u`` debe ser 1D o 2D."""
    marginal = MarginalSpec(kind="gaussian")

    with pytest.raises(ValueError, match="u must be a 1D or 2D array"):
        apply_marginal_transform(marginal, np.ones((2, 2, 2), dtype=float) * 0.5)



def test_apply_marginal_transform_raises_for_missing_student_t_df() -> None:
    """Verifica que student_t requiere ``df``."""
    marginal = MarginalSpec(kind="student_t")

    with pytest.raises(ValueError, match="student_t marginal requires 'df'"):
        apply_marginal_transform(marginal, np.array([0.1, 0.5]))



def test_apply_marginal_transform_raises_for_missing_skew_normal_shape() -> None:
    """Verifica que skew_normal requiere ``shape``."""
    marginal = MarginalSpec(kind="skew_normal")

    with pytest.raises(ValueError, match="skew_normal marginal requires 'shape'"):
        apply_marginal_transform(marginal, np.array([0.1, 0.5]))



def test_apply_marginal_transform_raises_for_heterogeneous_one_dimensional_input() -> None:
    """Verifica que heterogeneous requiere input bidimensional."""
    marginal = MarginalSpec(kind="heterogeneous", params={"specs": [MarginalSpec(kind="gaussian")]})

    with pytest.raises(ValueError, match="heterogeneous marginal requires a 2D array"):
        apply_marginal_transform(marginal, np.array([0.1, 0.5]))



def test_apply_marginal_transform_raises_for_missing_heterogeneous_specs() -> None:
    """Verifica que heterogeneous requiere ``specs``."""
    marginal = MarginalSpec(kind="heterogeneous")
    u = np.array([[0.1], [0.5]])

    with pytest.raises(ValueError, match="heterogeneous marginal requires 'specs'"):
        apply_marginal_transform(marginal, u)



def test_apply_marginal_transform_raises_for_non_list_heterogeneous_specs() -> None:
    """Verifica que ``specs`` debe ser una lista."""
    marginal = MarginalSpec(kind="heterogeneous", params={"specs": "x"})
    u = np.array([[0.1], [0.5]])

    with pytest.raises(TypeError, match="heterogeneous marginal 'specs' must be a list"):
        apply_marginal_transform(marginal, u)



def test_apply_marginal_transform_raises_for_heterogeneous_specs_length_mismatch() -> None:
    """Verifica que el numero de specs debe coincidir con las columnas de ``u``."""
    marginal = MarginalSpec(
        kind="heterogeneous",
        params={"specs": [MarginalSpec(kind="gaussian")]},
    )
    u = np.array([[0.1, 0.2], [0.5, 0.6]])

    with pytest.raises(
        ValueError,
        match="heterogeneous marginal specs length must match the number of columns in u",
    ):
        apply_marginal_transform(marginal, u)



def test_apply_marginal_transform_raises_for_invalid_heterogeneous_spec_element() -> None:
    """Verifica que cada spec heterogenea debe ser un ``MarginalSpec``."""
    marginal = MarginalSpec(kind="heterogeneous", params={"specs": ["x"]})
    u = np.array([[0.1], [0.5]])

    with pytest.raises(
        TypeError,
        match="heterogeneous marginal specs must contain MarginalSpec instances",
    ):
        apply_marginal_transform(marginal, u)



def test_apply_marginal_transform_raises_for_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que la transformacion resuelta debe preservar el shape de entrada."""
    marginal = MarginalSpec(kind="gaussian")
    u = np.array([0.1, 0.5, 0.9])

    monkeypatch.setattr(
        generator_module,
        "apply_standard_normal_inverse_cdf",
        lambda values: np.array([[1.0, 2.0]], dtype=float),
    )

    with pytest.raises(
        ValueError,
        match="resolved marginal transform shape must match input shape",
    ):
        generator_module.apply_marginal_transform(marginal, u)



def test_apply_marginal_transform_raises_for_unknown_kind() -> None:
    """Verifica que un kind desconocido produce un error claro."""
    marginal = MarginalSpec(kind="gaussian")
    marginal.kind = "unknown"

    with pytest.raises(ValueError, match="unsupported marginal kind: unknown"):
        apply_marginal_transform(marginal, np.array([0.1, 0.5]))

# === Added in Step 7.4 ===

def test_generate_static_scenario_data_matches_manual_construction_for_equicorrelation_gaussian_gaussian() -> None:
    """Verifica la coherencia exacta con la construccion manual del pipeline estatico."""
    n_obs = 6
    n_assets = 3
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.3})
    copula = CopulaSpec(kind="gaussian")
    marginal = MarginalSpec(kind="gaussian")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = generator_module.generate_static_scenario_data(
        n_obs,
        n_assets,
        structure,
        copula,
        marginal,
        rng_a,
    )
    correlation = resolve_structure_correlation(structure, n_assets)
    u = sample_uniform_from_copula(copula, n_obs, n_assets, rng_b, correlation=correlation)
    expected = apply_marginal_transform(marginal, u)

    assert np.array_equal(result, expected)
    assert result.shape == (n_obs, n_assets)
    assert np.issubdtype(result.dtype, np.floating)



def test_generate_static_scenario_data_handles_independence_with_gaussian_marginal() -> None:
    """Verifica el caso independence con marginal gaussiana."""
    n_obs = 5
    n_assets = 3
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.2})
    copula = CopulaSpec(kind="independence")
    marginal = MarginalSpec(kind="gaussian")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = generator_module.generate_static_scenario_data(
        n_obs,
        n_assets,
        structure,
        copula,
        marginal,
        rng_a,
    )
    expected = apply_marginal_transform(
        marginal,
        sample_uniform_from_copula(copula, n_obs, n_assets, rng_b, correlation=None),
    )

    assert np.array_equal(result, expected)
    assert result.shape == (n_obs, n_assets)
    assert np.issubdtype(result.dtype, np.floating)



def test_generate_static_scenario_data_handles_clayton_with_gaussian_marginal() -> None:
    """Verifica el caso Clayton con marginal gaussiana."""
    n_obs = 5
    n_assets = 3
    structure = StructureSpec(kind="equicorrelation", params={"rho": 0.2})
    copula = CopulaSpec(kind="clayton", params={"theta": 2.0})
    marginal = MarginalSpec(kind="gaussian")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = generator_module.generate_static_scenario_data(
        n_obs,
        n_assets,
        structure,
        copula,
        marginal,
        rng_a,
    )
    expected = apply_marginal_transform(
        marginal,
        sample_uniform_from_copula(copula, n_obs, n_assets, rng_b, correlation=None),
    )

    assert np.array_equal(result, expected)
    assert result.shape == (n_obs, n_assets)
    assert np.issubdtype(result.dtype, np.floating)



def test_generate_static_scenario_data_handles_factor_t_student_t_case() -> None:
    """Verifica un caso completo con estructura factor, t copula y marginal Student-t."""
    n_obs = 6
    n_assets = 3
    loadings = np.array(
        [
            [0.3, 0.1],
            [0.2, 0.4],
            [0.1, 0.2],
        ]
    )
    structure = StructureSpec(kind="factor", params={"loadings": loadings})
    copula = CopulaSpec(kind="t", params={"df": 5.0})
    marginal = MarginalSpec(kind="student_t", params={"df": 7.0})
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = generator_module.generate_static_scenario_data(
        n_obs,
        n_assets,
        structure,
        copula,
        marginal,
        rng_a,
    )
    correlation = resolve_structure_correlation(structure, n_assets)
    u = sample_uniform_from_copula(copula, n_obs, n_assets, rng_b, correlation=correlation)
    expected = apply_marginal_transform(marginal, u)

    assert np.array_equal(result, expected)
    assert result.shape == (n_obs, n_assets)
    assert np.issubdtype(result.dtype, np.floating)



def test_generate_static_scenario_data_raises_for_invalid_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        generator_module.generate_static_scenario_data(
            0,
            3,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_raises_for_invalid_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        generator_module.generate_static_scenario_data(
            5,
            0,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_raises_for_invalid_structure() -> None:
    """Verifica que ``structure`` debe ser un ``StructureSpec``."""
    with pytest.raises(TypeError, match="structure must be a StructureSpec"):
        generator_module.generate_static_scenario_data(
            5,
            3,
            "x",
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_raises_for_invalid_copula() -> None:
    """Verifica que ``copula`` debe ser un ``CopulaSpec``."""
    with pytest.raises(TypeError, match="copula must be a CopulaSpec"):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            "x",
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_raises_for_invalid_marginal() -> None:
    """Verifica que ``marginal`` debe ser un ``MarginalSpec``."""
    with pytest.raises(TypeError, match="marginal must be a MarginalSpec"):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            CopulaSpec(kind="independence"),
            "x",
            np.random.default_rng(123),
        )



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_generate_static_scenario_data_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            rng,
        )



def test_generate_static_scenario_data_raises_for_nonlinear_redundancy_structure() -> None:
    """Verifica que la version estatica no soporta nonlinear_redundancy."""
    with pytest.raises(
        ValueError,
        match="static generator does not support nonlinear_redundancy structures",
    ):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(kind="nonlinear_redundancy"),
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_propagates_structure_inconsistency() -> None:
    """Verifica que una inconsistencia entre ``n_assets`` y la estructura se propaga."""
    with pytest.raises(ValueError, match="resolved correlation shape must match n_assets"):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(
                kind="block",
                params={
                    "block_sizes": [2, 2],
                    "rho_within": 0.6,
                    "rho_between": 0.1,
                },
            ),
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_propagates_copula_inconsistency() -> None:
    """Verifica que una inconsistencia entre ``n_assets`` y la copula termina fallando."""
    with pytest.raises(ValueError, match="group_assignments length must match n_assets"):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            CopulaSpec(
                kind="grouped_t",
                params={
                    "group_assignments": np.array([0, 1]),
                    "group_dfs": {0: 5.0, 1: 10.0},
                },
            ),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )



def test_generate_static_scenario_data_raises_for_final_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que un shape final incorrecto produce un error claro."""
    monkeypatch.setattr(
        generator_module,
        "apply_marginal_transform",
        lambda marginal, u: np.ones((1, 1), dtype=float),
    )

    with pytest.raises(
        ValueError,
        match=r"generated data shape must match \(n_obs, n_assets\)",
    ):
        generator_module.generate_static_scenario_data(
            5,
            3,
            StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
            CopulaSpec(kind="independence"),
            MarginalSpec(kind="gaussian"),
            np.random.default_rng(123),
        )

# === Added in Step 9.1 ===

def test_sample_state_path_returns_binary_integer_path() -> None:
    """Verifica que la trayectoria devuelta tiene shape, dtype y estados validos."""
    n_obs = 8
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "initial_state": 0,
        },
        enabled=True,
    )
    rng = np.random.default_rng(123)

    result = generator_module._sample_state_path(state_spec, n_obs, rng)

    assert result.shape == (n_obs,)
    assert np.issubdtype(result.dtype, np.integer)
    assert set(np.unique(result)).issubset({0, 1})



def test_sample_state_path_raises_when_state_process_is_disabled() -> None:
    """Verifica el error si el proceso de estados no esta habilitado."""
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "initial_state": 0,
        },
        enabled=False,
    )

    with pytest.raises(ValueError, match="state process must be enabled"):
        generator_module._sample_state_path(state_spec, 8, np.random.default_rng(123))



def test_sample_state_path_raises_when_kind_is_not_markov() -> None:
    """Verifica el error si el kind del proceso de estados no es ``markov``."""
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "initial_state": 0,
        },
        enabled=True,
    )
    state_spec.kind = "unknown"

    with pytest.raises(ValueError, match="only 'markov' state processes are supported"):
        generator_module._sample_state_path(state_spec, 8, np.random.default_rng(123))



def test_sample_state_path_raises_when_transition_matrix_is_missing() -> None:
    """Verifica el error si falta ``transition_matrix``."""
    state_spec = StateProcessSpec(
        kind="markov",
        params={"initial_state": 0},
        enabled=True,
    )

    with pytest.raises(
        ValueError,
        match="state_spec.params must include 'transition_matrix'",
    ):
        generator_module._sample_state_path(state_spec, 8, np.random.default_rng(123))



def test_sample_state_path_raises_when_initial_state_is_missing() -> None:
    """Verifica el error si falta ``initial_state``."""
    state_spec = StateProcessSpec(
        kind="markov",
        params={"transition_matrix": np.array([[0.9, 0.1], [0.2, 0.8]])},
        enabled=True,
    )

    with pytest.raises(
        ValueError,
        match="state_spec.params must include 'initial_state'",
    ):
        generator_module._sample_state_path(state_spec, 8, np.random.default_rng(123))

# === Added in Step 9.2 ===

def _make_two_state_scenario(
    *,
    name: str,
    n_obs: int = 6,
    n_assets: int = 3,
    structure: StructureSpec | None = None,
    copula: CopulaSpec | None = None,
    marginal: MarginalSpec | None = None,
) -> ScenarioSpec:
    """Construye un ``ScenarioSpec`` pequeno para pruebas de dos estados."""
    return ScenarioSpec(
        name=name,
        n_assets=n_assets,
        n_obs=n_obs,
        structure=structure or StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
        copula=copula or CopulaSpec(kind="gaussian"),
        marginal=marginal or MarginalSpec(kind="gaussian"),
    )



def test_generate_two_state_observationwise_data_returns_expected_shapes_and_types() -> None:
    """Verifica el caso valido basico para generacion observacion a observacion."""
    n_obs = 6
    calm_spec = _make_two_state_scenario(name="calm", n_obs=n_obs)
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        n_obs=n_obs,
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )
    rng = np.random.default_rng(123)

    x, states = generator_module._generate_two_state_observationwise_data(
        calm_spec,
        crisis_spec,
        state_spec,
        rng,
    )

    assert x.shape == (n_obs, calm_spec.n_assets)
    assert states.shape == (n_obs,)
    assert set(np.unique(states)).issubset({0, 1})
    assert np.issubdtype(x.dtype, np.floating)



def test_generate_two_state_observationwise_data_is_reproducible() -> None:
    """Verifica reproducibilidad con la misma seed y los mismos specs."""
    n_obs = 6
    calm_spec = _make_two_state_scenario(name="calm", n_obs=n_obs)
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        n_obs=n_obs,
        marginal=MarginalSpec(kind="student_t", params={"df": 7.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    x_a, states_a = generator_module._generate_two_state_observationwise_data(
        calm_spec,
        crisis_spec,
        state_spec,
        rng_a,
    )
    x_b, states_b = generator_module._generate_two_state_observationwise_data(
        calm_spec,
        crisis_spec,
        state_spec,
        rng_b,
    )

    assert np.array_equal(x_a, x_b)
    assert np.array_equal(states_a, states_b)



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_generate_two_state_observationwise_data_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    calm_spec = _make_two_state_scenario(name="calm")
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        generator_module._generate_two_state_observationwise_data(
            calm_spec,
            crisis_spec,
            state_spec,
            rng,
        )



def test_generate_two_state_observationwise_data_raises_when_n_obs_differs() -> None:
    """Verifica el error exacto si calm y crisis difieren en ``n_obs``."""
    calm_spec = _make_two_state_scenario(name="calm", n_obs=5)
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        n_obs=6,
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    with pytest.raises(ValueError, match="calm and crisis scenarios must have the same n_obs"):
        generator_module._generate_two_state_observationwise_data(
            calm_spec,
            crisis_spec,
            state_spec,
            np.random.default_rng(123),
        )



def test_generate_two_state_observationwise_data_propagates_nonlinear_redundancy_error() -> None:
    """Verifica que se propaga el error del generador estatico para nonlinear_redundancy."""
    n_obs = 6
    calm_spec = _make_two_state_scenario(
        name="calm",
        n_obs=n_obs,
        structure=StructureSpec(kind="nonlinear_redundancy"),
    )
    crisis_spec = _make_two_state_scenario(name="crisis", n_obs=n_obs)
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.eye(2, dtype=float),
            "initial_state": 0,
        },
        enabled=True,
    )

    with pytest.raises(
        ValueError,
        match="static generator does not support nonlinear_redundancy structures",
    ):
        generator_module._generate_two_state_observationwise_data(
            calm_spec,
            crisis_spec,
            state_spec,
            np.random.default_rng(123),
        )

# === Added in Step 9.3 ===

def test_generate_two_state_scenario_data_returns_expected_shapes_and_states() -> None:
    """Verifica el caso valido basico para la API publica de dos estados."""
    n_obs = 6
    calm_spec = _make_two_state_scenario(name="calm", n_obs=n_obs)
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        n_obs=n_obs,
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    x, states = generator_module.generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        np.random.default_rng(123),
    )

    assert x.shape == (n_obs, calm_spec.n_assets)
    assert states.shape == (n_obs,)
    assert set(np.unique(states)).issubset({0, 1})



def test_generate_two_state_scenario_data_is_reproducible() -> None:
    """Verifica reproducibilidad con la misma seed y los mismos specs."""
    n_obs = 6
    calm_spec = _make_two_state_scenario(name="calm", n_obs=n_obs)
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        n_obs=n_obs,
        marginal=MarginalSpec(kind="student_t", params={"df": 7.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    x_a, states_a = generator_module.generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        rng_a,
    )
    x_b, states_b = generator_module.generate_two_state_scenario_data(
        calm_spec,
        crisis_spec,
        state_spec,
        rng_b,
    )

    assert np.array_equal(x_a, x_b)
    assert np.array_equal(states_a, states_b)



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_generate_two_state_scenario_data_propagates_invalid_rng_error(rng: object) -> None:
    """Verifica que la API publica propaga el error de ``rng`` invalido."""
    calm_spec = _make_two_state_scenario(name="calm")
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        generator_module.generate_two_state_scenario_data(
            calm_spec,
            crisis_spec,
            state_spec,
            rng,
        )



def test_generate_two_state_scenario_data_propagates_n_obs_mismatch_error() -> None:
    """Verifica que la API publica propaga el error si ``n_obs`` difiere."""
    calm_spec = _make_two_state_scenario(name="calm", n_obs=5)
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        n_obs=6,
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=True,
    )

    with pytest.raises(ValueError, match="calm and crisis scenarios must have the same n_obs"):
        generator_module.generate_two_state_scenario_data(
            calm_spec,
            crisis_spec,
            state_spec,
            np.random.default_rng(123),
        )



def test_generate_two_state_scenario_data_propagates_invalid_declarative_scenario_error() -> None:
    """Verifica que la API publica propaga el error declarativo del proceso de estados."""
    calm_spec = _make_two_state_scenario(name="calm")
    crisis_spec = _make_two_state_scenario(
        name="crisis",
        copula=CopulaSpec(kind="t", params={"df": 5.0}),
    )
    state_spec = StateProcessSpec(
        kind="markov",
        params={
            "transition_matrix": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "initial_state": 0,
        },
        enabled=False,
    )

    with pytest.raises(
        ValueError,
        match="state process must be enabled for two-state scenarios",
    ):
        generator_module.generate_two_state_scenario_data(
            calm_spec,
            crisis_spec,
            state_spec,
            np.random.default_rng(123),
        )
