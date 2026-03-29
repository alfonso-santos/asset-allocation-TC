"""Pruebas minimas para las marginales de ``tc_synthetic``."""

import numpy as np
import pytest
from scipy.stats import norm, skewnorm, t

from tc_synthetic.marginals import (
    apply_standard_normal_inverse_cdf,
    apply_standardized_skew_normal_inverse_cdf,
    apply_standardized_student_t_inverse_cdf,
    sample_heterogeneous_marginals,
    sample_standard_normal_marginal,
    sample_standardized_skew_normal_marginal,
    sample_standardized_student_t_marginal,
    standardize_1d_sample,
)
from tc_synthetic.specs import MarginalSpec



def test_standardize_1d_sample_returns_standardized_array() -> None:
    """Verifica que la muestra se estandariza correctamente."""
    sample = np.array([1.0, 2.0, 3.0])

    result = standardize_1d_sample(sample)

    assert result.shape == sample.shape
    assert np.issubdtype(result.dtype, np.floating)
    assert np.isclose(np.mean(result), 0.0)
    assert np.isclose(np.std(result, ddof=0), 1.0)



def test_standardize_1d_sample_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="sample must be a numpy.ndarray"):
        standardize_1d_sample([1.0, 2.0, 3.0])



def test_standardize_1d_sample_raises_for_two_dimensional_input() -> None:
    """Verifica que la muestra debe ser un array unidimensional."""
    with pytest.raises(ValueError, match="sample must be a 1D array"):
        standardize_1d_sample(np.array([[1.0, 2.0], [3.0, 4.0]]))



def test_standardize_1d_sample_raises_for_empty_input() -> None:
    """Verifica que la muestra no puede estar vacia."""
    with pytest.raises(ValueError, match="sample must not be empty"):
        standardize_1d_sample(np.array([], dtype=float))



def test_standardize_1d_sample_raises_for_boolean_values() -> None:
    """Verifica que no se aceptan booleanos."""
    with pytest.raises(TypeError, match="sample must not contain booleans"):
        standardize_1d_sample(np.array([True, False]))



def test_standardize_1d_sample_raises_for_non_numeric_values() -> None:
    """Verifica que la muestra debe contener valores numericos."""
    with pytest.raises(TypeError, match="sample must contain numeric values"):
        standardize_1d_sample(np.array(["a", "b"]))



def test_standardize_1d_sample_raises_for_nan_values() -> None:
    """Verifica que la muestra no acepta ``NaN``."""
    with pytest.raises(ValueError, match="sample must contain finite values"):
        standardize_1d_sample(np.array([1.0, np.nan]))



def test_standardize_1d_sample_raises_for_infinite_values() -> None:
    """Verifica que la muestra no acepta infinitos."""
    with pytest.raises(ValueError, match="sample must contain finite values"):
        standardize_1d_sample(np.array([1.0, np.inf]))



def test_standardize_1d_sample_raises_for_zero_standard_deviation() -> None:
    """Verifica que la desviacion tipica no puede ser cero."""
    with pytest.raises(ValueError, match="sample must have non-zero standard deviation"):
        standardize_1d_sample(np.array([2.0, 2.0, 2.0]))



def test_sample_standard_normal_marginal_returns_expected_shape() -> None:
    """Verifica que la muestra gaussiana tiene shape unidimensional correcto."""
    rng = np.random.default_rng(123)

    result = sample_standard_normal_marginal(5, rng)

    assert result.shape == (5,)



def test_sample_standard_normal_marginal_returns_float_dtype() -> None:
    """Verifica que la muestra gaussiana se devuelve como array flotante."""
    rng = np.random.default_rng(123)

    result = sample_standard_normal_marginal(5, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_standard_normal_marginal_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma secuencia."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_standard_normal_marginal(5, rng_a)
    sample_b = sample_standard_normal_marginal(5, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_standard_normal_marginal_accepts_valid_generator() -> None:
    """Verifica que la funcion usa correctamente un ``Generator`` valido."""
    rng = np.random.default_rng(321)
    expected = np.random.default_rng(321).standard_normal(size=4)

    result = sample_standard_normal_marginal(4, rng)

    assert np.array_equal(result, expected)



def test_sample_standard_normal_marginal_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_standard_normal_marginal(0, np.random.default_rng(123))



def test_sample_standard_normal_marginal_raises_for_negative_n_obs() -> None:
    """Verifica que ``n_obs=-1`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_standard_normal_marginal(-1, np.random.default_rng(123))



def test_sample_standard_normal_marginal_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_standard_normal_marginal(True, np.random.default_rng(123))



def test_sample_standard_normal_marginal_raises_for_non_integer_n_obs() -> None:
    """Verifica que ``n_obs`` debe ser entero."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_standard_normal_marginal(1.5, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_standard_normal_marginal_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_standard_normal_marginal(5, rng)



def test_sample_standardized_student_t_marginal_returns_expected_shape() -> None:
    """Verifica que la muestra Student-t tiene shape unidimensional correcto."""
    rng = np.random.default_rng(123)

    result = sample_standardized_student_t_marginal(5, 5.0, rng)

    assert result.shape == (5,)



def test_sample_standardized_student_t_marginal_returns_float_dtype() -> None:
    """Verifica que la muestra Student-t se devuelve como array flotante."""
    rng = np.random.default_rng(123)

    result = sample_standardized_student_t_marginal(5, 5.0, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_standardized_student_t_marginal_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma secuencia."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_standardized_student_t_marginal(5, 5.0, rng_a)
    sample_b = sample_standardized_student_t_marginal(5, 5.0, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_standardized_student_t_marginal_accepts_valid_parameters() -> None:
    """Verifica que la funcion opera correctamente con un caso valido."""
    rng = np.random.default_rng(321)

    result = sample_standardized_student_t_marginal(4, 5.0, rng)

    assert result.shape == (4,)



def test_sample_standardized_student_t_marginal_matches_theoretical_scaling() -> None:
    """Verifica la coherencia exacta con la formula de escalado teorico."""
    n_obs = 6
    df = 5.0
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_standardized_student_t_marginal(n_obs, df, rng_a)
    raw = rng_b.standard_t(df, size=n_obs)
    expected = raw * np.sqrt((df - 2.0) / df)

    assert np.array_equal(result, expected)



def test_sample_standardized_student_t_marginal_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_standardized_student_t_marginal(0, 5.0, np.random.default_rng(123))



def test_sample_standardized_student_t_marginal_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_standardized_student_t_marginal(True, 5.0, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_standardized_student_t_marginal_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_standardized_student_t_marginal(5, 5.0, rng)



def test_sample_standardized_student_t_marginal_raises_for_non_numeric_df() -> None:
    """Verifica que ``df`` debe ser numerico."""
    with pytest.raises(TypeError, match="df must be a number"):
        sample_standardized_student_t_marginal(5, "x", np.random.default_rng(123))



def test_sample_standardized_student_t_marginal_raises_for_boolean_df() -> None:
    """Verifica que ``df`` no acepta booleanos."""
    with pytest.raises(TypeError, match="df must be a number"):
        sample_standardized_student_t_marginal(5, True, np.random.default_rng(123))



def test_sample_standardized_student_t_marginal_raises_for_df_equal_to_two() -> None:
    """Verifica que ``df=2`` falla."""
    with pytest.raises(ValueError, match="df must be greater than 2"):
        sample_standardized_student_t_marginal(5, 2.0, np.random.default_rng(123))



def test_sample_standardized_student_t_marginal_raises_for_df_below_two() -> None:
    """Verifica que ``df<2`` falla."""
    with pytest.raises(ValueError, match="df must be greater than 2"):
        sample_standardized_student_t_marginal(5, 1.5, np.random.default_rng(123))



def test_sample_standardized_skew_normal_marginal_returns_expected_shape() -> None:
    """Verifica que la muestra skew-normal tiene shape unidimensional correcto."""
    rng = np.random.default_rng(123)

    result = sample_standardized_skew_normal_marginal(5, 4.0, rng)

    assert result.shape == (5,)



def test_sample_standardized_skew_normal_marginal_returns_float_dtype() -> None:
    """Verifica que la muestra skew-normal se devuelve como array flotante."""
    rng = np.random.default_rng(123)

    result = sample_standardized_skew_normal_marginal(5, 4.0, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_standardized_skew_normal_marginal_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma secuencia."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_standardized_skew_normal_marginal(5, 4.0, rng_a)
    sample_b = sample_standardized_skew_normal_marginal(5, 4.0, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_standardized_skew_normal_marginal_accepts_valid_parameters() -> None:
    """Verifica que la funcion opera correctamente con un caso valido."""
    rng = np.random.default_rng(321)

    result = sample_standardized_skew_normal_marginal(4, 4.0, rng)

    assert result.shape == (4,)



def test_sample_standardized_skew_normal_marginal_matches_theoretical_scaling() -> None:
    """Verifica la coherencia exacta con la formula de escalado teorico."""
    n_obs = 6
    shape = 4.0
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_standardized_skew_normal_marginal(n_obs, shape, rng_a)
    raw = skewnorm.rvs(a=shape, loc=0, scale=1, size=n_obs, random_state=rng_b)
    delta = shape / np.sqrt(1.0 + shape**2)
    mu = delta * np.sqrt(2.0 / np.pi)
    sigma = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    expected = (raw - mu) / sigma

    assert np.array_equal(result, expected)



def test_sample_standardized_skew_normal_marginal_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_standardized_skew_normal_marginal(0, 4.0, np.random.default_rng(123))



def test_sample_standardized_skew_normal_marginal_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_standardized_skew_normal_marginal(True, 4.0, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_standardized_skew_normal_marginal_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_standardized_skew_normal_marginal(5, 4.0, rng)



def test_sample_standardized_skew_normal_marginal_raises_for_non_numeric_shape() -> None:
    """Verifica que ``shape`` debe ser numerico."""
    with pytest.raises(TypeError, match="shape must be a number"):
        sample_standardized_skew_normal_marginal(5, "x", np.random.default_rng(123))



def test_sample_standardized_skew_normal_marginal_raises_for_boolean_shape() -> None:
    """Verifica que ``shape`` no acepta booleanos."""
    with pytest.raises(TypeError, match="shape must be a number"):
        sample_standardized_skew_normal_marginal(5, True, np.random.default_rng(123))



def test_sample_standardized_skew_normal_marginal_raises_for_nan_shape() -> None:
    """Verifica que ``shape=np.nan`` falla."""
    with pytest.raises(ValueError, match="shape must be finite"):
        sample_standardized_skew_normal_marginal(5, np.nan, np.random.default_rng(123))



def test_sample_standardized_skew_normal_marginal_raises_for_infinite_shape() -> None:
    """Verifica que ``shape=np.inf`` falla."""
    with pytest.raises(ValueError, match="shape must be finite"):
        sample_standardized_skew_normal_marginal(5, np.inf, np.random.default_rng(123))



def test_sample_heterogeneous_marginals_returns_expected_shape_and_dtype() -> None:
    """Verifica que el wrapper devuelve una matriz float con una columna por spec."""
    specs = [
        MarginalSpec(kind="gaussian"),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        MarginalSpec(kind="skew_normal", params={"shape": 4.0}),
    ]
    rng = np.random.default_rng(123)

    result = sample_heterogeneous_marginals(5, specs, rng)

    assert result.shape == (5, 3)
    assert np.issubdtype(result.dtype, np.floating)



def test_sample_heterogeneous_marginals_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma matriz."""
    specs = [
        MarginalSpec(kind="gaussian"),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        MarginalSpec(kind="skew_normal", params={"shape": 4.0}),
    ]
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_heterogeneous_marginals(5, specs, rng_a)
    sample_b = sample_heterogeneous_marginals(5, specs, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_heterogeneous_marginals_matches_columnwise_generation() -> None:
    """Verifica la coherencia exacta con la generacion secuencial columna a columna."""
    n_obs = 6
    specs = [
        MarginalSpec(kind="gaussian"),
        MarginalSpec(kind="student_t", params={"df": 5.0}),
        MarginalSpec(kind="skew_normal", params={"shape": 4.0}),
    ]
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_heterogeneous_marginals(n_obs, specs, rng_a)
    expected = np.column_stack(
        [
            sample_standard_normal_marginal(n_obs, rng_b),
            sample_standardized_student_t_marginal(n_obs, 5.0, rng_b),
            sample_standardized_skew_normal_marginal(n_obs, 4.0, rng_b),
        ]
    )

    assert np.array_equal(result, expected)



def test_sample_heterogeneous_marginals_raises_for_non_list_specs() -> None:
    """Verifica que ``specs`` debe ser una lista."""
    specs = (MarginalSpec(kind="gaussian"),)

    with pytest.raises(TypeError, match="specs must be a list"):
        sample_heterogeneous_marginals(5, specs, np.random.default_rng(123))



def test_sample_heterogeneous_marginals_raises_for_empty_specs() -> None:
    """Verifica que ``specs`` no puede estar vacia."""
    with pytest.raises(ValueError, match="specs must not be empty"):
        sample_heterogeneous_marginals(5, [], np.random.default_rng(123))



def test_sample_heterogeneous_marginals_raises_for_invalid_spec_element() -> None:
    """Verifica que cada elemento debe ser un ``MarginalSpec``."""
    specs = [MarginalSpec(kind="gaussian"), "x"]

    with pytest.raises(TypeError, match="specs must contain MarginalSpec instances"):
        sample_heterogeneous_marginals(5, specs, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123])
def test_sample_heterogeneous_marginals_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    specs = [MarginalSpec(kind="gaussian")]

    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_heterogeneous_marginals(5, specs, rng)



def test_sample_heterogeneous_marginals_raises_for_missing_student_t_df() -> None:
    """Verifica que la spec Student-t requiere ``df``."""
    specs = [MarginalSpec(kind="student_t")]

    with pytest.raises(ValueError, match="student_t spec requires 'df'"):
        sample_heterogeneous_marginals(5, specs, np.random.default_rng(123))



def test_sample_heterogeneous_marginals_raises_for_missing_skew_normal_shape() -> None:
    """Verifica que la spec skew-normal requiere ``shape``."""
    specs = [MarginalSpec(kind="skew_normal")]

    with pytest.raises(ValueError, match="skew_normal spec requires 'shape'"):
        sample_heterogeneous_marginals(5, specs, np.random.default_rng(123))



def test_sample_heterogeneous_marginals_raises_for_nested_heterogeneous_kind() -> None:
    """Verifica que no se acepta una spec de tipo ``heterogeneous`` dentro del wrapper."""
    specs = [MarginalSpec(kind="heterogeneous")]

    with pytest.raises(ValueError, match="heterogeneous specs are not supported here"):
        sample_heterogeneous_marginals(5, specs, np.random.default_rng(123))



def test_sample_heterogeneous_marginals_raises_for_unknown_kind() -> None:
    """Verifica que un kind desconocido produce error claro."""
    spec = MarginalSpec(kind="gaussian")
    spec.kind = "unknown"

    with pytest.raises(ValueError, match="unsupported marginal kind: unknown"):
        sample_heterogeneous_marginals(5, [spec], np.random.default_rng(123))


# === Added in Step 4.6 ===

def test_apply_standard_normal_inverse_cdf_preserves_one_dimensional_shape() -> None:
    """Verifica que la transformacion preserva el shape 1D."""
    u = np.array([0.1, 0.5, 0.9])

    result = apply_standard_normal_inverse_cdf(u)

    assert result.shape == u.shape



def test_apply_standard_normal_inverse_cdf_preserves_two_dimensional_shape() -> None:
    """Verifica que la transformacion preserva el shape 2D."""
    u = np.array([[0.1, 0.5], [0.9, 0.75]])

    result = apply_standard_normal_inverse_cdf(u)

    assert result.shape == u.shape



def test_apply_standard_normal_inverse_cdf_returns_float_dtype() -> None:
    """Verifica que la salida se devuelve como array flotante."""
    u = np.array([0.1, 0.5, 0.9])

    result = apply_standard_normal_inverse_cdf(u)

    assert np.issubdtype(result.dtype, np.floating)



def test_apply_standard_normal_inverse_cdf_matches_norm_ppf_exactly() -> None:
    """Verifica que el resultado coincide exactamente con ``norm.ppf``."""
    u = np.array([[0.1, 0.5], [0.9, 0.75]])

    result = apply_standard_normal_inverse_cdf(u)

    assert np.array_equal(result, norm.ppf(u))



def test_apply_standard_normal_inverse_cdf_accepts_valid_values_in_open_unit_interval() -> None:
    """Verifica que la funcion acepta valores validos en ``(0, 1)``."""
    u = np.array([1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6])

    result = apply_standard_normal_inverse_cdf(u)

    assert np.all(np.isfinite(result))



def test_apply_standard_normal_inverse_cdf_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="u must be a numpy.ndarray"):
        apply_standard_normal_inverse_cdf([0.1, 0.5, 0.9])



def test_apply_standard_normal_inverse_cdf_raises_for_three_dimensional_input() -> None:
    """Verifica que el array debe ser 1D o 2D."""
    with pytest.raises(ValueError, match="u must be a 1D or 2D array"):
        apply_standard_normal_inverse_cdf(np.ones((2, 2, 2), dtype=float) * 0.5)



def test_apply_standard_normal_inverse_cdf_raises_for_boolean_values() -> None:
    """Verifica que no se aceptan booleanos."""
    with pytest.raises(TypeError, match="u must not contain booleans"):
        apply_standard_normal_inverse_cdf(np.array([True, False]))



def test_apply_standard_normal_inverse_cdf_raises_for_non_numeric_values() -> None:
    """Verifica que el array debe contener valores numericos."""
    with pytest.raises(TypeError, match="u must contain numeric values"):
        apply_standard_normal_inverse_cdf(np.array(["a", "b"]))



def test_apply_standard_normal_inverse_cdf_raises_for_nan_values() -> None:
    """Verifica que el array no acepta ``NaN``."""
    with pytest.raises(ValueError, match="u must contain finite values"):
        apply_standard_normal_inverse_cdf(np.array([0.1, np.nan]))



def test_apply_standard_normal_inverse_cdf_raises_for_infinite_values() -> None:
    """Verifica que el array no acepta infinitos."""
    with pytest.raises(ValueError, match="u must contain finite values"):
        apply_standard_normal_inverse_cdf(np.array([0.1, np.inf]))



def test_apply_standard_normal_inverse_cdf_raises_for_zero_values() -> None:
    """Verifica que no se aceptan valores iguales a cero."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standard_normal_inverse_cdf(np.array([0.0, 0.5]))



def test_apply_standard_normal_inverse_cdf_raises_for_one_values() -> None:
    """Verifica que no se aceptan valores iguales a uno."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standard_normal_inverse_cdf(np.array([0.5, 1.0]))



def test_apply_standard_normal_inverse_cdf_raises_for_negative_values() -> None:
    """Verifica que no se aceptan valores negativos."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standard_normal_inverse_cdf(np.array([-0.1, 0.5]))



def test_apply_standard_normal_inverse_cdf_raises_for_values_above_one() -> None:
    """Verifica que no se aceptan valores mayores que uno."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standard_normal_inverse_cdf(np.array([0.5, 1.1]))




# === Added in Step 4.7 ===

def test_apply_standardized_student_t_inverse_cdf_preserves_one_dimensional_shape() -> None:
    """Verifica que la transformacion preserva el shape 1D."""
    u = np.array([0.1, 0.5, 0.9])

    result = apply_standardized_student_t_inverse_cdf(u, 5.0)

    assert result.shape == u.shape



def test_apply_standardized_student_t_inverse_cdf_preserves_two_dimensional_shape() -> None:
    """Verifica que la transformacion preserva el shape 2D."""
    u = np.array([[0.1, 0.5], [0.9, 0.75]])

    result = apply_standardized_student_t_inverse_cdf(u, 5.0)

    assert result.shape == u.shape



def test_apply_standardized_student_t_inverse_cdf_returns_float_dtype() -> None:
    """Verifica que la salida se devuelve como array flotante."""
    u = np.array([0.1, 0.5, 0.9])

    result = apply_standardized_student_t_inverse_cdf(u, 5.0)

    assert np.issubdtype(result.dtype, np.floating)



def test_apply_standardized_student_t_inverse_cdf_matches_theoretical_transform_exactly() -> None:
    """Verifica que el resultado coincide con la transformacion teorica exacta."""
    u = np.array([[0.1, 0.5], [0.9, 0.75]])
    df = 5.0

    result = apply_standardized_student_t_inverse_cdf(u, df)
    expected = t.ppf(u, df) * np.sqrt((df - 2.0) / df)

    assert np.array_equal(result, expected)



def test_apply_standardized_student_t_inverse_cdf_returns_finite_values() -> None:
    """Verifica que los resultados son finitos para inputs validos."""
    u = np.array([1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6])

    result = apply_standardized_student_t_inverse_cdf(u, 5.0)

    assert np.all(np.isfinite(result))



def test_apply_standardized_student_t_inverse_cdf_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="u must be a numpy.ndarray"):
        apply_standardized_student_t_inverse_cdf([0.1, 0.5, 0.9], 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_boolean_values() -> None:
    """Verifica que no se aceptan booleanos."""
    with pytest.raises(TypeError, match="u must not contain booleans"):
        apply_standardized_student_t_inverse_cdf(np.array([True, False]), 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_non_numeric_values() -> None:
    """Verifica que el array debe contener valores numericos."""
    with pytest.raises(TypeError, match="u must contain numeric values"):
        apply_standardized_student_t_inverse_cdf(np.array(["a", "b"]), 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_nan_values() -> None:
    """Verifica que el array no acepta ``NaN``."""
    with pytest.raises(ValueError, match="u must contain finite values"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, np.nan]), 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_infinite_values() -> None:
    """Verifica que el array no acepta infinitos."""
    with pytest.raises(ValueError, match="u must contain finite values"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, np.inf]), 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_non_positive_uniform_values() -> None:
    """Verifica que no se aceptan valores menores o iguales que cero."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standardized_student_t_inverse_cdf(np.array([0.0, 0.5]), 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_greater_than_or_equal_one_uniform_values() -> None:
    """Verifica que no se aceptan valores mayores o iguales que uno."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standardized_student_t_inverse_cdf(np.array([0.5, 1.0]), 5.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_non_numeric_df() -> None:
    """Verifica que ``df`` debe ser numerico."""
    with pytest.raises(TypeError, match="df must be a number"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, 0.5]), "x")



def test_apply_standardized_student_t_inverse_cdf_raises_for_boolean_df() -> None:
    """Verifica que ``df`` no acepta booleanos."""
    with pytest.raises(TypeError, match="df must be a number"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, 0.5]), True)



def test_apply_standardized_student_t_inverse_cdf_raises_for_df_equal_to_two() -> None:
    """Verifica que ``df=2`` falla."""
    with pytest.raises(ValueError, match="df must be greater than 2"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, 0.5]), 2.0)



def test_apply_standardized_student_t_inverse_cdf_raises_for_df_below_two() -> None:
    """Verifica que ``df<2`` falla."""
    with pytest.raises(ValueError, match="df must be greater than 2"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, 0.5]), 1.5)



def test_apply_standardized_student_t_inverse_cdf_raises_for_nan_df() -> None:
    """Verifica que ``df=np.nan`` falla."""
    with pytest.raises(ValueError, match="df must be finite"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, 0.5]), np.nan)



def test_apply_standardized_student_t_inverse_cdf_raises_for_infinite_df() -> None:
    """Verifica que ``df=np.inf`` falla."""
    with pytest.raises(ValueError, match="df must be finite"):
        apply_standardized_student_t_inverse_cdf(np.array([0.1, 0.5]), np.inf)


# === Added in Step 4.8 ===

def test_apply_standardized_skew_normal_inverse_cdf_preserves_one_dimensional_shape() -> None:
    """Verifica que la transformacion preserva el shape 1D."""
    u = np.array([0.1, 0.5, 0.9])

    result = apply_standardized_skew_normal_inverse_cdf(u, 4.0)

    assert result.shape == u.shape



def test_apply_standardized_skew_normal_inverse_cdf_preserves_two_dimensional_shape() -> None:
    """Verifica que la transformacion preserva el shape 2D."""
    u = np.array([[0.1, 0.5], [0.9, 0.75]])

    result = apply_standardized_skew_normal_inverse_cdf(u, 4.0)

    assert result.shape == u.shape



def test_apply_standardized_skew_normal_inverse_cdf_returns_float_dtype() -> None:
    """Verifica que la salida se devuelve como array flotante."""
    u = np.array([0.1, 0.5, 0.9])

    result = apply_standardized_skew_normal_inverse_cdf(u, 4.0)

    assert np.issubdtype(result.dtype, np.floating)



def test_apply_standardized_skew_normal_inverse_cdf_matches_theoretical_transform_exactly() -> None:
    """Verifica que el resultado coincide con la transformacion teorica exacta."""
    u = np.array([[0.1, 0.5], [0.9, 0.75]])
    shape = 4.0

    result = apply_standardized_skew_normal_inverse_cdf(u, shape)
    raw = skewnorm.ppf(u, a=shape, loc=0, scale=1)
    delta = shape / np.sqrt(1.0 + shape**2)
    mu = delta * np.sqrt(2.0 / np.pi)
    sigma = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    expected = (raw - mu) / sigma

    assert np.array_equal(result, expected)



def test_apply_standardized_skew_normal_inverse_cdf_returns_finite_values() -> None:
    """Verifica que los resultados son finitos para inputs validos."""
    u = np.array([1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6])

    result = apply_standardized_skew_normal_inverse_cdf(u, 4.0)

    assert np.all(np.isfinite(result))



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="u must be a numpy.ndarray"):
        apply_standardized_skew_normal_inverse_cdf([0.1, 0.5, 0.9], 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_boolean_values() -> None:
    """Verifica que no se aceptan booleanos."""
    with pytest.raises(TypeError, match="u must not contain booleans"):
        apply_standardized_skew_normal_inverse_cdf(np.array([True, False]), 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_non_numeric_values() -> None:
    """Verifica que el array debe contener valores numericos."""
    with pytest.raises(TypeError, match="u must contain numeric values"):
        apply_standardized_skew_normal_inverse_cdf(np.array(["a", "b"]), 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_nan_values() -> None:
    """Verifica que el array no acepta ``NaN``."""
    with pytest.raises(ValueError, match="u must contain finite values"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.1, np.nan]), 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_infinite_values() -> None:
    """Verifica que el array no acepta infinitos."""
    with pytest.raises(ValueError, match="u must contain finite values"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.1, np.inf]), 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_non_positive_uniform_values() -> None:
    """Verifica que no se aceptan valores menores o iguales que cero."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.0, 0.5]), 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_greater_than_or_equal_one_uniform_values() -> None:
    """Verifica que no se aceptan valores mayores o iguales que uno."""
    with pytest.raises(ValueError, match="u values must be strictly between 0 and 1"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.5, 1.0]), 4.0)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_non_numeric_shape() -> None:
    """Verifica que ``shape`` debe ser numerico."""
    with pytest.raises(TypeError, match="shape must be a number"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.1, 0.5]), "x")



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_boolean_shape() -> None:
    """Verifica que ``shape`` no acepta booleanos."""
    with pytest.raises(TypeError, match="shape must be a number"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.1, 0.5]), True)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_nan_shape() -> None:
    """Verifica que ``shape=np.nan`` falla."""
    with pytest.raises(ValueError, match="shape must be finite"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.1, 0.5]), np.nan)



def test_apply_standardized_skew_normal_inverse_cdf_raises_for_infinite_shape() -> None:
    """Verifica que ``shape=np.inf`` falla."""
    with pytest.raises(ValueError, match="shape must be finite"):
        apply_standardized_skew_normal_inverse_cdf(np.array([0.1, 0.5]), np.inf)
