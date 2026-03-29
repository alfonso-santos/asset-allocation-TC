"""Pruebas minimas para las copulas de ``tc_synthetic``."""

import numpy as np
import pytest
from scipy.stats import norm, t

from tc_synthetic.copulas import (
    gaussian_latent_to_uniform,
    sample_gaussian_copula,
    sample_grouped_t_copula,
    sample_independence_copula,
    sample_t_copula,
)



def test_gaussian_latent_to_uniform_preserves_shape() -> None:
    """Verifica que la transformacion preserva el shape de entrada."""
    latent = np.array([[0.0, 1.0], [-1.0, 0.5]])

    result = gaussian_latent_to_uniform(latent)

    assert result.shape == latent.shape



def test_gaussian_latent_to_uniform_returns_float_dtype() -> None:
    """Verifica que la salida se devuelve como array flotante."""
    latent = np.array([[0.0, 1.0], [-1.0, 0.5]])

    result = gaussian_latent_to_uniform(latent)

    assert np.issubdtype(result.dtype, np.floating)



def test_gaussian_latent_to_uniform_matches_norm_cdf_exactly() -> None:
    """Verifica que el resultado coincide exactamente con ``norm.cdf``."""
    latent = np.array([[0.0, 1.0], [-1.0, 0.5]])

    result = gaussian_latent_to_uniform(latent)

    assert np.array_equal(result, norm.cdf(latent))



def test_gaussian_latent_to_uniform_returns_values_between_zero_and_one() -> None:
    """Verifica que la salida queda entre cero y uno."""
    latent = np.array([[0.0, 1.0], [-1.0, 0.5]])

    result = gaussian_latent_to_uniform(latent)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)



def test_gaussian_latent_to_uniform_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="latent must be a numpy.ndarray"):
        gaussian_latent_to_uniform([[0.0, 1.0], [-1.0, 0.5]])



def test_gaussian_latent_to_uniform_raises_for_one_dimensional_input() -> None:
    """Verifica que la entrada debe ser un array bidimensional."""
    with pytest.raises(ValueError, match="latent must be a 2D array"):
        gaussian_latent_to_uniform(np.array([0.0, 1.0]))



def test_gaussian_latent_to_uniform_raises_for_zero_rows() -> None:
    """Verifica que la entrada debe tener al menos una fila."""
    with pytest.raises(ValueError, match="latent must have at least one row"):
        gaussian_latent_to_uniform(np.empty((0, 2)))



def test_gaussian_latent_to_uniform_raises_for_zero_columns() -> None:
    """Verifica que la entrada debe tener al menos una columna."""
    with pytest.raises(ValueError, match="latent must have at least one column"):
        gaussian_latent_to_uniform(np.empty((2, 0)))



def test_gaussian_latent_to_uniform_raises_for_boolean_values() -> None:
    """Verifica que no se aceptan booleanos."""
    with pytest.raises(TypeError, match="latent must not contain booleans"):
        gaussian_latent_to_uniform(np.array([[True, False], [False, True]]))



def test_gaussian_latent_to_uniform_raises_for_non_numeric_values() -> None:
    """Verifica que la entrada debe contener valores numericos."""
    with pytest.raises(TypeError, match="latent must contain numeric values"):
        gaussian_latent_to_uniform(np.array([["a", "b"], ["c", "d"]]))



def test_gaussian_latent_to_uniform_raises_for_nan_values() -> None:
    """Verifica que la entrada no acepta ``NaN``."""
    with pytest.raises(ValueError, match="latent must contain finite values"):
        gaussian_latent_to_uniform(np.array([[0.0, np.nan], [1.0, 2.0]]))



def test_gaussian_latent_to_uniform_raises_for_infinite_values() -> None:
    """Verifica que la entrada no acepta infinitos."""
    with pytest.raises(ValueError, match="latent must contain finite values"):
        gaussian_latent_to_uniform(np.array([[0.0, np.inf], [1.0, 2.0]]))



def test_sample_independence_copula_returns_expected_shape() -> None:
    """Verifica que la copula devuelve una matriz con shape correcto."""
    rng = np.random.default_rng(123)

    result = sample_independence_copula(4, 3, rng)

    assert result.shape == (4, 3)



def test_sample_independence_copula_returns_float_dtype() -> None:
    """Verifica que la salida se devuelve como array flotante."""
    rng = np.random.default_rng(123)

    result = sample_independence_copula(4, 3, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_independence_copula_returns_values_in_unit_interval() -> None:
    """Verifica que la salida queda en el intervalo unitario semiabierto."""
    rng = np.random.default_rng(123)

    result = sample_independence_copula(4, 3, rng)

    assert np.all(result >= 0.0)
    assert np.all(result < 1.0)



def test_sample_independence_copula_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma matriz."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_independence_copula(4, 3, rng_a)
    sample_b = sample_independence_copula(4, 3, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_independence_copula_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_independence_copula(0, 3, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_negative_n_obs() -> None:
    """Verifica que ``n_obs=-1`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_independence_copula(-1, 3, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_zero_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        sample_independence_copula(4, 0, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_negative_n_assets() -> None:
    """Verifica que ``n_assets=-1`` falla."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        sample_independence_copula(4, -1, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_independence_copula(True, 3, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_boolean_n_assets() -> None:
    """Verifica que ``n_assets`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_assets must be an integer"):
        sample_independence_copula(4, True, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_non_integer_n_obs() -> None:
    """Verifica que ``n_obs`` debe ser entero."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_independence_copula(1.5, 3, np.random.default_rng(123))



def test_sample_independence_copula_raises_for_non_integer_n_assets() -> None:
    """Verifica que ``n_assets`` debe ser entero."""
    with pytest.raises(TypeError, match="n_assets must be an integer"):
        sample_independence_copula(4, 2.5, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_independence_copula_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_independence_copula(4, 3, rng)



def test_sample_gaussian_copula_returns_expected_shape() -> None:
    """Verifica que la Gaussian copula devuelve una matriz con shape correcto."""
    correlation = np.array(
        [
            [1.0, 0.3],
            [0.3, 1.0],
        ]
    )
    rng = np.random.default_rng(123)

    result = sample_gaussian_copula(5, correlation, rng)

    assert result.shape == (5, 2)



def test_sample_gaussian_copula_returns_float_dtype() -> None:
    """Verifica que la Gaussian copula devuelve dtype flotante."""
    correlation = np.eye(2, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_gaussian_copula(5, correlation, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_gaussian_copula_returns_values_between_zero_and_one() -> None:
    """Verifica que la salida queda en el intervalo unitario."""
    correlation = np.eye(2, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_gaussian_copula(5, correlation, rng)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)



def test_sample_gaussian_copula_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma matriz."""
    correlation = np.array(
        [
            [1.0, 0.3],
            [0.3, 1.0],
        ]
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_gaussian_copula(5, correlation, rng_a)
    sample_b = sample_gaussian_copula(5, correlation, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_gaussian_copula_accepts_identity_correlation() -> None:
    """Verifica que una matriz identidad es una correlacion valida."""
    correlation = np.eye(3, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_gaussian_copula(4, correlation, rng)

    assert result.shape == (4, 3)



def test_sample_gaussian_copula_matches_manual_construction() -> None:
    """Verifica la coherencia exacta con la construccion manual."""
    n_obs = 6
    correlation = np.array(
        [
            [1.0, 0.4],
            [0.4, 1.0],
        ]
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_gaussian_copula(n_obs, correlation, rng_a)
    latent = rng_b.multivariate_normal(
        mean=np.zeros(correlation.shape[0], dtype=float),
        cov=correlation,
        size=n_obs,
    )
    expected = gaussian_latent_to_uniform(latent)

    assert np.array_equal(result, expected)



def test_sample_gaussian_copula_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_gaussian_copula(0, np.eye(2, dtype=float), np.random.default_rng(123))



def test_sample_gaussian_copula_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_gaussian_copula(True, np.eye(2, dtype=float), np.random.default_rng(123))



def test_sample_gaussian_copula_raises_for_non_integer_n_obs() -> None:
    """Verifica que ``n_obs`` debe ser entero."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_gaussian_copula(1.5, np.eye(2, dtype=float), np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_gaussian_copula_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_gaussian_copula(4, np.eye(2, dtype=float), rng)



def test_sample_gaussian_copula_raises_for_non_array_correlation() -> None:
    """Verifica que ``correlation`` debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="correlation must be a numpy.ndarray"):
        sample_gaussian_copula(4, [[1.0, 0.0], [0.0, 1.0]], np.random.default_rng(123))



def test_sample_gaussian_copula_raises_for_non_square_correlation() -> None:
    """Verifica que ``correlation`` debe ser cuadrada."""
    with pytest.raises(ValueError, match="correlation must be a square matrix"):
        sample_gaussian_copula(
            4,
            np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4]]),
            np.random.default_rng(123),
        )



def test_sample_gaussian_copula_raises_for_non_symmetric_correlation() -> None:
    """Verifica que ``correlation`` debe ser simetrica."""
    with pytest.raises(ValueError, match="correlation must be symmetric"):
        sample_gaussian_copula(
            4,
            np.array([[1.0, 0.2], [0.1, 1.0]]),
            np.random.default_rng(123),
        )



def test_sample_gaussian_copula_raises_for_non_unit_diagonal_correlation() -> None:
    """Verifica que ``correlation`` debe tener diagonal unitaria."""
    with pytest.raises(ValueError, match="correlation must have a unit diagonal"):
        sample_gaussian_copula(
            4,
            np.array([[1.0, 0.2], [0.2, 0.9]]),
            np.random.default_rng(123),
        )



def test_sample_gaussian_copula_raises_for_non_psd_correlation() -> None:
    """Verifica que ``correlation`` debe ser PSD."""
    with pytest.raises(ValueError, match="correlation must be positive semidefinite"):
        sample_gaussian_copula(
            4,
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.random.default_rng(123),
        )



def test_sample_gaussian_copula_raises_for_nan_correlation() -> None:
    """Verifica que ``correlation`` no acepta ``NaN``."""
    with pytest.raises(ValueError, match="correlation must contain finite values"):
        sample_gaussian_copula(
            4,
            np.array([[1.0, np.nan], [np.nan, 1.0]]),
            np.random.default_rng(123),
        )



def test_sample_gaussian_copula_raises_for_infinite_correlation() -> None:
    """Verifica que ``correlation`` no acepta infinitos."""
    with pytest.raises(ValueError, match="correlation must contain finite values"):
        sample_gaussian_copula(
            4,
            np.array([[1.0, np.inf], [np.inf, 1.0]]),
            np.random.default_rng(123),
        )



def test_sample_t_copula_returns_expected_shape() -> None:
    """Verifica que la t copula devuelve una matriz con shape correcto."""
    correlation = np.array(
        [
            [1.0, 0.3],
            [0.3, 1.0],
        ]
    )
    rng = np.random.default_rng(123)

    result = sample_t_copula(5, correlation, 5.0, rng)

    assert result.shape == (5, 2)



def test_sample_t_copula_returns_float_dtype() -> None:
    """Verifica que la t copula devuelve dtype flotante."""
    correlation = np.eye(2, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_t_copula(5, correlation, 5.0, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_t_copula_returns_values_between_zero_and_one() -> None:
    """Verifica que la salida queda en el intervalo unitario."""
    correlation = np.eye(2, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_t_copula(5, correlation, 5.0, rng)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)



def test_sample_t_copula_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma matriz."""
    correlation = np.array(
        [
            [1.0, 0.3],
            [0.3, 1.0],
        ]
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_t_copula(5, correlation, 5.0, rng_a)
    sample_b = sample_t_copula(5, correlation, 5.0, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_t_copula_accepts_identity_correlation() -> None:
    """Verifica que una matriz identidad es una correlacion valida."""
    correlation = np.eye(3, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_t_copula(4, correlation, 5.0, rng)

    assert result.shape == (4, 3)



def test_sample_t_copula_matches_manual_construction() -> None:
    """Verifica la coherencia exacta con la construccion manual."""
    n_obs = 6
    df = 5.0
    correlation = np.array(
        [
            [1.0, 0.4],
            [0.4, 1.0],
        ]
    )
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_t_copula(n_obs, correlation, df, rng_a)
    latent = rng_b.multivariate_normal(
        mean=np.zeros(correlation.shape[0], dtype=float),
        cov=correlation,
        size=n_obs,
    )
    chi2 = rng_b.chisquare(df, size=n_obs)
    t_latent = latent / np.sqrt(chi2 / df)[:, None]
    expected = t.cdf(t_latent, df=df)

    assert np.array_equal(result, expected)



def test_sample_t_copula_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_t_copula(0, np.eye(2, dtype=float), 5.0, np.random.default_rng(123))



def test_sample_t_copula_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_t_copula(True, np.eye(2, dtype=float), 5.0, np.random.default_rng(123))



def test_sample_t_copula_raises_for_non_integer_n_obs() -> None:
    """Verifica que ``n_obs`` debe ser entero."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_t_copula(1.5, np.eye(2, dtype=float), 5.0, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_t_copula_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_t_copula(4, np.eye(2, dtype=float), 5.0, rng)



def test_sample_t_copula_raises_for_non_numeric_df() -> None:
    """Verifica que ``df`` debe ser numerico."""
    with pytest.raises(TypeError, match="df must be a number"):
        sample_t_copula(4, np.eye(2, dtype=float), "x", np.random.default_rng(123))



def test_sample_t_copula_raises_for_boolean_df() -> None:
    """Verifica que ``df`` no acepta booleanos."""
    with pytest.raises(TypeError, match="df must be a number"):
        sample_t_copula(4, np.eye(2, dtype=float), True, np.random.default_rng(123))



def test_sample_t_copula_raises_for_zero_df() -> None:
    """Verifica que ``df=0`` falla."""
    with pytest.raises(ValueError, match="df must be greater than 0"):
        sample_t_copula(4, np.eye(2, dtype=float), 0.0, np.random.default_rng(123))



def test_sample_t_copula_raises_for_negative_df() -> None:
    """Verifica que ``df<0`` falla."""
    with pytest.raises(ValueError, match="df must be greater than 0"):
        sample_t_copula(4, np.eye(2, dtype=float), -1.0, np.random.default_rng(123))



def test_sample_t_copula_raises_for_nan_df() -> None:
    """Verifica que ``df=np.nan`` falla."""
    with pytest.raises(ValueError, match="df must be finite"):
        sample_t_copula(4, np.eye(2, dtype=float), np.nan, np.random.default_rng(123))



def test_sample_t_copula_raises_for_infinite_df() -> None:
    """Verifica que ``df=np.inf`` falla."""
    with pytest.raises(ValueError, match="df must be finite"):
        sample_t_copula(4, np.eye(2, dtype=float), np.inf, np.random.default_rng(123))



def test_sample_t_copula_raises_for_non_array_correlation() -> None:
    """Verifica que ``correlation`` debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="correlation must be a numpy.ndarray"):
        sample_t_copula(4, [[1.0, 0.0], [0.0, 1.0]], 5.0, np.random.default_rng(123))



def test_sample_t_copula_raises_for_non_square_correlation() -> None:
    """Verifica que ``correlation`` debe ser cuadrada."""
    with pytest.raises(ValueError, match="correlation must be a square matrix"):
        sample_t_copula(
            4,
            np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4]]),
            5.0,
            np.random.default_rng(123),
        )



def test_sample_t_copula_raises_for_non_symmetric_correlation() -> None:
    """Verifica que ``correlation`` debe ser simetrica."""
    with pytest.raises(ValueError, match="correlation must be symmetric"):
        sample_t_copula(
            4,
            np.array([[1.0, 0.2], [0.1, 1.0]]),
            5.0,
            np.random.default_rng(123),
        )



def test_sample_t_copula_raises_for_non_unit_diagonal_correlation() -> None:
    """Verifica que ``correlation`` debe tener diagonal unitaria."""
    with pytest.raises(ValueError, match="correlation must have a unit diagonal"):
        sample_t_copula(
            4,
            np.array([[1.0, 0.2], [0.2, 0.9]]),
            5.0,
            np.random.default_rng(123),
        )



def test_sample_t_copula_raises_for_non_psd_correlation() -> None:
    """Verifica que ``correlation`` debe ser PSD."""
    with pytest.raises(ValueError, match="correlation must be positive semidefinite"):
        sample_t_copula(
            4,
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            5.0,
            np.random.default_rng(123),
        )



def test_sample_t_copula_raises_for_nan_correlation() -> None:
    """Verifica que ``correlation`` no acepta ``NaN``."""
    with pytest.raises(ValueError, match="correlation must contain finite values"):
        sample_t_copula(
            4,
            np.array([[1.0, np.nan], [np.nan, 1.0]]),
            5.0,
            np.random.default_rng(123),
        )



def test_sample_t_copula_raises_for_infinite_correlation() -> None:
    """Verifica que ``correlation`` no acepta infinitos."""
    with pytest.raises(ValueError, match="correlation must contain finite values"):
        sample_t_copula(
            4,
            np.array([[1.0, np.inf], [np.inf, 1.0]]),
            5.0,
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_returns_expected_shape() -> None:
    """Verifica que la grouped t copula devuelve una matriz con shape correcto."""
    correlation = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.25],
            [0.1, 0.25, 1.0],
        ]
    )
    group_assignments = np.array([0, 0, 1])
    group_dfs = {0: 5.0, 1: 10.0}
    rng = np.random.default_rng(123)

    result = sample_grouped_t_copula(5, correlation, group_assignments, group_dfs, rng)

    assert result.shape == (5, 3)



def test_sample_grouped_t_copula_returns_float_dtype() -> None:
    """Verifica que la grouped t copula devuelve dtype flotante."""
    correlation = np.eye(3, dtype=float)
    group_assignments = np.array([0, 0, 1])
    group_dfs = {0: 5.0, 1: 10.0}
    rng = np.random.default_rng(123)

    result = sample_grouped_t_copula(5, correlation, group_assignments, group_dfs, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_grouped_t_copula_returns_values_between_zero_and_one() -> None:
    """Verifica que la salida queda en el intervalo unitario."""
    correlation = np.eye(3, dtype=float)
    group_assignments = np.array([0, 0, 1])
    group_dfs = {0: 5.0, 1: 10.0}
    rng = np.random.default_rng(123)

    result = sample_grouped_t_copula(5, correlation, group_assignments, group_dfs, rng)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)



def test_sample_grouped_t_copula_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma matriz."""
    correlation = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.25],
            [0.1, 0.25, 1.0],
        ]
    )
    group_assignments = np.array([0, 0, 1])
    group_dfs = {0: 5.0, 1: 10.0}
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_grouped_t_copula(5, correlation, group_assignments, group_dfs, rng_a)
    sample_b = sample_grouped_t_copula(5, correlation, group_assignments, group_dfs, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_grouped_t_copula_matches_manual_construction() -> None:
    """Verifica la coherencia exacta con la construccion manual."""
    n_obs = 6
    correlation = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.25],
            [0.1, 0.25, 1.0],
        ]
    )
    group_assignments = np.array([0, 0, 1])
    group_dfs = {0: 5.0, 1: 10.0}
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_grouped_t_copula(n_obs, correlation, group_assignments, group_dfs, rng_a)
    latent = rng_b.multivariate_normal(
        mean=np.zeros(correlation.shape[0], dtype=float),
        cov=correlation,
        size=n_obs,
    )
    expected = np.empty((n_obs, correlation.shape[0]), dtype=float)
    for group in np.unique(group_assignments):
        group = int(group)
        group_mask = group_assignments == group
        group_df = group_dfs[group]
        chi2 = rng_b.chisquare(group_df, size=n_obs)
        t_latent = latent[:, group_mask] / np.sqrt(chi2 / group_df)[:, None]
        expected[:, group_mask] = t.cdf(t_latent, df=group_df)

    assert np.array_equal(result, expected)



def test_sample_grouped_t_copula_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_grouped_t_copula(
            0,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: 5.0, 1: 10.0},
            np.random.default_rng(123),
        )



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_grouped_t_copula_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: 5.0, 1: 10.0},
            rng,
        )



def test_sample_grouped_t_copula_raises_for_non_array_group_assignments() -> None:
    """Verifica que ``group_assignments`` debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="group_assignments must be a numpy.ndarray"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            [0, 0, 1],
            {0: 5.0, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_non_one_dimensional_group_assignments() -> None:
    """Verifica que ``group_assignments`` debe ser 1D."""
    with pytest.raises(ValueError, match="group_assignments must be a 1D array"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([[0, 0, 1]]),
            {0: 5.0, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_group_assignment_length_mismatch() -> None:
    """Verifica que la longitud debe coincidir con ``n_assets``."""
    with pytest.raises(ValueError, match="group_assignments length must match n_assets"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0]),
            {0: 5.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_boolean_group_assignments() -> None:
    """Verifica que ``group_assignments`` no acepta booleanos."""
    with pytest.raises(TypeError, match="group_assignments must contain integers"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([True, False, True]),
            {0: 5.0, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_float_group_assignments() -> None:
    """Verifica que ``group_assignments`` debe contener enteros."""
    with pytest.raises(TypeError, match="group_assignments must contain integers"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0.0, 0.0, 1.0]),
            {0: 5.0, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_non_dict_group_dfs() -> None:
    """Verifica que ``group_dfs`` debe ser un diccionario."""
    with pytest.raises(TypeError, match="group_dfs must be a dict"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            [(0, 5.0), (1, 10.0)],
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_empty_group_dfs() -> None:
    """Verifica que ``group_dfs`` no puede estar vacio."""
    with pytest.raises(ValueError, match="group_dfs must not be empty"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_missing_group_df() -> None:
    """Verifica que ``group_dfs`` debe cubrir exactamente los grupos presentes."""
    with pytest.raises(ValueError, match="group_dfs must cover exactly the groups in group_assignments"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: 5.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_non_numeric_group_df() -> None:
    """Verifica que cada group df debe ser numerico."""
    with pytest.raises(TypeError, match="group df must be a number"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: "x", 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_boolean_group_df() -> None:
    """Verifica que cada group df no acepta booleanos."""
    with pytest.raises(TypeError, match="group df must be a number"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: True, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_nan_group_df() -> None:
    """Verifica que cada group df debe ser finito."""
    with pytest.raises(ValueError, match="group df must be finite"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: np.nan, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_infinite_group_df() -> None:
    """Verifica que cada group df no acepta infinitos."""
    with pytest.raises(ValueError, match="group df must be finite"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: np.inf, 1: 10.0},
            np.random.default_rng(123),
        )



def test_sample_grouped_t_copula_raises_for_non_positive_group_df() -> None:
    """Verifica que cada group df debe ser positivo."""
    with pytest.raises(ValueError, match="group df must be greater than 0"):
        sample_grouped_t_copula(
            4,
            np.eye(3, dtype=float),
            np.array([0, 0, 1]),
            {0: 0.0, 1: 10.0},
            np.random.default_rng(123),
        )

# === Added in Step 5.6 ===
from tc_synthetic.copulas import sample_clayton_copula



def test_sample_clayton_copula_returns_expected_shape() -> None:
    """Verifica que la Clayton copula devuelve una matriz con shape correcto."""
    rng = np.random.default_rng(123)

    result = sample_clayton_copula(5, 3, 2.0, rng)

    assert result.shape == (5, 3)



def test_sample_clayton_copula_returns_float_dtype() -> None:
    """Verifica que la Clayton copula devuelve dtype flotante."""
    rng = np.random.default_rng(123)

    result = sample_clayton_copula(5, 3, 2.0, rng)

    assert np.issubdtype(result.dtype, np.floating)



def test_sample_clayton_copula_returns_values_between_zero_and_one() -> None:
    """Verifica que la salida queda en el intervalo unitario."""
    rng = np.random.default_rng(123)

    result = sample_clayton_copula(5, 3, 2.0, rng)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)



def test_sample_clayton_copula_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma matriz."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_clayton_copula(5, 3, 2.0, rng_a)
    sample_b = sample_clayton_copula(5, 3, 2.0, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_clayton_copula_matches_manual_construction() -> None:
    """Verifica la coherencia exacta con la construccion manual."""
    n_obs = 6
    n_assets = 3
    theta = 2.0
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_clayton_copula(n_obs, n_assets, theta, rng_a)
    v = rng_b.gamma(shape=1.0 / theta, scale=1.0, size=n_obs)
    e = rng_b.exponential(scale=1.0, size=(n_obs, n_assets))
    expected = (1.0 + e / v[:, None]) ** (-1.0 / theta)

    assert np.array_equal(result, expected)



def test_sample_clayton_copula_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_clayton_copula(0, 3, 2.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_zero_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        sample_clayton_copula(5, 0, 2.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_boolean_n_obs() -> None:
    """Verifica que ``n_obs`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_clayton_copula(True, 3, 2.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_boolean_n_assets() -> None:
    """Verifica que ``n_assets`` no acepta booleanos."""
    with pytest.raises(TypeError, match="n_assets must be an integer"):
        sample_clayton_copula(5, True, 2.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_non_integer_n_obs() -> None:
    """Verifica que ``n_obs`` debe ser entero."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        sample_clayton_copula(1.5, 3, 2.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_non_integer_n_assets() -> None:
    """Verifica que ``n_assets`` debe ser entero."""
    with pytest.raises(TypeError, match="n_assets must be an integer"):
        sample_clayton_copula(5, 2.5, 2.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_non_numeric_theta() -> None:
    """Verifica que ``theta`` debe ser numerico."""
    with pytest.raises(TypeError, match="theta must be a number"):
        sample_clayton_copula(5, 3, "x", np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_boolean_theta() -> None:
    """Verifica que ``theta`` no acepta booleanos."""
    with pytest.raises(TypeError, match="theta must be a number"):
        sample_clayton_copula(5, 3, True, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_zero_theta() -> None:
    """Verifica que ``theta=0`` falla."""
    with pytest.raises(ValueError, match="theta must be greater than 0"):
        sample_clayton_copula(5, 3, 0.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_negative_theta() -> None:
    """Verifica que ``theta<0`` falla."""
    with pytest.raises(ValueError, match="theta must be greater than 0"):
        sample_clayton_copula(5, 3, -1.0, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_nan_theta() -> None:
    """Verifica que ``theta=np.nan`` falla."""
    with pytest.raises(ValueError, match="theta must be finite"):
        sample_clayton_copula(5, 3, np.nan, np.random.default_rng(123))



def test_sample_clayton_copula_raises_for_infinite_theta() -> None:
    """Verifica que ``theta=np.inf`` falla."""
    with pytest.raises(ValueError, match="theta must be finite"):
        sample_clayton_copula(5, 3, np.inf, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_clayton_copula_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_clayton_copula(5, 3, 2.0, rng)
