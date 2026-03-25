"""Pruebas minimas para las utilidades de ``tc_synthetic.utils``."""

import numpy as np
import pytest

from tc_synthetic.utils import (
    make_rng,
    validate_n_assets,
    validate_n_obs,
    validate_positive_semidefinite_matrix,
    validate_square_matrix,
    validate_symmetric_matrix,
    validate_unit_diagonal,
)


def test_make_rng_returns_generator() -> None:
    """Verifica que ``make_rng`` devuelve un generador de NumPy."""
    rng = make_rng(123)

    assert isinstance(rng, np.random.Generator)



def test_make_rng_is_reproducible_with_same_seed() -> None:
    """Verifica que la misma semilla produce la misma secuencia."""
    left = make_rng(123)
    right = make_rng(123)

    assert np.array_equal(left.normal(size=5), right.normal(size=5))



def test_make_rng_accepts_none_seed() -> None:
    """Verifica que ``seed=None`` devuelve un generador valido."""
    rng = make_rng()

    assert isinstance(rng, np.random.Generator)


@pytest.mark.parametrize("seed", [True, "123"])
def test_make_rng_raises_for_invalid_seed(seed: object) -> None:
    """Verifica que una semilla invalida produce error."""
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        make_rng(seed=seed)



def test_validate_n_assets_accepts_positive_integer() -> None:
    """Verifica que ``validate_n_assets`` acepta enteros positivos."""
    assert validate_n_assets(5) == 5


@pytest.mark.parametrize("n_assets", [0, -1])
def test_validate_n_assets_raises_for_non_positive_values(n_assets: int) -> None:
    """Verifica que ``n_assets`` debe ser mayor que cero."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        validate_n_assets(n_assets)


@pytest.mark.parametrize("n_assets", [True, 1.5])
def test_validate_n_assets_raises_for_invalid_type(n_assets: object) -> None:
    """Verifica que ``n_assets`` debe ser un entero estricto."""
    with pytest.raises(TypeError, match="n_assets must be an integer"):
        validate_n_assets(n_assets)



def test_validate_n_obs_accepts_positive_integer() -> None:
    """Verifica que ``validate_n_obs`` acepta enteros positivos."""
    assert validate_n_obs(5) == 5


@pytest.mark.parametrize("n_obs", [0, -1])
def test_validate_n_obs_raises_for_non_positive_values(n_obs: int) -> None:
    """Verifica que ``n_obs`` debe ser mayor que cero."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        validate_n_obs(n_obs)


@pytest.mark.parametrize("n_obs", [True, 1.5])
def test_validate_n_obs_raises_for_invalid_type(n_obs: object) -> None:
    """Verifica que ``n_obs`` debe ser un entero estricto."""
    with pytest.raises(TypeError, match="n_obs must be an integer"):
        validate_n_obs(n_obs)



def test_validate_square_matrix_accepts_square_2d_array() -> None:
    """Verifica que una matriz cuadrada bidimensional es valida."""
    matrix = np.eye(3)

    assert validate_square_matrix(matrix) is matrix



def test_validate_square_matrix_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="matrix must be a numpy.ndarray"):
        validate_square_matrix([[1.0, 0.0], [0.0, 1.0]])



def test_validate_square_matrix_raises_for_one_dimensional_array() -> None:
    """Verifica que la matriz debe ser bidimensional."""
    with pytest.raises(ValueError, match="covariance must be a 2D array"):
        validate_square_matrix(np.array([1.0, 2.0, 3.0]), name="covariance")



def test_validate_square_matrix_raises_for_rectangular_array() -> None:
    """Verifica que la matriz debe ser cuadrada."""
    with pytest.raises(ValueError, match="covariance must be a square matrix"):
        validate_square_matrix(np.ones((2, 3)), name="covariance")



def test_validate_symmetric_matrix_accepts_symmetric_matrix() -> None:
    """Verifica que una matriz simetrica es valida."""
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])

    assert validate_symmetric_matrix(matrix) is matrix



def test_validate_symmetric_matrix_accepts_nearly_symmetric_matrix() -> None:
    """Verifica que una matriz casi simetrica puede aceptarse por tolerancia."""
    matrix = np.array([[1.0, 0.2], [0.200000001, 1.0]])

    assert validate_symmetric_matrix(matrix, atol=1e-8) is matrix



def test_validate_symmetric_matrix_raises_for_non_symmetric_matrix() -> None:
    """Verifica que una matriz no simetrica falla."""
    matrix = np.array([[1.0, 0.2], [0.4, 1.0]])

    with pytest.raises(ValueError, match="correlation must be symmetric"):
        validate_symmetric_matrix(matrix, name="correlation")



def test_validate_symmetric_matrix_raises_for_negative_atol() -> None:
    """Verifica que la tolerancia no puede ser negativa."""
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol must be non-negative"):
        validate_symmetric_matrix(matrix, atol=-1e-8)



def test_validate_symmetric_matrix_raises_for_invalid_atol_type() -> None:
    """Verifica que la tolerancia debe ser numerica."""
    matrix = np.eye(2)

    with pytest.raises(TypeError, match="atol must be a non-negative number"):
        validate_symmetric_matrix(matrix, atol="x")



def test_validate_symmetric_matrix_returns_same_matrix() -> None:
    """Verifica que la funcion devuelve la misma matriz si valida."""
    matrix = np.eye(3)

    assert validate_symmetric_matrix(matrix) is matrix



def test_validate_unit_diagonal_accepts_unit_diagonal_matrix() -> None:
    """Verifica que una diagonal unitaria es valida."""
    matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

    assert validate_unit_diagonal(matrix) is matrix



def test_validate_unit_diagonal_accepts_nearly_unit_diagonal() -> None:
    """Verifica que una diagonal casi unitaria puede aceptarse por tolerancia."""
    matrix = np.array([[1.0 + 1e-9, 0.3], [0.3, 1.0 - 1e-9]])

    assert validate_unit_diagonal(matrix, atol=1e-8) is matrix



def test_validate_unit_diagonal_raises_for_non_unit_diagonal() -> None:
    """Verifica que una diagonal distinta de uno falla."""
    matrix = np.array([[0.9, 0.3], [0.3, 1.0]])

    with pytest.raises(ValueError, match="correlation must have a unit diagonal"):
        validate_unit_diagonal(matrix, name="correlation")



def test_validate_unit_diagonal_raises_for_negative_atol() -> None:
    """Verifica que la tolerancia no puede ser negativa."""
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol must be non-negative"):
        validate_unit_diagonal(matrix, atol=-1e-8)



def test_validate_unit_diagonal_raises_for_invalid_atol_type() -> None:
    """Verifica que la tolerancia debe ser numerica."""
    matrix = np.eye(2)

    with pytest.raises(TypeError, match="atol must be a non-negative number"):
        validate_unit_diagonal(matrix, atol="x")



def test_validate_unit_diagonal_returns_same_matrix() -> None:
    """Verifica que la funcion devuelve la misma matriz si valida."""
    matrix = np.eye(3)

    assert validate_unit_diagonal(matrix) is matrix



def test_validate_positive_semidefinite_matrix_accepts_identity() -> None:
    """Verifica que la identidad pasa la validacion PSD."""
    matrix = np.eye(3)

    assert validate_positive_semidefinite_matrix(matrix) is matrix



def test_validate_positive_semidefinite_matrix_accepts_singular_psd_matrix() -> None:
    """Verifica que una matriz PSD singular tambien pasa."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    assert validate_positive_semidefinite_matrix(matrix) is matrix



def test_validate_positive_semidefinite_matrix_returns_same_matrix() -> None:
    """Verifica que la funcion devuelve la misma matriz si valida."""
    matrix = np.eye(2)

    assert validate_positive_semidefinite_matrix(matrix) is matrix



def test_validate_positive_semidefinite_matrix_raises_for_non_psd_matrix() -> None:
    """Verifica que una matriz simetrica no PSD falla."""
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="covariance must be positive semidefinite"):
        validate_positive_semidefinite_matrix(matrix, name="covariance")



def test_validate_positive_semidefinite_matrix_raises_for_negative_atol() -> None:
    """Verifica que la tolerancia PSD no puede ser negativa."""
    matrix = np.eye(2)

    with pytest.raises(ValueError, match="atol must be non-negative"):
        validate_positive_semidefinite_matrix(matrix, atol=-1e-8)


@pytest.mark.parametrize("atol", ["x", True])
def test_validate_positive_semidefinite_matrix_raises_for_invalid_atol_type(
    atol: object,
) -> None:
    """Verifica que la tolerancia PSD debe ser numerica y no booleana."""
    matrix = np.eye(2)

    with pytest.raises(TypeError, match="atol must be a non-negative number"):
        validate_positive_semidefinite_matrix(matrix, atol=atol)
