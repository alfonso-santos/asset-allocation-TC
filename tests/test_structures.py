"""Pruebas minimas para las estructuras de ``tc_synthetic``."""

import numpy as np
import pytest

from tc_synthetic.structures import (
    build_block_correlation_matrix,
    build_equicorrelation_matrix,
    build_factor_correlation_matrix,
    build_near_duplicate_correlation_matrix,
    build_nonlinear_redundancy_groups,
    build_one_factor_correlation_matrix,
)
from tc_synthetic.utils import (
    validate_positive_semidefinite_matrix,
    validate_square_matrix,
    validate_symmetric_matrix,
    validate_unit_diagonal,
)


def test_build_equicorrelation_matrix_returns_expected_matrix() -> None:
    """Verifica que la matriz equicorrelacionada coincide con la esperada."""
    matrix = build_equicorrelation_matrix(3, 0.2)
    expected = np.array(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.2, 1.0],
        ],
        dtype=float,
    )

    assert np.array_equal(matrix, expected)



def test_build_equicorrelation_matrix_has_unit_diagonal() -> None:
    """Verifica que la diagonal queda fijada en uno."""
    matrix = build_equicorrelation_matrix(4, 0.3)

    assert np.array_equal(np.diag(matrix), np.ones(4))



def test_build_equicorrelation_matrix_sets_off_diagonal_to_rho() -> None:
    """Verifica que los elementos fuera de diagonal valen ``rho``."""
    rho = 0.4
    matrix = build_equicorrelation_matrix(4, rho)
    off_diagonal = matrix[~np.eye(4, dtype=bool)]

    assert np.array_equal(off_diagonal, np.full(12, rho))



def test_build_equicorrelation_matrix_has_expected_shape() -> None:
    """Verifica que la matriz tiene la forma esperada."""
    matrix = build_equicorrelation_matrix(5, 0.1)

    assert matrix.shape == (5, 5)



def test_build_equicorrelation_matrix_has_float_dtype() -> None:
    """Verifica que la matriz devuelta tiene tipo flotante."""
    matrix = build_equicorrelation_matrix(3, 0)

    assert np.issubdtype(matrix.dtype, np.floating)



def test_build_equicorrelation_matrix_accepts_upper_boundary() -> None:
    """Verifica que ``rho=1.0`` se acepta."""
    matrix = build_equicorrelation_matrix(3, 1.0)

    assert np.array_equal(matrix, np.ones((3, 3), dtype=float))



def test_build_equicorrelation_matrix_accepts_lower_boundary() -> None:
    """Verifica que la frontera inferior admisible se acepta."""
    rho = -1 / 2
    matrix = build_equicorrelation_matrix(3, rho)

    assert np.array_equal(
        matrix,
        np.array(
            [
                [1.0, rho, rho],
                [rho, 1.0, rho],
                [rho, rho, 1.0],
            ],
            dtype=float,
        ),
    )



def test_build_equicorrelation_matrix_returns_singleton_matrix_for_one_asset() -> None:
    """Verifica que con un activo se devuelve ``[[1.0]]``."""
    matrix = build_equicorrelation_matrix(1, 0.7)

    assert np.array_equal(matrix, np.array([[1.0]], dtype=float))



def test_build_equicorrelation_matrix_raises_for_invalid_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        build_equicorrelation_matrix(0, 0.2)


@pytest.mark.parametrize("rho", ["x", True])
def test_build_equicorrelation_matrix_raises_for_invalid_rho_type(rho: object) -> None:
    """Verifica que ``rho`` debe ser numerico y no booleano."""
    with pytest.raises(TypeError, match="rho must be a number"):
        build_equicorrelation_matrix(3, rho)



def test_build_equicorrelation_matrix_raises_for_rho_above_one() -> None:
    """Verifica que ``rho > 1`` falla."""
    with pytest.raises(ValueError, match="rho must be less than or equal to 1"):
        build_equicorrelation_matrix(3, 1.1)



def test_build_equicorrelation_matrix_raises_for_rho_below_lower_bound() -> None:
    """Verifica que ``rho`` por debajo de la cota inferior falla."""
    with pytest.raises(ValueError, match="rho must be greater than or equal to -0.5"):
        build_equicorrelation_matrix(3, -0.6)



def test_build_equicorrelation_matrix_passes_matrix_validators() -> None:
    """Verifica que una matriz valida pasa los validadores matriciales basicos."""
    matrix = build_equicorrelation_matrix(4, 0.25)

    assert validate_square_matrix(matrix, name="correlation") is matrix
    assert validate_symmetric_matrix(matrix, name="correlation") is matrix
    assert validate_unit_diagonal(matrix, name="correlation") is matrix



def test_build_block_correlation_matrix_has_expected_shape() -> None:
    """Verifica que la matriz por bloques tiene la forma esperada."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert matrix.shape == (5, 5)



def test_build_block_correlation_matrix_has_unit_diagonal() -> None:
    """Verifica que la diagonal de la matriz por bloques es unitaria."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert np.array_equal(np.diag(matrix), np.ones(5))



def test_build_block_correlation_matrix_sets_within_block_values() -> None:
    """Verifica que los bloques internos usan ``rho_within`` fuera de diagonal."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert np.array_equal(matrix[:2, :2], np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float))
    assert np.array_equal(
        matrix[2:, 2:],
        np.array(
            [
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ],
            dtype=float,
        ),
    )



def test_build_block_correlation_matrix_sets_between_block_values() -> None:
    """Verifica que los bloques fuera de diagonal usan ``rho_between``."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert np.array_equal(matrix[:2, 2:], np.full((2, 3), 0.1, dtype=float))
    assert np.array_equal(matrix[2:, :2], np.full((3, 2), 0.1, dtype=float))



def test_build_block_correlation_matrix_has_float_dtype() -> None:
    """Verifica que la matriz por bloques tiene tipo flotante."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert np.issubdtype(matrix.dtype, np.floating)



def test_build_block_correlation_matrix_accepts_single_block_case() -> None:
    """Verifica que el caso de un solo bloque funciona correctamente."""
    matrix = build_block_correlation_matrix([4], 0.5, 0.1)
    expected = np.array(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.5],
            [0.5, 0.5, 0.5, 1.0],
        ],
        dtype=float,
    )

    assert np.array_equal(matrix, expected)



def test_build_block_correlation_matrix_raises_for_empty_block_sizes() -> None:
    """Verifica que ``block_sizes`` no puede estar vacio."""
    with pytest.raises(ValueError, match="block_sizes must not be empty"):
        build_block_correlation_matrix([], 0.5, 0.1)



def test_build_block_correlation_matrix_raises_for_non_list_block_sizes() -> None:
    """Verifica que ``block_sizes`` debe ser una lista."""
    with pytest.raises(TypeError, match="block_sizes must be a list"):
        build_block_correlation_matrix((2, 3), 0.5, 0.1)


@pytest.mark.parametrize("block_sizes", [[2, 0], [2, -1]])
def test_build_block_correlation_matrix_raises_for_non_positive_block_sizes(
    block_sizes: list[int],
) -> None:
    """Verifica que los tamanos de bloque deben ser positivos."""
    with pytest.raises(ValueError, match="block_sizes must contain positive integers"):
        build_block_correlation_matrix(block_sizes, 0.5, 0.1)



def test_build_block_correlation_matrix_raises_for_boolean_block_size() -> None:
    """Verifica que los tamanos de bloque no aceptan booleanos."""
    with pytest.raises(TypeError, match="block_sizes must contain integers"):
        build_block_correlation_matrix([2, True], 0.5, 0.1)



def test_build_block_correlation_matrix_raises_for_invalid_rho_within_type() -> None:
    """Verifica que ``rho_within`` debe ser numerico y no booleano."""
    with pytest.raises(TypeError, match="rho_within must be a number"):
        build_block_correlation_matrix([2, 3], "x", 0.1)



def test_build_block_correlation_matrix_raises_for_invalid_rho_between_type() -> None:
    """Verifica que ``rho_between`` debe ser numerico y no booleano."""
    with pytest.raises(TypeError, match="rho_between must be a number"):
        build_block_correlation_matrix([2, 3], 0.5, "x")



def test_build_block_correlation_matrix_raises_for_boolean_rho_within() -> None:
    """Verifica que ``rho_within`` no acepta booleanos."""
    with pytest.raises(TypeError, match="rho_within must be a number"):
        build_block_correlation_matrix([2, 3], True, 0.1)



def test_build_block_correlation_matrix_raises_for_boolean_rho_between() -> None:
    """Verifica que ``rho_between`` no acepta booleanos."""
    with pytest.raises(TypeError, match="rho_between must be a number"):
        build_block_correlation_matrix([2, 3], 0.5, True)



def test_build_block_correlation_matrix_raises_for_rho_within_above_one() -> None:
    """Verifica que ``rho_within > 1`` falla."""
    with pytest.raises(ValueError, match="rho_within must be less than or equal to 1"):
        build_block_correlation_matrix([2, 3], 1.1, 0.1)



def test_build_block_correlation_matrix_raises_for_rho_between_above_one() -> None:
    """Verifica que ``rho_between > 1`` falla."""
    with pytest.raises(ValueError, match="rho_between must be less than or equal to 1"):
        build_block_correlation_matrix([2, 3], 0.5, 1.1)



def test_build_block_correlation_matrix_valid_case_still_works() -> None:
    """Verifica que un caso valido de bloques sigue funcionando."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert matrix.shape == (5, 5)



def test_build_block_correlation_matrix_raises_for_non_psd_configuration() -> None:
    """Verifica que una configuracion no PSD produce error."""
    with pytest.raises(ValueError, match="correlation must be positive semidefinite"):
        build_block_correlation_matrix([2, 3], 0.5, 0.9)



def test_build_block_correlation_matrix_passes_matrix_validators() -> None:
    """Verifica que una matriz por bloques valida pasa los validadores matriciales."""
    matrix = build_block_correlation_matrix([2, 3], 0.5, 0.1)

    assert validate_square_matrix(matrix, name="correlation") is matrix
    assert validate_symmetric_matrix(matrix, name="correlation") is matrix
    assert validate_unit_diagonal(matrix, name="correlation") is matrix
    assert validate_positive_semidefinite_matrix(matrix, name="correlation") is matrix



def test_build_near_duplicate_correlation_matrix_has_expected_shape() -> None:
    """Verifica que la matriz near-duplicates tiene la forma esperada."""
    matrix = build_near_duplicate_correlation_matrix([2, 2], 0.95, 0.1)

    assert matrix.shape == (4, 4)



def test_build_near_duplicate_correlation_matrix_has_unit_diagonal() -> None:
    """Verifica que la diagonal es unitaria."""
    matrix = build_near_duplicate_correlation_matrix([2, 2], 0.95, 0.1)

    assert np.array_equal(np.diag(matrix), np.ones(4))



def test_build_near_duplicate_correlation_matrix_sets_intra_group_values() -> None:
    """Verifica que la correlacion intra-grupo vale ``rho_duplicate``."""
    matrix = build_near_duplicate_correlation_matrix([2, 2], 0.95, 0.1)

    assert np.array_equal(matrix[:2, :2], np.array([[1.0, 0.95], [0.95, 1.0]], dtype=float))
    assert np.array_equal(matrix[2:, 2:], np.array([[1.0, 0.95], [0.95, 1.0]], dtype=float))



def test_build_near_duplicate_correlation_matrix_sets_inter_group_values() -> None:
    """Verifica que la correlacion inter-grupo vale ``rho_background``."""
    matrix = build_near_duplicate_correlation_matrix([2, 2], 0.95, 0.1)

    assert np.array_equal(matrix[:2, 2:], np.full((2, 2), 0.1, dtype=float))
    assert np.array_equal(matrix[2:, :2], np.full((2, 2), 0.1, dtype=float))



def test_build_near_duplicate_correlation_matrix_has_float_dtype() -> None:
    """Verifica que la matriz devuelta tiene dtype flotante."""
    matrix = build_near_duplicate_correlation_matrix([2, 2], 0.95, 0.1)

    assert np.issubdtype(matrix.dtype, np.floating)



def test_build_near_duplicate_correlation_matrix_accepts_single_group_case() -> None:
    """Verifica que el caso de un solo grupo funciona correctamente."""
    matrix = build_near_duplicate_correlation_matrix([3], 0.95, 0.1)
    expected = np.array(
        [
            [1.0, 0.95, 0.95],
            [0.95, 1.0, 0.95],
            [0.95, 0.95, 1.0],
        ],
        dtype=float,
    )

    assert np.array_equal(matrix, expected)



def test_build_near_duplicate_correlation_matrix_raises_for_empty_group_sizes() -> None:
    """Verifica que ``group_sizes`` no puede estar vacio."""
    with pytest.raises(ValueError, match="block_sizes must not be empty"):
        build_near_duplicate_correlation_matrix([], 0.95, 0.1)



def test_build_near_duplicate_correlation_matrix_raises_for_non_list_group_sizes() -> None:
    """Verifica que ``group_sizes`` debe ser una lista."""
    with pytest.raises(TypeError, match="block_sizes must be a list"):
        build_near_duplicate_correlation_matrix((2, 2), 0.95, 0.1)


@pytest.mark.parametrize("group_sizes", [[2, 0], [2, -1]])
def test_build_near_duplicate_correlation_matrix_raises_for_non_positive_group_sizes(
    group_sizes: list[int],
) -> None:
    """Verifica que los tamanos de grupo deben ser positivos."""
    with pytest.raises(ValueError, match="block_sizes must contain positive integers"):
        build_near_duplicate_correlation_matrix(group_sizes, 0.95, 0.1)



def test_build_near_duplicate_correlation_matrix_raises_for_boolean_group_size() -> None:
    """Verifica que los tamanos de grupo no aceptan booleanos."""
    with pytest.raises(TypeError, match="block_sizes must contain integers"):
        build_near_duplicate_correlation_matrix([2, True], 0.95, 0.1)



def test_build_near_duplicate_correlation_matrix_raises_for_invalid_rho_duplicate_type() -> None:
    """Verifica que ``rho_duplicate`` debe ser numerico y no booleano."""
    with pytest.raises(TypeError, match="rho_duplicate must be a number"):
        build_near_duplicate_correlation_matrix([2, 2], "x", 0.1)



def test_build_near_duplicate_correlation_matrix_raises_for_invalid_rho_background_type() -> None:
    """Verifica que ``rho_background`` debe ser numerico y no booleano."""
    with pytest.raises(TypeError, match="rho_background must be a number"):
        build_near_duplicate_correlation_matrix([2, 2], 0.95, "x")



def test_build_near_duplicate_correlation_matrix_raises_for_boolean_rho_duplicate() -> None:
    """Verifica que ``rho_duplicate`` no acepta booleanos."""
    with pytest.raises(TypeError, match="rho_duplicate must be a number"):
        build_near_duplicate_correlation_matrix([2, 2], True, 0.1)



def test_build_near_duplicate_correlation_matrix_raises_for_boolean_rho_background() -> None:
    """Verifica que ``rho_background`` no acepta booleanos."""
    with pytest.raises(TypeError, match="rho_background must be a number"):
        build_near_duplicate_correlation_matrix([2, 2], 0.95, True)



def test_build_near_duplicate_correlation_matrix_raises_for_rho_duplicate_above_one() -> None:
    """Verifica que ``rho_duplicate > 1`` falla."""
    with pytest.raises(ValueError, match="rho_duplicate must be less than or equal to 1"):
        build_near_duplicate_correlation_matrix([2, 2], 1.1, 0.1)



def test_build_near_duplicate_correlation_matrix_raises_for_rho_background_above_one() -> None:
    """Verifica que ``rho_background > 1`` falla."""
    with pytest.raises(ValueError, match="rho_background must be less than or equal to 1"):
        build_near_duplicate_correlation_matrix([2, 2], 0.95, 1.1)



def test_build_near_duplicate_correlation_matrix_raises_when_duplicate_is_below_background() -> None:
    """Verifica que ``rho_duplicate`` no puede ser menor que ``rho_background``."""
    with pytest.raises(ValueError, match="rho_duplicate must be greater than or equal to rho_background"):
        build_near_duplicate_correlation_matrix([2, 2], 0.1, 0.2)



def test_near_duplicate_builder_matches_block_builder() -> None:
    """Verifica la coherencia con la funcion de bloques."""
    group_sizes = [2, 2]
    rho_duplicate = 0.95
    rho_background = 0.1

    assert np.array_equal(
        build_near_duplicate_correlation_matrix(group_sizes, rho_duplicate, rho_background),
        build_block_correlation_matrix(group_sizes, rho_duplicate, rho_background),
    )



def test_build_factor_correlation_matrix_returns_expected_shape() -> None:
    """Verifica que la matriz multifactor tiene el shape esperado."""
    loadings = np.array([[0.5, 0.1], [0.2, -0.3], [0.0, 0.4]])
    matrix = build_factor_correlation_matrix(loadings)

    assert matrix.shape == (3, 3)



def test_build_factor_correlation_matrix_has_unit_diagonal() -> None:
    """Verifica que la matriz multifactor tiene diagonal unitaria."""
    loadings = np.array([[0.5, 0.1], [0.2, -0.3], [0.0, 0.4]])
    matrix = build_factor_correlation_matrix(loadings)

    assert np.array_equal(np.diag(matrix), np.ones(3))



def test_build_factor_correlation_matrix_has_float_dtype() -> None:
    """Verifica que la matriz multifactor tiene dtype flotante."""
    loadings = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
    matrix = build_factor_correlation_matrix(loadings)

    assert np.issubdtype(matrix.dtype, np.floating)



def test_build_factor_correlation_matrix_passes_matrix_validators() -> None:
    """Verifica que una matriz multifactor valida pasa los validadores."""
    loadings = np.array([[0.5, 0.1], [0.2, -0.3], [0.0, 0.4]])
    matrix = build_factor_correlation_matrix(loadings)

    assert validate_square_matrix(matrix, name="correlation") is matrix
    assert validate_symmetric_matrix(matrix, name="correlation") is matrix
    assert validate_unit_diagonal(matrix, name="correlation") is matrix
    assert validate_positive_semidefinite_matrix(matrix, name="correlation") is matrix



def test_build_factor_correlation_matrix_accepts_row_norm_boundary() -> None:
    """Verifica que una fila con suma de cuadrados exactamente uno se acepta."""
    loadings = np.array([[0.6, 0.8], [0.0, 0.0]])
    matrix = build_factor_correlation_matrix(loadings)

    assert np.array_equal(np.diag(matrix), np.ones(2))



def test_build_factor_correlation_matrix_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="loadings must be a numpy.ndarray"):
        build_factor_correlation_matrix([[0.5, 0.1], [0.2, -0.3]])



def test_build_factor_correlation_matrix_raises_for_one_dimensional_input() -> None:
    """Verifica que ``loadings`` debe ser un array bidimensional."""
    with pytest.raises(ValueError, match="loadings must be a 2D array"):
        build_factor_correlation_matrix(np.array([0.5, 0.2, -0.3]))



def test_build_factor_correlation_matrix_raises_for_empty_rows() -> None:
    """Verifica que ``loadings`` debe tener al menos una fila."""
    with pytest.raises(ValueError, match="loadings must have at least one row"):
        build_factor_correlation_matrix(np.empty((0, 2)))



def test_build_factor_correlation_matrix_raises_for_empty_columns() -> None:
    """Verifica que ``loadings`` debe tener al menos una columna."""
    with pytest.raises(ValueError, match="loadings must have at least one column"):
        build_factor_correlation_matrix(np.empty((3, 0)))



def test_build_factor_correlation_matrix_raises_for_boolean_values() -> None:
    """Verifica que ``loadings`` no acepta booleanos."""
    with pytest.raises(TypeError, match="loadings must not contain booleans"):
        build_factor_correlation_matrix(np.array([[True, False], [False, True]]))



def test_build_factor_correlation_matrix_raises_for_non_numeric_values() -> None:
    """Verifica que ``loadings`` debe contener valores numericos."""
    with pytest.raises(TypeError, match="loadings must contain numeric values"):
        build_factor_correlation_matrix(np.array([["a", "b"], ["c", "d"]]))



def test_build_factor_correlation_matrix_raises_for_nan_values() -> None:
    """Verifica que ``loadings`` no acepta ``NaN``."""
    with pytest.raises(ValueError, match="loadings must contain finite values"):
        build_factor_correlation_matrix(np.array([[0.5, 0.1], [np.nan, 0.2]]))



def test_build_factor_correlation_matrix_raises_for_infinite_values() -> None:
    """Verifica que ``loadings`` no acepta infinitos."""
    with pytest.raises(ValueError, match="loadings must contain finite values"):
        build_factor_correlation_matrix(np.array([[0.5, 0.1], [np.inf, 0.2]]))



def test_build_factor_correlation_matrix_raises_for_row_norm_above_one() -> None:
    """Verifica que cada fila debe cumplir la cota de suma de cuadrados."""
    with pytest.raises(ValueError, match="each row must satisfy sum of squares <= 1"):
        build_factor_correlation_matrix(np.array([[0.9, 0.9], [0.1, 0.2]]))



def test_one_factor_builder_matches_general_factor_builder() -> None:
    """Verifica la coherencia entre la version de un factor y la general."""
    beta = np.array([0.5, 0.2, -0.3])

    assert np.array_equal(
        build_one_factor_correlation_matrix(beta),
        build_factor_correlation_matrix(beta.reshape(-1, 1)),
    )



def test_build_one_factor_correlation_matrix_returns_expected_shape() -> None:
    """Verifica que la matriz de un factor tiene el shape esperado."""
    beta = np.array([0.5, 0.2, -0.3])
    matrix = build_one_factor_correlation_matrix(beta)

    assert matrix.shape == (3, 3)



def test_build_one_factor_correlation_matrix_has_unit_diagonal() -> None:
    """Verifica que la diagonal queda fijada en uno."""
    beta = np.array([0.5, 0.2, -0.3])
    matrix = build_one_factor_correlation_matrix(beta)

    assert np.array_equal(np.diag(matrix), np.ones(3))



def test_build_one_factor_correlation_matrix_sets_off_diagonal_from_loadings() -> None:
    """Verifica que la fuera de diagonal vale ``beta_i * beta_j``."""
    beta = np.array([0.5, 0.2, -0.3])
    matrix = build_one_factor_correlation_matrix(beta)
    expected = np.outer(beta, beta) + np.diag(1.0 - beta**2)

    assert np.array_equal(matrix, expected)



def test_build_one_factor_correlation_matrix_has_float_dtype() -> None:
    """Verifica que la matriz devuelta tiene tipo flotante."""
    beta = np.array([1, 0, -1], dtype=int)
    matrix = build_one_factor_correlation_matrix(beta)

    assert np.issubdtype(matrix.dtype, np.floating)



def test_build_one_factor_correlation_matrix_accepts_upper_boundary() -> None:
    """Verifica que un loading igual a 1.0 se acepta."""
    matrix = build_one_factor_correlation_matrix(np.array([1.0, 0.0]))

    assert np.array_equal(matrix, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))



def test_build_one_factor_correlation_matrix_accepts_lower_boundary() -> None:
    """Verifica que un loading igual a -1.0 se acepta."""
    matrix = build_one_factor_correlation_matrix(np.array([-1.0, 0.0]))

    assert np.array_equal(matrix, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))



def test_build_one_factor_correlation_matrix_passes_matrix_validators() -> None:
    """Verifica que una matriz valida de un factor pasa los validadores."""
    beta = np.array([0.5, 0.2, -0.3])
    matrix = build_one_factor_correlation_matrix(beta)

    assert validate_square_matrix(matrix, name="correlation") is matrix
    assert validate_symmetric_matrix(matrix, name="correlation") is matrix
    assert validate_unit_diagonal(matrix, name="correlation") is matrix
    assert validate_positive_semidefinite_matrix(matrix, name="correlation") is matrix



def test_build_one_factor_correlation_matrix_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="loadings must be a numpy.ndarray"):
        build_one_factor_correlation_matrix([0.5, 0.2, -0.3])



def test_build_one_factor_correlation_matrix_raises_for_two_dimensional_input() -> None:
    """Verifica que ``loadings`` debe ser un array unidimensional."""
    with pytest.raises(ValueError, match="loadings must be a 1D array"):
        build_one_factor_correlation_matrix(np.array([[0.5, 0.2, -0.3]]))



def test_build_one_factor_correlation_matrix_raises_for_empty_input() -> None:
    """Verifica que ``loadings`` no puede estar vacio."""
    with pytest.raises(ValueError, match="loadings must not be empty"):
        build_one_factor_correlation_matrix(np.array([], dtype=float))



def test_build_one_factor_correlation_matrix_raises_for_boolean_values() -> None:
    """Verifica que ``loadings`` no acepta booleanos."""
    with pytest.raises(TypeError, match="loadings must not contain booleans"):
        build_one_factor_correlation_matrix(np.array([True, False]))



def test_build_one_factor_correlation_matrix_raises_for_non_numeric_values() -> None:
    """Verifica que ``loadings`` debe contener valores numericos."""
    with pytest.raises(TypeError, match="loadings must contain numeric values"):
        build_one_factor_correlation_matrix(np.array(["a", "b"]))



def test_build_one_factor_correlation_matrix_raises_for_nan_values() -> None:
    """Verifica que ``loadings`` no acepta ``NaN``."""
    with pytest.raises(ValueError, match="loadings must contain finite values"):
        build_one_factor_correlation_matrix(np.array([0.5, np.nan]))



def test_build_one_factor_correlation_matrix_raises_for_infinite_values() -> None:
    """Verifica que ``loadings`` no acepta infinitos."""
    with pytest.raises(ValueError, match="loadings must contain finite values"):
        build_one_factor_correlation_matrix(np.array([0.5, np.inf]))



def test_build_one_factor_correlation_matrix_raises_for_loading_above_one() -> None:
    """Verifica que ``abs(beta_i) > 1`` falla."""
    with pytest.raises(ValueError, match=r"loadings must satisfy abs\(value\) <= 1"):
        build_one_factor_correlation_matrix(np.array([1.2, 0.3]))


def test_build_nonlinear_redundancy_groups_returns_normalized_structure() -> None:
    """Verifica que la funcion devuelve el diccionario esperado."""
    result = build_nonlinear_redundancy_groups([[0, 1], [2, 3]], n_assets=4, strength=0.8)

    assert isinstance(result, dict)
    assert set(result) == {"groups", "n_assets", "strength"}
    assert result["groups"] == [[0, 1], [2, 3]]
    assert result["n_assets"] == 4
    assert result["strength"] == 0.8



def test_build_nonlinear_redundancy_groups_accepts_single_group() -> None:
    """Verifica que se acepta un unico grupo."""
    result = build_nonlinear_redundancy_groups([[1, 3]], n_assets=4, strength=1.0)

    assert result == {
        "groups": [[1, 3]],
        "n_assets": 4,
        "strength": 1.0,
    }



def test_build_nonlinear_redundancy_groups_allows_partial_grouping() -> None:
    """Verifica que no es necesario agrupar todos los activos."""
    result = build_nonlinear_redundancy_groups([[0, 2]], n_assets=4, strength=0.5)

    assert result == {
        "groups": [[0, 2]],
        "n_assets": 4,
        "strength": 0.5,
    }



def test_build_nonlinear_redundancy_groups_raises_for_non_list_groups() -> None:
    """Verifica que ``groups`` debe ser una lista."""
    with pytest.raises(TypeError, match="groups must be a list"):
        build_nonlinear_redundancy_groups(([0, 1], [2, 3]), n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_non_list_group() -> None:
    """Verifica que cada grupo debe ser una lista."""
    with pytest.raises(TypeError, match="each group must be a list"):
        build_nonlinear_redundancy_groups([[0, 1], (2, 3)], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_empty_group() -> None:
    """Verifica que no se aceptan grupos vacios."""
    with pytest.raises(ValueError, match="groups must not contain empty lists"):
        build_nonlinear_redundancy_groups([[0, 1], []], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_non_integer_index() -> None:
    """Verifica que los indices deben ser enteros estrictos."""
    with pytest.raises(TypeError, match="group indices must be integers"):
        build_nonlinear_redundancy_groups([[0, 1.5]], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_boolean_index() -> None:
    """Verifica que los indices no aceptan booleanos."""
    with pytest.raises(TypeError, match="group indices must be integers"):
        build_nonlinear_redundancy_groups([[0, True]], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_negative_index() -> None:
    """Verifica que los indices negativos fallan."""
    with pytest.raises(ValueError, match="group indices must be within range"):
        build_nonlinear_redundancy_groups([[0, -1]], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_out_of_range_index() -> None:
    """Verifica que los indices fuera de rango fallan."""
    with pytest.raises(ValueError, match="group indices must be within range"):
        build_nonlinear_redundancy_groups([[0, 4]], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_repeated_index_across_groups() -> None:
    """Verifica que un indice no puede repetirse en grupos distintos."""
    with pytest.raises(ValueError, match="group indices must be unique across groups"):
        build_nonlinear_redundancy_groups([[0, 1], [1, 2]], n_assets=4, strength=0.8)



def test_build_nonlinear_redundancy_groups_raises_for_invalid_n_assets() -> None:
    """Verifica que ``n_assets=0`` falla."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        build_nonlinear_redundancy_groups([[0, 1]], n_assets=0, strength=0.8)



@pytest.mark.parametrize("strength", ["x", True])
def test_build_nonlinear_redundancy_groups_raises_for_invalid_strength_type(
    strength: object,
) -> None:
    """Verifica que ``strength`` debe ser numerico y no booleano."""
    with pytest.raises(TypeError, match="strength must be a number"):
        build_nonlinear_redundancy_groups([[0, 1]], n_assets=4, strength=strength)



def test_build_nonlinear_redundancy_groups_raises_for_negative_strength() -> None:
    """Verifica que ``strength < 0`` falla."""
    with pytest.raises(ValueError, match="strength must be between 0 and 1"):
        build_nonlinear_redundancy_groups([[0, 1]], n_assets=4, strength=-0.1)



def test_build_nonlinear_redundancy_groups_raises_for_strength_above_one() -> None:
    """Verifica que ``strength > 1`` falla."""
    with pytest.raises(ValueError, match="strength must be between 0 and 1"):
        build_nonlinear_redundancy_groups([[0, 1]], n_assets=4, strength=1.1)






