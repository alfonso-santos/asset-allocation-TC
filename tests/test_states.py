"""Pruebas minimas para los estados de ``tc_synthetic``."""

import numpy as np
import pytest

from tc_synthetic.states import validate_transition_matrix



def test_validate_transition_matrix_accepts_valid_matrix() -> None:
    """Verifica que una matriz valida se acepta."""
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])

    result = validate_transition_matrix(matrix)

    assert np.array_equal(result, matrix)



def test_validate_transition_matrix_returns_float_dtype() -> None:
    """Verifica que la salida se devuelve con dtype flotante."""
    matrix = np.array([[1, 0], [0, 1]], dtype=int)

    result = validate_transition_matrix(matrix)

    assert np.issubdtype(result.dtype, np.floating)



def test_validate_transition_matrix_raises_for_non_array_input() -> None:
    """Verifica que el input debe ser un ``ndarray``."""
    with pytest.raises(TypeError, match="matrix must be a numpy.ndarray"):
        validate_transition_matrix([[0.9, 0.1], [0.2, 0.8]])



def test_validate_transition_matrix_raises_for_one_dimensional_input() -> None:
    """Verifica que la matriz debe ser bidimensional."""
    with pytest.raises(ValueError, match="matrix must be a 2D array"):
        validate_transition_matrix(np.array([0.9, 0.1]))



def test_validate_transition_matrix_raises_for_invalid_shape() -> None:
    """Verifica que la matriz debe tener shape ``(2, 2)``."""
    with pytest.raises(ValueError, match=r"matrix must have shape \(2, 2\)"):
        validate_transition_matrix(np.eye(3, dtype=float))



def test_validate_transition_matrix_raises_for_boolean_values() -> None:
    """Verifica que no se aceptan booleanos."""
    with pytest.raises(TypeError, match="matrix must not contain booleans"):
        validate_transition_matrix(np.array([[True, False], [False, True]]))



def test_validate_transition_matrix_raises_for_non_numeric_values() -> None:
    """Verifica que la matriz debe contener valores numericos."""
    with pytest.raises(TypeError, match="matrix must contain numeric values"):
        validate_transition_matrix(np.array([["a", "b"], ["c", "d"]]))



def test_validate_transition_matrix_raises_for_nan_values() -> None:
    """Verifica que la matriz no acepta ``NaN``."""
    with pytest.raises(ValueError, match="matrix must contain finite values"):
        validate_transition_matrix(np.array([[0.9, np.nan], [0.2, 0.8]]))



def test_validate_transition_matrix_raises_for_infinite_values() -> None:
    """Verifica que la matriz no acepta infinitos."""
    with pytest.raises(ValueError, match="matrix must contain finite values"):
        validate_transition_matrix(np.array([[0.9, np.inf], [0.2, 0.8]]))



def test_validate_transition_matrix_raises_for_values_below_zero() -> None:
    """Verifica que los valores deben estar entre cero y uno."""
    with pytest.raises(ValueError, match="matrix values must be between 0 and 1"):
        validate_transition_matrix(np.array([[1.1, -0.1], [0.2, 0.8]]))



def test_validate_transition_matrix_raises_for_values_above_one() -> None:
    """Verifica que los valores no pueden exceder uno."""
    with pytest.raises(ValueError, match="matrix values must be between 0 and 1"):
        validate_transition_matrix(np.array([[1.2, -0.2], [0.2, 0.8]]))



def test_validate_transition_matrix_raises_for_rows_not_summing_to_one() -> None:
    """Verifica que cada fila debe sumar uno."""
    with pytest.raises(ValueError, match="matrix rows must sum to 1"):
        validate_transition_matrix(np.array([[0.9, 0.2], [0.2, 0.8]]))

# === Added in Step 6.2 ===
from tc_synthetic.states import sample_markov_states



def test_sample_markov_states_returns_expected_shape() -> None:
    """Verifica que la trayectoria devuelve shape unidimensional correcto."""
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    rng = np.random.default_rng(123)

    result = sample_markov_states(6, transition_matrix, 0, rng)

    assert result.shape == (6,)



def test_sample_markov_states_returns_integer_dtype() -> None:
    """Verifica que la trayectoria se devuelve con dtype entero."""
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    rng = np.random.default_rng(123)

    result = sample_markov_states(6, transition_matrix, 0, rng)

    assert np.issubdtype(result.dtype, np.integer)



def test_sample_markov_states_returns_only_binary_states() -> None:
    """Verifica que la trayectoria solo contiene estados 0 y 1."""
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    rng = np.random.default_rng(123)

    result = sample_markov_states(6, transition_matrix, 0, rng)

    assert set(np.unique(result)).issubset({0, 1})



def test_sample_markov_states_starts_at_initial_state() -> None:
    """Verifica que el primer estado coincide con ``initial_state``."""
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    rng = np.random.default_rng(123)

    result = sample_markov_states(6, transition_matrix, 1, rng)

    assert result[0] == 1



def test_sample_markov_states_is_reproducible_for_same_seed() -> None:
    """Verifica que dos generadores con la misma semilla producen la misma trayectoria."""
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_markov_states(6, transition_matrix, 0, rng_a)
    sample_b = sample_markov_states(6, transition_matrix, 0, rng_b)

    assert np.array_equal(sample_a, sample_b)



def test_sample_markov_states_handles_deterministic_identity_case() -> None:
    """Verifica el caso determinista con matriz identidad."""
    transition_matrix = np.eye(2, dtype=float)
    rng = np.random.default_rng(123)

    result = sample_markov_states(6, transition_matrix, 1, rng)

    assert np.array_equal(result, np.ones(6, dtype=int))



def test_sample_markov_states_matches_manual_construction() -> None:
    """Verifica la coherencia exacta con la construccion manual."""
    n_obs = 6
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial_state = 0
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    result = sample_markov_states(n_obs, transition_matrix, initial_state, rng_a)
    expected = np.empty(n_obs, dtype=int)
    expected[0] = initial_state
    for t in range(1, n_obs):
        expected[t] = rng_b.choice([0, 1], p=transition_matrix[expected[t - 1]])

    assert np.array_equal(result, expected)



def test_sample_markov_states_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs=0`` falla."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        sample_markov_states(0, np.array([[0.9, 0.1], [0.2, 0.8]]), 0, np.random.default_rng(123))



def test_sample_markov_states_raises_for_boolean_initial_state() -> None:
    """Verifica que ``initial_state`` no acepta booleanos."""
    with pytest.raises(TypeError, match="initial_state must be an integer"):
        sample_markov_states(6, np.array([[0.9, 0.1], [0.2, 0.8]]), True, np.random.default_rng(123))



def test_sample_markov_states_raises_for_non_integer_initial_state() -> None:
    """Verifica que ``initial_state`` debe ser entero."""
    with pytest.raises(TypeError, match="initial_state must be an integer"):
        sample_markov_states(6, np.array([[0.9, 0.1], [0.2, 0.8]]), 1.5, np.random.default_rng(123))



def test_sample_markov_states_raises_for_negative_initial_state() -> None:
    """Verifica que ``initial_state=-1`` falla."""
    with pytest.raises(ValueError, match="initial_state must be 0 or 1"):
        sample_markov_states(6, np.array([[0.9, 0.1], [0.2, 0.8]]), -1, np.random.default_rng(123))



def test_sample_markov_states_raises_for_initial_state_above_one() -> None:
    """Verifica que ``initial_state=2`` falla."""
    with pytest.raises(ValueError, match="initial_state must be 0 or 1"):
        sample_markov_states(6, np.array([[0.9, 0.1], [0.2, 0.8]]), 2, np.random.default_rng(123))



@pytest.mark.parametrize("rng", [None, 123, np.random.RandomState(123)])
def test_sample_markov_states_raises_for_invalid_rng(rng: object) -> None:
    """Verifica que ``rng`` debe ser un ``Generator`` valido."""
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sample_markov_states(6, np.array([[0.9, 0.1], [0.2, 0.8]]), 0, rng)



def test_sample_markov_states_raises_for_invalid_transition_matrix() -> None:
    """Verifica que se propaga el error de una matriz de transicion invalida."""
    with pytest.raises(ValueError, match="matrix rows must sum to 1"):
        sample_markov_states(6, np.array([[0.9, 0.2], [0.2, 0.8]]), 0, np.random.default_rng(123))
# === Added in Step 6.3 ===
from tc_synthetic.states import build_two_state_markov_transition_matrix



def test_build_two_state_markov_transition_matrix_returns_expected_matrix() -> None:
    """Verifica que la matriz calm/crisis coincide con la esperada."""
    result = build_two_state_markov_transition_matrix(0.9, 0.8)

    assert np.allclose(
        result,
        np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ],
            dtype=float,
        ),
    )



def test_build_two_state_markov_transition_matrix_returns_float_dtype() -> None:
    """Verifica que la matriz se devuelve con dtype flotante."""
    result = build_two_state_markov_transition_matrix(1, 0)

    assert np.issubdtype(result.dtype, np.floating)



def test_build_two_state_markov_transition_matrix_returns_expected_shape() -> None:
    """Verifica que la matriz tiene shape (2, 2)."""
    result = build_two_state_markov_transition_matrix(0.9, 0.8)

    assert result.shape == (2, 2)



def test_build_two_state_markov_transition_matrix_returns_valid_transition_matrix() -> None:
    """Verifica que la matriz construida pasa la validacion de transicion."""
    result = build_two_state_markov_transition_matrix(0.9, 0.8)

    assert validate_transition_matrix(result) is result



def test_build_two_state_markov_transition_matrix_accepts_lower_boundary_for_calm() -> None:
    """Verifica que ``p_stay_calm=0.0`` se acepta."""
    result = build_two_state_markov_transition_matrix(0.0, 0.8)

    assert np.allclose(
        result,
        np.array(
            [
                [0.0, 1.0],
                [0.2, 0.8],
            ],
            dtype=float,
        ),
    )



def test_build_two_state_markov_transition_matrix_accepts_upper_boundary_for_calm() -> None:
    """Verifica que ``p_stay_calm=1.0`` se acepta."""
    result = build_two_state_markov_transition_matrix(1.0, 0.8)

    assert np.allclose(
        result,
        np.array(
            [
                [1.0, 0.0],
                [0.2, 0.8],
            ],
            dtype=float,
        ),
    )



def test_build_two_state_markov_transition_matrix_accepts_lower_boundary_for_crisis() -> None:
    """Verifica que ``p_stay_crisis=0.0`` se acepta."""
    result = build_two_state_markov_transition_matrix(0.9, 0.0)

    assert np.allclose(
        result,
        np.array(
            [
                [0.9, 0.1],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
    )



def test_build_two_state_markov_transition_matrix_accepts_upper_boundary_for_crisis() -> None:
    """Verifica que ``p_stay_crisis=1.0`` se acepta."""
    result = build_two_state_markov_transition_matrix(0.9, 1.0)

    assert np.allclose(
        result,
        np.array(
            [
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    )



def test_build_two_state_markov_transition_matrix_raises_for_non_numeric_calm_probability() -> None:
    """Verifica que ``p_stay_calm`` debe ser numerico."""
    with pytest.raises(TypeError, match="p_stay_calm must be a number"):
        build_two_state_markov_transition_matrix("x", 0.8)



def test_build_two_state_markov_transition_matrix_raises_for_non_numeric_crisis_probability() -> None:
    """Verifica que ``p_stay_crisis`` debe ser numerico."""
    with pytest.raises(TypeError, match="p_stay_crisis must be a number"):
        build_two_state_markov_transition_matrix(0.9, "x")



def test_build_two_state_markov_transition_matrix_raises_for_boolean_calm_probability() -> None:
    """Verifica que ``p_stay_calm`` no acepta booleanos."""
    with pytest.raises(TypeError, match="p_stay_calm must be a number"):
        build_two_state_markov_transition_matrix(True, 0.8)



def test_build_two_state_markov_transition_matrix_raises_for_boolean_crisis_probability() -> None:
    """Verifica que ``p_stay_crisis`` no acepta booleanos."""
    with pytest.raises(TypeError, match="p_stay_crisis must be a number"):
        build_two_state_markov_transition_matrix(0.9, True)



def test_build_two_state_markov_transition_matrix_raises_for_nan_calm_probability() -> None:
    """Verifica que ``p_stay_calm=np.nan`` falla."""
    with pytest.raises(ValueError, match="p_stay_calm must be finite"):
        build_two_state_markov_transition_matrix(np.nan, 0.8)



def test_build_two_state_markov_transition_matrix_raises_for_infinite_crisis_probability() -> None:
    """Verifica que ``p_stay_crisis=np.inf`` falla."""
    with pytest.raises(ValueError, match="p_stay_crisis must be finite"):
        build_two_state_markov_transition_matrix(0.9, np.inf)



def test_build_two_state_markov_transition_matrix_raises_for_calm_probability_below_zero() -> None:
    """Verifica que ``p_stay_calm < 0`` falla."""
    with pytest.raises(ValueError, match="p_stay_calm must be between 0 and 1"):
        build_two_state_markov_transition_matrix(-0.1, 0.8)



def test_build_two_state_markov_transition_matrix_raises_for_crisis_probability_below_zero() -> None:
    """Verifica que ``p_stay_crisis < 0`` falla."""
    with pytest.raises(ValueError, match="p_stay_crisis must be between 0 and 1"):
        build_two_state_markov_transition_matrix(0.9, -0.1)



def test_build_two_state_markov_transition_matrix_raises_for_calm_probability_above_one() -> None:
    """Verifica que ``p_stay_calm > 1`` falla."""
    with pytest.raises(ValueError, match="p_stay_calm must be between 0 and 1"):
        build_two_state_markov_transition_matrix(1.1, 0.8)



def test_build_two_state_markov_transition_matrix_raises_for_crisis_probability_above_one() -> None:
    """Verifica que ``p_stay_crisis > 1`` falla."""
    with pytest.raises(ValueError, match="p_stay_crisis must be between 0 and 1"):
        build_two_state_markov_transition_matrix(0.9, 1.1)

