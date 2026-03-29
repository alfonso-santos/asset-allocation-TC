"""Pruebas funcionales para los estados de ``tc_synthetic``."""

import numpy as np
import pytest

from tc_synthetic.states import (
    build_two_state_markov_transition_matrix,
    sample_markov_states,
)


N_OBS = 100_000



def _empirical_transition_matrix(states: np.ndarray, n_states: int = 2) -> np.ndarray:
    """Construye la matriz de transicion empirica a partir de una trayectoria."""
    counts = np.zeros((n_states, n_states), dtype=float)
    for previous_state, next_state in zip(states[:-1], states[1:]):
        counts[int(previous_state), int(next_state)] += 1.0

    row_sums = np.sum(counts, axis=1, keepdims=True)
    assert np.all(row_sums > 0.0)
    return counts / row_sums



def _state_frequencies(states: np.ndarray, n_states: int = 2) -> np.ndarray:
    """Calcula las frecuencias empiricas de permanencia por estado."""
    return np.bincount(states.astype(int), minlength=n_states).astype(float) / states.size



def test_sampled_states_have_valid_shape_dtype_and_support() -> None:
    """Verifica shape, dtype y soporte binario del muestreo de estados."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])

    states = sample_markov_states(N_OBS, transition_matrix, 0, rng)

    assert states.shape == (N_OBS,)
    assert np.issubdtype(states.dtype, np.integer)
    assert set(np.unique(states)).issubset({0, 1})



def test_empirical_transition_matrix_is_close_to_target() -> None:
    """Verifica que la dinamica empirica aproxima bien la matriz objetivo."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])

    states = sample_markov_states(N_OBS, transition_matrix, 0, rng)
    empirical = _empirical_transition_matrix(states)

    assert np.allclose(empirical, transition_matrix, atol=0.02)



def test_persistent_regime_has_more_self_transitions_than_switches() -> None:
    """Verifica que un regimen persistente produce mas permanencias que cambios."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.97, 0.03], [0.08, 0.92]])

    states = sample_markov_states(N_OBS, transition_matrix, 0, rng)
    empirical = _empirical_transition_matrix(states)

    assert empirical[0, 0] > empirical[0, 1]
    assert empirical[1, 1] > empirical[1, 0]



def test_empirical_state_frequencies_are_close_to_stationary_distribution() -> None:
    """Verifica que las frecuencias empiricas se aproximan a la distribucion estacionaria."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.95, 0.05], [0.20, 0.80]])
    a = transition_matrix[0, 1]
    b = transition_matrix[1, 0]
    stationary = np.array([b / (a + b), a / (a + b)])

    states = sample_markov_states(N_OBS, transition_matrix, 0, rng)
    empirical_freq = _state_frequencies(states)

    assert np.allclose(empirical_freq, stationary, atol=0.02)



def test_starting_from_state_zero_preserves_calm_convention_at_t0() -> None:
    """Verifica que ``initial_state=0`` se conserva en la primera observacion."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])

    states = sample_markov_states(N_OBS, transition_matrix, 0, rng)

    assert states[0] == 0



def test_starting_from_state_one_preserves_crisis_convention_at_t0() -> None:
    """Verifica que ``initial_state=1`` se conserva en la primera observacion."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])

    states = sample_markov_states(N_OBS, transition_matrix, 1, rng)

    assert states[0] == 1



def test_builder_for_two_state_transition_matrix_creates_intended_matrix() -> None:
    """Verifica que el builder calm/crisis construye la matriz esperada."""
    result = build_two_state_markov_transition_matrix(0.9, 0.8)
    expected = np.array([[0.9, 0.1], [0.2, 0.8]])

    assert np.allclose(result, expected)



def test_no_nan_or_inf_in_sampled_states_or_empirical_transitions() -> None:
    """Verifica finitud total en estados muestreados y transiciones empiricas."""
    rng = np.random.default_rng(123)
    transition_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])

    states = sample_markov_states(N_OBS, transition_matrix, 0, rng)
    empirical = _empirical_transition_matrix(states)

    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(empirical))

