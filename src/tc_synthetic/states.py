"""Modulo de estados del toolbox de datos sinteticos."""

import numpy as np

from tc_synthetic.utils import validate_n_obs

__all__ = [
    "validate_transition_matrix",
    "sample_markov_states",
    "build_two_state_markov_transition_matrix",
]



def validate_transition_matrix(matrix: np.ndarray) -> np.ndarray:
    """Valida una matriz de transicion de Markov de 2 estados."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")
    if matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    if np.issubdtype(matrix.dtype, np.bool_):
        raise TypeError("matrix must not contain booleans")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("matrix must contain numeric values")

    values = matrix.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("matrix values must be between 0 and 1")
    if not np.allclose(np.sum(values, axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("matrix rows must sum to 1")

    return values



def sample_markov_states(
    n_obs: int,
    transition_matrix: np.ndarray,
    initial_state: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Muestrea una trayectoria de una cadena de Markov de 2 estados."""
    n_obs = validate_n_obs(n_obs)
    transition = validate_transition_matrix(transition_matrix)
    if isinstance(initial_state, bool) or not isinstance(initial_state, int):
        raise TypeError("initial_state must be an integer")
    if initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 or 1")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    states = np.empty(n_obs, dtype=int)
    states[0] = initial_state
    for t in range(1, n_obs):
        previous_state = states[t - 1]
        states[t] = rng.choice([0, 1], p=transition[previous_state])
    return states



def _validate_probability(value: float, name: str) -> float:
    """Valida una probabilidad escalar en el intervalo unitario."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value



def build_two_state_markov_transition_matrix(
    p_stay_calm: float,
    p_stay_crisis: float,
) -> np.ndarray:
    """Construye una matriz de transicion calm/crisis para una cadena de 2 estados."""
    p_stay_calm = _validate_probability(p_stay_calm, name="p_stay_calm")
    p_stay_crisis = _validate_probability(p_stay_crisis, name="p_stay_crisis")

    matrix = np.array(
        [
            [p_stay_calm, 1.0 - p_stay_calm],
            [1.0 - p_stay_crisis, p_stay_crisis],
        ],
        dtype=float,
    )
    return validate_transition_matrix(matrix)
