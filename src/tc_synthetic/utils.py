"""Modulo de utilidades transversales para el toolbox de datos sinteticos."""

import numpy as np

__all__ = [
    "make_rng",
    "validate_n_assets",
    "validate_n_obs",
    "validate_square_matrix",
    "validate_symmetric_matrix",
    "validate_unit_diagonal",
    "validate_positive_semidefinite_matrix",
]


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Crea un generador aleatorio reproducible a partir de una semilla opcional."""
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    return np.random.default_rng(seed)



def validate_n_assets(n_assets: int) -> int:
    """Valida que el numero de activos sea un entero positivo."""
    if isinstance(n_assets, bool) or not isinstance(n_assets, int):
        raise TypeError("n_assets must be an integer")
    if n_assets <= 0:
        raise ValueError("n_assets must be greater than 0")
    return n_assets



def validate_n_obs(n_obs: int) -> int:
    """Valida que el numero de observaciones sea un entero positivo."""
    if isinstance(n_obs, bool) or not isinstance(n_obs, int):
        raise TypeError("n_obs must be an integer")
    if n_obs <= 0:
        raise ValueError("n_obs must be greater than 0")
    return n_obs



def validate_square_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Valida que una matriz sea un ``ndarray`` bidimensional y cuadrado."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"{name} must be a square matrix")
    return matrix



def _validate_atol(atol: float) -> float:
    """Valida que la tolerancia absoluta sea numerica y no negativa."""
    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        raise TypeError("atol must be a non-negative number")
    if atol < 0:
        raise ValueError("atol must be non-negative")
    return float(atol)



def validate_symmetric_matrix(
    matrix: np.ndarray,
    name: str = "matrix",
    atol: float = 1e-8,
) -> np.ndarray:
    """Valida que una matriz cuadrada sea simetrica dentro de una tolerancia."""
    validate_square_matrix(matrix, name=name)
    atol = _validate_atol(atol)
    if not np.allclose(matrix, matrix.T, atol=atol, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return matrix



def validate_unit_diagonal(
    matrix: np.ndarray,
    name: str = "matrix",
    atol: float = 1e-8,
) -> np.ndarray:
    """Valida que una matriz cuadrada tenga diagonal unitaria aproximada."""
    validate_square_matrix(matrix, name=name)
    atol = _validate_atol(atol)
    if not np.allclose(np.diag(matrix), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{name} must have a unit diagonal")
    return matrix



def validate_positive_semidefinite_matrix(
    matrix: np.ndarray,
    name: str = "matrix",
    atol: float = 1e-8,
) -> np.ndarray:
    """Valida que una matriz cuadrada y simetrica sea positiva semidefinida."""
    validate_square_matrix(matrix, name=name)
    atol = _validate_atol(atol)
    validate_symmetric_matrix(matrix, name=name, atol=atol)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues < -atol):
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix
