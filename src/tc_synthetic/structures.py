"""Modulo de estructuras del toolbox de datos sinteticos."""

import numpy as np

from tc_synthetic.utils import (
    validate_n_assets,
    validate_positive_semidefinite_matrix,
    validate_square_matrix,
    validate_symmetric_matrix,
    validate_unit_diagonal,
)

__all__ = [
    "build_equicorrelation_matrix",
    "build_block_correlation_matrix",
    "build_near_duplicate_correlation_matrix",
    "build_nonlinear_redundancy_groups",
    "build_one_factor_correlation_matrix",
    "build_factor_correlation_matrix",
]


def build_equicorrelation_matrix(n_assets: int, rho: float) -> np.ndarray:
    """Construye una matriz de correlacion equicorrelacionada."""
    n_assets = validate_n_assets(n_assets)
    if isinstance(rho, bool) or not isinstance(rho, (int, float)):
        raise TypeError("rho must be a number")
    if rho > 1:
        raise ValueError("rho must be less than or equal to 1")
    if n_assets > 1:
        lower_bound = -1 / (n_assets - 1)
        if rho < lower_bound:
            raise ValueError(f"rho must be greater than or equal to {lower_bound}")

    matrix = np.full((n_assets, n_assets), float(rho), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return matrix



def build_block_correlation_matrix(
    block_sizes: list[int],
    rho_within: float,
    rho_between: float,
) -> np.ndarray:
    """Construye una matriz de correlacion por bloques con correlaciones constantes."""
    if not isinstance(block_sizes, list):
        raise TypeError("block_sizes must be a list")
    if not block_sizes:
        raise ValueError("block_sizes must not be empty")
    for block_size in block_sizes:
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("block_sizes must contain integers")
        if block_size <= 0:
            raise ValueError("block_sizes must contain positive integers")
    if isinstance(rho_within, bool) or not isinstance(rho_within, (int, float)):
        raise TypeError("rho_within must be a number")
    if isinstance(rho_between, bool) or not isinstance(rho_between, (int, float)):
        raise TypeError("rho_between must be a number")
    if rho_within > 1:
        raise ValueError("rho_within must be less than or equal to 1")
    if rho_between > 1:
        raise ValueError("rho_between must be less than or equal to 1")

    n_assets = sum(block_sizes)
    matrix = np.full((n_assets, n_assets), float(rho_between), dtype=float)
    start = 0
    for block_size in block_sizes:
        stop = start + block_size
        matrix[start:stop, start:stop] = float(rho_within)
        start = stop

    np.fill_diagonal(matrix, 1.0)
    validate_square_matrix(matrix, name="correlation")
    validate_symmetric_matrix(matrix, name="correlation")
    validate_unit_diagonal(matrix, name="correlation")
    validate_positive_semidefinite_matrix(matrix, name="correlation")
    return matrix



def build_near_duplicate_correlation_matrix(
    group_sizes: list[int],
    rho_duplicate: float,
    rho_background: float = 0.0,
) -> np.ndarray:
    """Construye una matriz de correlacion para grupos de activos casi duplicados."""
    if isinstance(rho_duplicate, bool) or not isinstance(rho_duplicate, (int, float)):
        raise TypeError("rho_duplicate must be a number")
    if isinstance(rho_background, bool) or not isinstance(rho_background, (int, float)):
        raise TypeError("rho_background must be a number")
    if rho_duplicate > 1:
        raise ValueError("rho_duplicate must be less than or equal to 1")
    if rho_background > 1:
        raise ValueError("rho_background must be less than or equal to 1")
    if rho_duplicate < rho_background:
        raise ValueError("rho_duplicate must be greater than or equal to rho_background")
    return build_block_correlation_matrix(group_sizes, rho_duplicate, rho_background)



def build_nonlinear_redundancy_groups(
    groups: list[list[int]],
    n_assets: int,
    strength: float,
) -> dict[str, object]:
    """Valida y normaliza grupos de redundancia no lineal potencial."""
    n_assets = validate_n_assets(n_assets)
    if not isinstance(groups, list):
        raise TypeError("groups must be a list")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError("strength must be a number")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be between 0 and 1")

    normalized_groups: list[list[int]] = []
    seen_indices: set[int] = set()
    for group in groups:
        if not isinstance(group, list):
            raise TypeError("each group must be a list")
        if not group:
            raise ValueError("groups must not contain empty lists")

        normalized_group: list[int] = []
        for index in group:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("group indices must be integers")
            if index < 0 or index >= n_assets:
                raise ValueError("group indices must be within range")
            if index in seen_indices:
                raise ValueError("group indices must be unique across groups")
            normalized_group.append(index)
            seen_indices.add(index)

        normalized_groups.append(normalized_group)

    return {
        "groups": normalized_groups,
        "n_assets": n_assets,
        "strength": float(strength),
    }


def build_factor_correlation_matrix(loadings: np.ndarray) -> np.ndarray:
    """Construye una matriz de correlacion multifactor a partir de una matriz de loadings."""
    if not isinstance(loadings, np.ndarray):
        raise TypeError("loadings must be a numpy.ndarray")
    if loadings.ndim != 2:
        raise ValueError("loadings must be a 2D array")
    if loadings.shape[0] == 0:
        raise ValueError("loadings must have at least one row")
    if loadings.shape[1] == 0:
        raise ValueError("loadings must have at least one column")
    if np.issubdtype(loadings.dtype, np.bool_):
        raise TypeError("loadings must not contain booleans")
    if not np.issubdtype(loadings.dtype, np.number):
        raise TypeError("loadings must contain numeric values")

    beta = loadings.astype(float, copy=False)
    if not np.all(np.isfinite(beta)):
        raise ValueError("loadings must contain finite values")

    row_norms_sq = np.sum(beta**2, axis=1)
    if np.any(row_norms_sq > 1.0):
        raise ValueError("each row must satisfy sum of squares <= 1")

    common = beta @ beta.T
    unique = np.diag(1.0 - row_norms_sq)
    matrix = common + unique
    validate_square_matrix(matrix, name="correlation")
    validate_symmetric_matrix(matrix, name="correlation")
    validate_unit_diagonal(matrix, name="correlation")
    validate_positive_semidefinite_matrix(matrix, name="correlation")
    return matrix



def build_one_factor_correlation_matrix(loadings: np.ndarray) -> np.ndarray:
    """Construye una matriz de correlacion de un factor a partir de loadings."""
    if not isinstance(loadings, np.ndarray):
        raise TypeError("loadings must be a numpy.ndarray")
    if loadings.ndim != 1:
        raise ValueError("loadings must be a 1D array")
    if loadings.size == 0:
        raise ValueError("loadings must not be empty")
    if np.issubdtype(loadings.dtype, np.number) and not np.issubdtype(loadings.dtype, np.bool_):
        beta = loadings.astype(float, copy=False)
        if np.all(np.isfinite(beta)) and np.any(np.abs(beta) > 1.0):
            raise ValueError("loadings must satisfy abs(value) <= 1")
    return build_factor_correlation_matrix(loadings.reshape(-1, 1))


