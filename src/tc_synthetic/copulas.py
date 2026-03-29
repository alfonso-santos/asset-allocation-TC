"""Modulo de copulas del toolbox de datos sinteticos."""

import numpy as np
from scipy.stats import norm, t

from tc_synthetic.utils import (
    validate_n_assets,
    validate_n_obs,
    validate_positive_semidefinite_matrix,
    validate_square_matrix,
    validate_symmetric_matrix,
    validate_unit_diagonal,
)

__all__ = [
    "gaussian_latent_to_uniform",
    "sample_independence_copula",
    "sample_gaussian_copula",
    "sample_t_copula",
    "sample_grouped_t_copula",
    "sample_clayton_copula",
]



def gaussian_latent_to_uniform(latent: np.ndarray) -> np.ndarray:
    """Transforma variables latentes gaussianas en uniforms usando la CDF normal."""
    if not isinstance(latent, np.ndarray):
        raise TypeError("latent must be a numpy.ndarray")
    if latent.ndim != 2:
        raise ValueError("latent must be a 2D array")
    if latent.shape[0] == 0:
        raise ValueError("latent must have at least one row")
    if latent.shape[1] == 0:
        raise ValueError("latent must have at least one column")
    if np.issubdtype(latent.dtype, np.bool_):
        raise TypeError("latent must not contain booleans")
    if not np.issubdtype(latent.dtype, np.number):
        raise TypeError("latent must contain numeric values")

    values = latent.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("latent must contain finite values")

    return norm.cdf(values)



def _validate_correlation_matrix(correlation: np.ndarray) -> np.ndarray:
    """Valida y normaliza una matriz de correlacion."""
    if not isinstance(correlation, np.ndarray):
        raise TypeError("correlation must be a numpy.ndarray")
    validate_square_matrix(correlation, name="correlation")
    if np.issubdtype(correlation.dtype, np.bool_):
        raise TypeError("correlation must not contain booleans")
    if not np.issubdtype(correlation.dtype, np.number):
        raise TypeError("correlation must contain numeric values")

    values = correlation.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("correlation must contain finite values")

    validate_square_matrix(values, name="correlation")
    validate_symmetric_matrix(values, name="correlation")
    validate_unit_diagonal(values, name="correlation")
    validate_positive_semidefinite_matrix(values, name="correlation")
    validate_n_assets(values.shape[0])
    return values



def _validate_positive_df(df: float, name: str = "df") -> float:
    """Valida que un grado de libertad sea numerico, finito y positivo."""
    if isinstance(df, bool) or not isinstance(df, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not np.isfinite(df):
        raise ValueError(f"{name} must be finite")
    df = float(df)
    if df <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return df



def sample_independence_copula(
    n_obs: int,
    n_assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera uniforms i.i.d. para una copula de independencia."""
    n_obs = validate_n_obs(n_obs)
    n_assets = validate_n_assets(n_assets)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng.random(size=(n_obs, n_assets))



def sample_gaussian_copula(
    n_obs: int,
    correlation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera uniforms dependientes a partir de una Gaussian copula."""
    n_obs = validate_n_obs(n_obs)
    values = _validate_correlation_matrix(correlation)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    n_assets = values.shape[0]
    latent = rng.multivariate_normal(
        mean=np.zeros(n_assets, dtype=float),
        cov=values,
        size=n_obs,
    )
    return gaussian_latent_to_uniform(latent)



def sample_t_copula(
    n_obs: int,
    correlation: np.ndarray,
    df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera uniforms dependientes a partir de una t copula."""
    n_obs = validate_n_obs(n_obs)
    values = _validate_correlation_matrix(correlation)
    df = _validate_positive_df(df, name="df")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    n_assets = values.shape[0]
    latent = rng.multivariate_normal(
        mean=np.zeros(n_assets, dtype=float),
        cov=values,
        size=n_obs,
    )
    chi2 = rng.chisquare(df, size=n_obs)
    t_latent = latent / np.sqrt(chi2 / df)[:, None]
    return t.cdf(t_latent, df=df)



def sample_grouped_t_copula(
    n_obs: int,
    correlation: np.ndarray,
    group_assignments: np.ndarray,
    group_dfs: dict[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera uniforms dependientes a partir de una grouped t copula."""
    n_obs = validate_n_obs(n_obs)
    values = _validate_correlation_matrix(correlation)
    n_assets = values.shape[0]

    if not isinstance(group_assignments, np.ndarray):
        raise TypeError("group_assignments must be a numpy.ndarray")
    if group_assignments.ndim != 1:
        raise ValueError("group_assignments must be a 1D array")
    if group_assignments.shape[0] != n_assets:
        raise ValueError("group_assignments length must match n_assets")
    if np.issubdtype(group_assignments.dtype, np.bool_):
        raise TypeError("group_assignments must contain integers")
    if not np.issubdtype(group_assignments.dtype, np.integer):
        raise TypeError("group_assignments must contain integers")

    if not isinstance(group_dfs, dict):
        raise TypeError("group_dfs must be a dict")
    if not group_dfs:
        raise ValueError("group_dfs must not be empty")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    assignments = group_assignments.astype(int, copy=False)
    present_groups = {int(group) for group in np.unique(assignments)}
    if set(group_dfs) != present_groups:
        raise ValueError("group_dfs must cover exactly the groups in group_assignments")

    validated_group_dfs: dict[int, float] = {}
    for group in present_groups:
        validated_group_dfs[group] = _validate_positive_df(group_dfs[group], name="group df")

    latent = rng.multivariate_normal(
        mean=np.zeros(n_assets, dtype=float),
        cov=values,
        size=n_obs,
    )
    uniforms = np.empty((n_obs, n_assets), dtype=float)

    for group in np.unique(assignments):
        group = int(group)
        group_mask = assignments == group
        group_df = validated_group_dfs[group]
        chi2 = rng.chisquare(group_df, size=n_obs)
        t_latent = latent[:, group_mask] / np.sqrt(chi2 / group_df)[:, None]
        uniforms[:, group_mask] = t.cdf(t_latent, df=group_df)

    return uniforms



def sample_clayton_copula(
    n_obs: int,
    n_assets: int,
    theta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera uniforms dependientes a partir de una Clayton copula."""
    n_obs = validate_n_obs(n_obs)
    n_assets = validate_n_assets(n_assets)
    theta = _validate_positive_df(theta, name="theta")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_obs)
    e = rng.exponential(scale=1.0, size=(n_obs, n_assets))
    return (1.0 + e / v[:, None]) ** (-1.0 / theta)
