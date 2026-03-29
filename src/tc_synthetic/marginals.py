"""Modulo de marginales del toolbox de datos sinteticos."""

import numpy as np
from scipy.stats import norm, skewnorm, t

from tc_synthetic.specs import MarginalSpec
from tc_synthetic.utils import validate_n_obs

__all__ = [
    "standardize_1d_sample",
    "apply_standard_normal_inverse_cdf",
    "apply_standardized_student_t_inverse_cdf",
    "apply_standardized_skew_normal_inverse_cdf",
    "sample_standard_normal_marginal",
    "sample_standardized_student_t_marginal",
    "sample_standardized_skew_normal_marginal",
    "sample_heterogeneous_marginals",
]



def standardize_1d_sample(sample: np.ndarray) -> np.ndarray:
    """Estandariza una muestra univariante usando media muestral y desvio poblacional."""
    if not isinstance(sample, np.ndarray):
        raise TypeError("sample must be a numpy.ndarray")
    if sample.ndim != 1:
        raise ValueError("sample must be a 1D array")
    if sample.size == 0:
        raise ValueError("sample must not be empty")
    if np.issubdtype(sample.dtype, np.bool_):
        raise TypeError("sample must not contain booleans")
    if not np.issubdtype(sample.dtype, np.number):
        raise TypeError("sample must contain numeric values")

    values = sample.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("sample must contain finite values")

    mean = np.mean(values)
    std = np.std(values, ddof=0)
    if std == 0.0:
        raise ValueError("sample must have non-zero standard deviation")

    return (values - mean) / std



def apply_standard_normal_inverse_cdf(u: np.ndarray) -> np.ndarray:
    """Aplica la inversa de la CDF normal estandar a un array de uniforms."""
    if not isinstance(u, np.ndarray):
        raise TypeError("u must be a numpy.ndarray")
    if u.ndim not in (1, 2):
        raise ValueError("u must be a 1D or 2D array")
    if np.issubdtype(u.dtype, np.bool_):
        raise TypeError("u must not contain booleans")
    if not np.issubdtype(u.dtype, np.number):
        raise TypeError("u must contain numeric values")

    values = u.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("u must contain finite values")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("u values must be strictly between 0 and 1")

    return norm.ppf(values)



def apply_standardized_student_t_inverse_cdf(
    u: np.ndarray,
    df: float,
) -> np.ndarray:
    """Aplica la inversa de la CDF Student-t estandarizada a un array de uniforms."""
    if not isinstance(u, np.ndarray):
        raise TypeError("u must be a numpy.ndarray")
    if u.ndim not in (1, 2):
        raise ValueError("u must be a 1D or 2D array")
    if np.issubdtype(u.dtype, np.bool_):
        raise TypeError("u must not contain booleans")
    if not np.issubdtype(u.dtype, np.number):
        raise TypeError("u must contain numeric values")

    values = u.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("u must contain finite values")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("u values must be strictly between 0 and 1")
    if isinstance(df, bool) or not isinstance(df, (int, float)):
        raise TypeError("df must be a number")
    if not np.isfinite(df):
        raise ValueError("df must be finite")
    if df <= 2:
        raise ValueError("df must be greater than 2")

    scale = np.sqrt((float(df) - 2.0) / float(df))
    return t.ppf(values, df) * scale



def apply_standardized_skew_normal_inverse_cdf(
    u: np.ndarray,
    shape: float,
) -> np.ndarray:
    """Aplica la inversa de la CDF skew-normal estandarizada a un array de uniforms."""
    if not isinstance(u, np.ndarray):
        raise TypeError("u must be a numpy.ndarray")
    if u.ndim not in (1, 2):
        raise ValueError("u must be a 1D or 2D array")
    if np.issubdtype(u.dtype, np.bool_):
        raise TypeError("u must not contain booleans")
    if not np.issubdtype(u.dtype, np.number):
        raise TypeError("u must contain numeric values")

    values = u.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("u must contain finite values")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("u values must be strictly between 0 and 1")
    if isinstance(shape, bool) or not isinstance(shape, (int, float)):
        raise TypeError("shape must be a number")
    if not np.isfinite(shape):
        raise ValueError("shape must be finite")

    shape = float(shape)
    raw = skewnorm.ppf(values, a=shape, loc=0, scale=1)
    delta = shape / np.sqrt(1.0 + shape**2)
    mu = delta * np.sqrt(2.0 / np.pi)
    sigma = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    return (raw - mu) / sigma



def sample_standard_normal_marginal(
    n_obs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una muestra univariante desde una marginal normal estandar."""
    n_obs = validate_n_obs(n_obs)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng.standard_normal(size=n_obs)



def sample_standardized_student_t_marginal(
    n_obs: int,
    df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una muestra Student-t estandarizada con varianza unitaria."""
    n_obs = validate_n_obs(n_obs)
    if isinstance(df, bool) or not isinstance(df, (int, float)):
        raise TypeError("df must be a number")
    if df <= 2:
        raise ValueError("df must be greater than 2")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    scale = np.sqrt((float(df) - 2.0) / float(df))
    return rng.standard_t(df, size=n_obs) * scale



def sample_standardized_skew_normal_marginal(
    n_obs: int,
    shape: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una muestra skew-normal estandarizada con media cero y varianza unitaria."""
    n_obs = validate_n_obs(n_obs)
    if isinstance(shape, bool) or not isinstance(shape, (int, float)):
        raise TypeError("shape must be a number")
    if not np.isfinite(shape):
        raise ValueError("shape must be finite")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    shape = float(shape)
    raw_sample = skewnorm.rvs(a=shape, loc=0, scale=1, size=n_obs, random_state=rng)
    delta = shape / np.sqrt(1.0 + shape**2)
    mu = delta * np.sqrt(2.0 / np.pi)
    sigma = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    return (raw_sample - mu) / sigma



def sample_heterogeneous_marginals(
    n_obs: int,
    specs: list[MarginalSpec],
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una matriz de marginales heterogeneas, una columna por activo."""
    n_obs = validate_n_obs(n_obs)
    if not isinstance(specs, list):
        raise TypeError("specs must be a list")
    if not specs:
        raise ValueError("specs must not be empty")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    columns: list[np.ndarray] = []
    for spec in specs:
        if not isinstance(spec, MarginalSpec):
            raise TypeError("specs must contain MarginalSpec instances")

        if spec.kind == "gaussian":
            column = sample_standard_normal_marginal(n_obs, rng)
        elif spec.kind == "student_t":
            if "df" not in spec.params:
                raise ValueError("student_t spec requires 'df'")
            column = sample_standardized_student_t_marginal(n_obs, spec.params["df"], rng)
        elif spec.kind == "skew_normal":
            if "shape" not in spec.params:
                raise ValueError("skew_normal spec requires 'shape'")
            column = sample_standardized_skew_normal_marginal(n_obs, spec.params["shape"], rng)
        elif spec.kind == "heterogeneous":
            raise ValueError("heterogeneous specs are not supported here")
        else:
            raise ValueError(f"unsupported marginal kind: {spec.kind}")

        columns.append(column)

    return np.column_stack(columns).astype(float, copy=False)







