"""Modulo de estimacion de magnitudes de informacion con RBIG."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Literal

import numpy as np
from scipy.stats import norm

__all__ = [
    "estimate_rbig_total_correlation",
    "estimate_rbig_joint_entropy",
    "estimate_rbig_mutual_information",
    "compute_gaussianized_correlation",
    "compute_gaussian_total_correlation",
    "compute_excess_total_correlation",
]


def _validate_2d_array(x: np.ndarray, name: str) -> None:
    """Valida un array bidimensional no vacio."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must have positive shape in both dimensions")


def _convert_information_units(value: float, units: Literal["nats", "bits"]) -> float:
    """Convierte una magnitud de informacion a las unidades pedidas."""
    if units == "nats":
        return float(value)
    if units == "bits":
        return float(value) / float(np.log(2.0))
    raise ValueError("units must be either 'nats' or 'bits'")


def estimate_rbig_total_correlation(
    x: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> float:
    """Estima la correlacion total usando RBIG."""
    _validate_2d_array(x, "x")

    try:
        from rbig import rbig_total_corr
    except ImportError as error:
        raise ImportError(
            "RBIG is required for information diagnostics. "
            "Install it with: pip install git+https://github.com/ipl-uv/rbig.git"
        ) from error

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        tc = rbig_total_corr(
            X=x,
            rotation=rotation,
            max_layers=max_layers,
        )

    return _convert_information_units(float(tc), units)


def estimate_rbig_joint_entropy(
    x: np.ndarray,
    *,
    bins: str = "auto",
    correction: bool = True,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> float:
    """Estima la entropia conjunta usando entropias marginales y TC de RBIG."""
    _validate_2d_array(x, "x")

    try:
        from rbig import entropy_marginal, rbig_total_corr
    except ImportError as error:
        raise ImportError(
            "RBIG is required for information diagnostics. "
            "Install it with: pip install git+https://github.com/ipl-uv/rbig.git"
        ) from error

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        h_x = entropy_marginal(x, bins=bins, correction=correction)
        tc = rbig_total_corr(
            X=x,
            rotation=rotation,
            max_layers=max_layers,
        )

    h_joint = float(np.sum(h_x) - float(tc))
    return _convert_information_units(h_joint, units)


def estimate_rbig_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> float:
    """Estima la informacion mutua entre dos bloques de variables con RBIG."""
    _validate_2d_array(x, "x")
    _validate_2d_array(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations")

    try:
        from rbig._src.mutual_info import MutualInfoRBIG
    except ImportError as error:
        raise ImportError(
            "RBIG is required for information diagnostics. "
            "Install it with: pip install git+https://github.com/ipl-uv/rbig.git"
        ) from error

    model = MutualInfoRBIG(
        max_layers=max_layers,
        rotation=rotation,
    )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        model.fit(x, y)
        mi = model.mutual_info()

    return _convert_information_units(float(mi), units)


def compute_gaussianized_correlation(x: np.ndarray) -> np.ndarray:
    """Calcula la correlacion tras gaussianizar cada marginal por ranks."""
    _validate_2d_array(x, "x")

    n_obs, n_features = x.shape
    gaussianized_columns: list[np.ndarray] = []
    for column_index in range(n_features):
        order = np.argsort(x[:, column_index], kind="mergesort")
        ranks = np.empty(n_obs, dtype=float)
        ranks[order] = np.arange(1, n_obs + 1, dtype=float)
        uniforms = (ranks - 0.5) / n_obs
        gaussianized_columns.append(norm.ppf(uniforms))

    z = np.column_stack(gaussianized_columns).astype(float, copy=False)
    return np.atleast_2d(np.corrcoef(z, rowvar=False)).astype(float, copy=False)


def compute_gaussian_total_correlation(
    x: np.ndarray,
    *,
    regularization: float = 1e-10,
    units: Literal["nats", "bits"] = "nats",
) -> float:
    """Calcula la correlacion total gaussiana a partir de la correlacion gaussianizada."""
    _validate_2d_array(x, "x")

    correlation = compute_gaussianized_correlation(x)
    correlation_reg = correlation + regularization * np.eye(correlation.shape[0], dtype=float)
    sign, logdet = np.linalg.slogdet(correlation_reg)
    if sign <= 0:
        raise ValueError("correlation matrix is not positive definite")

    tc_gauss = -0.5 * float(logdet)
    return _convert_information_units(tc_gauss, units)


def compute_excess_total_correlation(
    x: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    regularization: float = 1e-10,
    units: Literal["nats", "bits"] = "nats",
) -> float:
    """Calcula el exceso de correlacion total respecto al caso gaussiano equivalente."""
    _validate_2d_array(x, "x")

    tc = estimate_rbig_total_correlation(
        x,
        max_layers=max_layers,
        rotation=rotation,
        units=units,
    )
    tc_gauss = compute_gaussian_total_correlation(
        x,
        regularization=regularization,
        units=units,
    )
    return float(tc - tc_gauss)
