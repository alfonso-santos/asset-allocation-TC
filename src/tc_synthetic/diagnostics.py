"""Modulo de diagnostico del toolbox de datos sinteticos."""

from typing import Literal

import numpy as np
from tc_synthetic.information import (
    estimate_rbig_joint_entropy,
    estimate_rbig_mutual_information,
    estimate_rbig_total_correlation,
)

__all__ = [
    "compute_basic_diagnostics",
    "compute_marginal_distribution_diagnostics",
    "compute_correlation_diagnostics",
    "compute_state_conditioned_diagnostics",
    "compute_information_diagnostics",
    "compute_state_conditioned_information_diagnostics",
    "compute_pairwise_mutual_information",
    "compute_pairwise_information_correlation",
    "compute_state_conditioned_mutual_information",
]


def _validate_2d_data_array(x: np.ndarray) -> None:
    """Valida un array bidimensional no vacio."""
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must have positive shape in both dimensions")


def compute_basic_diagnostics(
    x: np.ndarray,
) -> dict[str, np.ndarray | tuple[int, int] | bool]:
    """Calcula estadisticos basicos por columna para una matriz de datos."""
    _validate_2d_data_array(x)

    return {
        "shape": x.shape,
        "is_finite": bool(np.all(np.isfinite(x))),
        "column_means": np.mean(x, axis=0),
        "column_stds": np.std(x, axis=0, ddof=0),
        "column_mins": np.min(x, axis=0),
        "column_maxs": np.max(x, axis=0),
    }


def compute_marginal_distribution_diagnostics(
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula estadisticos marginales extendidos por columna."""
    _validate_2d_data_array(x)

    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    std = np.sqrt(var)
    centered = x - mean
    skew = np.mean(centered**3, axis=0) / np.where(std > 0.0, std**3, 1.0)
    kurtosis = np.mean(centered**4, axis=0) / np.where(var > 0.0, var**2, 1.0)
    p5 = np.percentile(x, 5, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p50 = np.percentile(x, 50, axis=0)
    p75 = np.percentile(x, 75, axis=0)
    p95 = np.percentile(x, 95, axis=0)
    abs_x = np.abs(x)
    tail_2sigma = np.mean(abs_x > (2.0 * std), axis=0)
    tail_3sigma = np.mean(abs_x > (3.0 * std), axis=0)

    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "tail_2sigma": tail_2sigma,
        "tail_3sigma": tail_3sigma,
    }


def compute_correlation_diagnostics(
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula diagnosticos basicos de correlacion para una matriz de datos."""
    _validate_2d_data_array(x)

    return {
        "correlation": np.corrcoef(x, rowvar=False),
        "column_means": np.mean(x, axis=0),
        "column_stds": np.std(x, axis=0, ddof=0),
    }


def compute_state_conditioned_diagnostics(
    x: np.ndarray,
    states: np.ndarray,
) -> dict[str, dict[str, np.ndarray | tuple[int, int] | bool | int]]:
    """Calcula diagnosticos por estado usando solo los estados presentes en la muestra."""
    _validate_2d_data_array(x)
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy.ndarray")
    if states.ndim != 1:
        raise ValueError("states must be a 1D array")
    if states.shape[0] == 0:
        raise ValueError("states must be non-empty")
    if x.shape[0] != states.shape[0]:
        raise ValueError("x and states must have the same number of observations")

    diagnostics: dict[str, dict[str, np.ndarray | tuple[int, int] | bool | int]] = {}
    for state in np.sort(np.unique(states)):
        sub_x = x[states == state]
        correlation_diagnostics = compute_correlation_diagnostics(sub_x)
        diagnostics[f"state_{state}"] = {
            "n_obs": int(sub_x.shape[0]),
            "shape": sub_x.shape,
            "is_finite": bool(np.all(np.isfinite(sub_x))),
            "column_means": correlation_diagnostics["column_means"],
            "column_stds": correlation_diagnostics["column_stds"],
            "correlation": correlation_diagnostics["correlation"],
        }

    return diagnostics


def compute_information_diagnostics(
    x: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> dict[str, object]:
    """Calcula diagnosticos de informacion usando wrappers RBIG del toolbox."""
    _validate_2d_data_array(x)

    return {
        "shape": x.shape,
        "total_correlation": estimate_rbig_total_correlation(
            x,
            max_layers=max_layers,
            rotation=rotation,
            units=units,
        ),
        "joint_entropy": estimate_rbig_joint_entropy(
            x,
            max_layers=max_layers,
            rotation=rotation,
            units=units,
        ),
        "units": units,
    }


def compute_state_conditioned_information_diagnostics(
    x: np.ndarray,
    states: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> dict[str, dict[str, object]]:
    """Calcula diagnosticos de informacion por estado para los estados presentes."""
    _validate_2d_data_array(x)
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy.ndarray")
    if states.ndim != 1:
        raise ValueError("states must be a 1D array")
    if states.shape[0] == 0:
        raise ValueError("states must be non-empty")
    if x.shape[0] != states.shape[0]:
        raise ValueError("x and states must have the same number of observations")

    diagnostics: dict[str, dict[str, object]] = {}
    for state in np.sort(np.unique(states)):
        sub_x = x[states == state]
        diagnostics[f"state_{state}"] = compute_information_diagnostics(
            sub_x,
            max_layers=max_layers,
            rotation=rotation,
            units=units,
        )

    return diagnostics


def compute_pairwise_mutual_information(
    x: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> np.ndarray:
    """Calcula la matriz simetrica de informacion mutua por pares."""
    _validate_2d_data_array(x)

    n_features = x.shape[1]
    mi_matrix = np.zeros((n_features, n_features), dtype=float)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = estimate_rbig_mutual_information(
                x[:, [i]],
                x[:, [j]],
                max_layers=max_layers,
                rotation=rotation,
                units=units,
            )
            mi_matrix[i, j] = float(mi)
            mi_matrix[j, i] = float(mi)

    return mi_matrix


def compute_pairwise_information_correlation(
    x: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
) -> np.ndarray:
    """Calcula una correlacion equivalente derivada de la informacion mutua."""
    mi_matrix = compute_pairwise_mutual_information(
        x,
        max_layers=max_layers,
        rotation=rotation,
        units="nats",
    )
    return np.sqrt(1.0 - np.exp(-2.0 * mi_matrix))


def compute_state_conditioned_mutual_information(
    x: np.ndarray,
    states: np.ndarray,
    *,
    max_layers: int = 200,
    rotation: str = "PCA",
    units: Literal["nats", "bits"] = "nats",
) -> dict[str, np.ndarray]:
    """Calcula una matriz de informacion mutua por cada estado presente."""
    _validate_2d_data_array(x)
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy.ndarray")
    if states.ndim != 1:
        raise ValueError("states must be a 1D array")
    if states.shape[0] == 0:
        raise ValueError("states must be non-empty")
    if x.shape[0] != states.shape[0]:
        raise ValueError("x and states must have the same number of observations")

    diagnostics: dict[str, np.ndarray] = {}
    for state in np.sort(np.unique(states)):
        sub_x = x[states == state]
        diagnostics[f"state_{state}"] = compute_pairwise_mutual_information(
            sub_x,
            max_layers=max_layers,
            rotation=rotation,
            units=units,
        )

    return diagnostics
