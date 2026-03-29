"""Modulo de graficos del toolbox de datos sinteticos."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "plot_sample_paths",
    "plot_marginal_histograms",
    "plot_correlation_heatmap",
    "plot_state_path",
    "plot_state_conditioned_histograms",
    "plot_state_conditioned_correlation_heatmaps",
]


def _validate_2d_data_array(x: np.ndarray) -> None:
    """Valida un array bidimensional no vacio."""
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must have positive shape in both dimensions")


def _validate_states_array(states: np.ndarray) -> None:
    """Valida un array unidimensional no vacio de estados."""
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy.ndarray")
    if states.ndim != 1:
        raise ValueError("states must be a 1D array")
    if states.shape[0] == 0:
        raise ValueError("states must be non-empty")


def _validate_x_states_compatibility(x: np.ndarray, states: np.ndarray) -> None:
    """Valida compatibilidad entre matriz de datos y trayectoria de estados."""
    _validate_2d_data_array(x)
    _validate_states_array(states)
    if x.shape[0] != states.shape[0]:
        raise ValueError("x and states must have the same number of observations")


def plot_sample_paths(x: np.ndarray) -> tuple[Figure, Axes]:
    """Representa cada columna de ``x`` como una trayectoria temporal."""
    _validate_2d_data_array(x)

    fig, ax = plt.subplots()
    observation_index = np.arange(x.shape[0], dtype=int)
    for column_index in range(x.shape[1]):
        ax.plot(observation_index, x[:, column_index])

    ax.set_title("Sample paths")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Value")
    ax.grid(True)
    return fig, ax


def plot_marginal_histograms(x: np.ndarray) -> tuple[Figure, np.ndarray]:
    """Representa un histograma por columna de ``x``."""
    _validate_2d_data_array(x)

    fig, axes = plt.subplots(1, x.shape[1], squeeze=False)
    axes_array = axes.ravel()
    for column_index, ax in enumerate(axes_array):
        ax.hist(x[:, column_index], bins=30)
        ax.set_title(f"Histogram {column_index}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    return fig, axes_array


def plot_correlation_heatmap(x: np.ndarray) -> tuple[Figure, Axes]:
    """Representa un mapa de calor de la correlacion empirica de ``x``."""
    _validate_2d_data_array(x)

    correlation = np.atleast_2d(np.corrcoef(x, rowvar=False))
    fig, ax = plt.subplots()
    image = ax.imshow(correlation)
    fig.colorbar(image, ax=ax)
    ax.set_title("Correlation heatmap")
    return fig, ax


def plot_state_path(states: np.ndarray) -> tuple[Figure, Axes]:
    """Representa la trayectoria temporal de los estados."""
    _validate_states_array(states)

    fig, ax = plt.subplots()
    observation_index = np.arange(states.shape[0], dtype=int)
    ax.plot(observation_index, states)
    ax.set_title("State path")
    ax.set_xlabel("Observation")
    ax.set_ylabel("State")
    ax.grid(True)
    return fig, ax


def plot_state_conditioned_histograms(
    x: np.ndarray,
    states: np.ndarray,
    column_index: int = 0,
) -> tuple[Figure, Axes]:
    """Representa histogramas de una columna condicionados por estado."""
    _validate_x_states_compatibility(x, states)
    if isinstance(column_index, bool) or not isinstance(column_index, int):
        raise TypeError("column_index must be an integer")
    if column_index < 0 or column_index >= x.shape[1]:
        raise ValueError("column_index is out of bounds")

    fig, ax = plt.subplots()
    for state in np.sort(np.unique(states)):
        ax.hist(
            x[states == state, column_index],
            bins=30,
            alpha=0.5,
            label=f"state_{state}",
        )

    ax.set_title(f"State-conditioned histograms (column {column_index})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig, ax


def plot_state_conditioned_correlation_heatmaps(
    x: np.ndarray,
    states: np.ndarray,
) -> tuple[Figure, np.ndarray]:
    """Representa un mapa de calor de correlacion por cada estado presente."""
    _validate_x_states_compatibility(x, states)

    present_states = np.sort(np.unique(states))
    fig, axes = plt.subplots(1, present_states.shape[0], squeeze=False)
    axes_array = axes.ravel()
    for ax, state in zip(axes_array, present_states):
        correlation = np.atleast_2d(np.corrcoef(x[states == state], rowvar=False))
        image = ax.imshow(correlation)
        ax.set_title(f"Correlation state_{state}")
        fig.colorbar(image, ax=ax)

    return fig, axes_array
