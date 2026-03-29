"""Pruebas para ``tc_synthetic.plots``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tc_synthetic.plots import (
    plot_correlation_heatmap,
    plot_marginal_histograms,
    plot_sample_paths,
)


def test_plot_sample_paths_returns_figure_axes_and_expected_labels() -> None:
    """Verifica retorno, numero de lineas y etiquetas del grafico de trayectorias."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    fig, ax = plot_sample_paths(x)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == x.shape[1]
    assert ax.get_title() == "Sample paths"
    assert ax.get_xlabel() == "Observation"
    assert ax.get_ylabel() == "Value"
    plt.close(fig)


def test_plot_sample_paths_raises_for_non_array_input() -> None:
    """Verifica que ``x`` debe ser un ``numpy.ndarray``."""
    with pytest.raises(TypeError, match="x must be a numpy.ndarray"):
        plot_sample_paths([[1.0, 2.0], [3.0, 4.0]])


def test_plot_sample_paths_raises_for_non_2d_input() -> None:
    """Verifica que ``x`` debe ser bidimensional."""
    with pytest.raises(ValueError, match="x must be a 2D array"):
        plot_sample_paths(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "x",
    [
        np.empty((0, 2), dtype=float),
        np.empty((2, 0), dtype=float),
    ],
)
def test_plot_sample_paths_raises_when_any_dimension_is_zero(x: np.ndarray) -> None:
    """Verifica que ambas dimensiones de ``x`` deben ser positivas."""
    with pytest.raises(ValueError, match="x must have positive shape in both dimensions"):
        plot_sample_paths(x)


def test_plot_marginal_histograms_returns_1d_axes_array_for_multiple_columns() -> None:
    """Verifica que los histogramas devuelven un array 1D de axes con titulos correctos."""
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    fig, axes = plot_marginal_histograms(x)

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 1
    assert len(axes) == x.shape[1]
    assert [ax.get_title() for ax in axes] == [
        "Histogram 0",
        "Histogram 1",
        "Histogram 2",
    ]
    plt.close(fig)


def test_plot_marginal_histograms_returns_1d_axes_array_for_single_column() -> None:
    """Verifica que una sola columna sigue devolviendo un array 1D de longitud 1."""
    x = np.array([[1.0], [2.0], [3.0]])

    fig, axes = plot_marginal_histograms(x)

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 1
    assert len(axes) == 1
    assert axes[0].get_title() == "Histogram 0"
    plt.close(fig)


def test_plot_correlation_heatmap_returns_figure_axes_and_image() -> None:
    """Verifica retorno, titulo e imagen del mapa de calor de correlacion."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    fig, ax = plot_correlation_heatmap(x)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Correlation heatmap"
    assert len(ax.images) == 1
    plt.close(fig)


def test_plot_correlation_heatmap_raises_for_non_array_input() -> None:
    """Verifica que ``x`` debe ser un ``numpy.ndarray``."""
    with pytest.raises(TypeError, match="x must be a numpy.ndarray"):
        plot_correlation_heatmap([[1.0, 2.0], [3.0, 4.0]])


def test_plot_correlation_heatmap_raises_for_non_2d_input() -> None:
    """Verifica que ``x`` debe ser bidimensional."""
    with pytest.raises(ValueError, match="x must be a 2D array"):
        plot_correlation_heatmap(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "x",
    [
        np.empty((0, 2), dtype=float),
        np.empty((2, 0), dtype=float),
    ],
)
def test_plot_correlation_heatmap_raises_when_any_dimension_is_zero(x: np.ndarray) -> None:
    """Verifica que ambas dimensiones de ``x`` deben ser positivas."""
    with pytest.raises(ValueError, match="x must have positive shape in both dimensions"):
        plot_correlation_heatmap(x)


# === Added in Step 12.2 ===

from tc_synthetic.plots import (
    plot_state_conditioned_correlation_heatmaps,
    plot_state_conditioned_histograms,
    plot_state_path,
)


def test_plot_state_path_returns_figure_axes_and_expected_labels() -> None:
    """Verifica retorno, linea y etiquetas del grafico de estados."""
    states = np.array([0, 0, 1, 1, 0])

    fig, ax = plot_state_path(states)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1
    assert ax.get_title() == "State path"
    assert ax.get_xlabel() == "Observation"
    assert ax.get_ylabel() == "State"
    plt.close(fig)


def test_plot_state_path_raises_for_non_array_input() -> None:
    """Verifica que ``states`` debe ser un ``numpy.ndarray``."""
    with pytest.raises(TypeError, match="states must be a numpy.ndarray"):
        plot_state_path([0, 1, 0])


def test_plot_state_path_raises_for_non_1d_input() -> None:
    """Verifica que ``states`` debe ser un array unidimensional."""
    with pytest.raises(ValueError, match="states must be a 1D array"):
        plot_state_path(np.array([[0, 1, 0]]))


def test_plot_state_path_raises_for_empty_states() -> None:
    """Verifica que ``states`` no puede estar vacio."""
    with pytest.raises(ValueError, match="states must be non-empty"):
        plot_state_path(np.array([], dtype=int))


def test_plot_state_conditioned_histograms_returns_expected_labels_and_legend() -> None:
    """Verifica el caso valido basico del histograma condicionado por estado."""
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [4.0, 13.0],
        ]
    )
    states = np.array([0, 0, 1, 1])

    fig, ax = plot_state_conditioned_histograms(x, states, column_index=0)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_title() == "State-conditioned histograms (column 0)"
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["state_0", "state_1"]
    plt.close(fig)


def test_plot_state_conditioned_histograms_raises_for_length_mismatch() -> None:
    """Verifica que ``x`` y ``states`` deben tener igual numero de observaciones."""
    x = np.array([[1.0], [2.0], [3.0]])
    states = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="x and states must have the same number of observations",
    ):
        plot_state_conditioned_histograms(x, states)


def test_plot_state_conditioned_histograms_raises_for_non_integer_column_index() -> None:
    """Verifica que ``column_index`` debe ser un entero valido."""
    x = np.array([[1.0], [2.0], [3.0]])
    states = np.array([0, 1, 0])

    with pytest.raises(TypeError, match="column_index must be an integer"):
        plot_state_conditioned_histograms(x, states, column_index=0.5)


def test_plot_state_conditioned_histograms_raises_for_out_of_bounds_column_index() -> None:
    """Verifica que ``column_index`` debe caer dentro del rango de columnas."""
    x = np.array([[1.0], [2.0], [3.0]])
    states = np.array([0, 1, 0])

    with pytest.raises(ValueError, match="column_index is out of bounds"):
        plot_state_conditioned_histograms(x, states, column_index=1)


def test_plot_state_conditioned_correlation_heatmaps_returns_expected_axes_for_two_states() -> None:
    """Verifica retorno y titulos correctos con dos estados presentes."""
    x = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]
    )
    states = np.array([0, 0, 1, 1])

    fig, axes = plot_state_conditioned_correlation_heatmaps(x, states)

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 1
    assert len(axes) == 2
    assert [ax.get_title() for ax in axes] == ["Correlation state_0", "Correlation state_1"]
    plt.close(fig)


def test_plot_state_conditioned_correlation_heatmaps_returns_1d_axes_for_single_state() -> None:
    """Verifica que un solo estado sigue devolviendo un array 1D de longitud 1."""
    x = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    states = np.array([1, 1, 1])

    fig, axes = plot_state_conditioned_correlation_heatmaps(x, states)

    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.ndim == 1
    assert len(axes) == 1
    assert axes[0].get_title() == "Correlation state_1"
    plt.close(fig)


def test_plot_state_conditioned_correlation_heatmaps_raises_for_length_mismatch() -> None:
    """Verifica que ``x`` y ``states`` deben tener igual numero de observaciones."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    states = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="x and states must have the same number of observations",
    ):
        plot_state_conditioned_correlation_heatmaps(x, states)
