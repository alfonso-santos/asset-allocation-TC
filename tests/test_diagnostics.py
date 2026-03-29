"""Pruebas para ``tc_synthetic.diagnostics``."""

import numpy as np
import pytest

from tc_synthetic.diagnostics import compute_basic_diagnostics


def test_compute_basic_diagnostics_returns_expected_keys_and_shapes() -> None:
    """Verifica claves, shape y tamanos por columna en el caso valido basico."""
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    result = compute_basic_diagnostics(x)

    assert list(result.keys()) == [
        "shape",
        "is_finite",
        "column_means",
        "column_stds",
        "column_mins",
        "column_maxs",
    ]
    assert result["shape"] == (2, 3)
    assert result["is_finite"] is True
    assert result["column_means"].shape == (3,)
    assert result["column_stds"].shape == (3,)
    assert result["column_mins"].shape == (3,)
    assert result["column_maxs"].shape == (3,)


def test_compute_basic_diagnostics_matches_known_numeric_values() -> None:
    """Verifica medias, desvios, minimos y maximos en un ejemplo conocido."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = compute_basic_diagnostics(x)

    np.testing.assert_allclose(result["column_means"], np.array([3.0, 4.0]))
    np.testing.assert_allclose(
        result["column_stds"],
        np.array(
            [
                np.sqrt(8.0 / 3.0),
                np.sqrt(8.0 / 3.0),
            ]
        ),
    )
    np.testing.assert_allclose(result["column_mins"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(result["column_maxs"], np.array([5.0, 6.0]))


def test_compute_basic_diagnostics_raises_for_non_array_input() -> None:
    """Verifica que ``x`` debe ser un ``numpy.ndarray``."""
    with pytest.raises(TypeError, match="x must be a numpy.ndarray"):
        compute_basic_diagnostics([[1.0, 2.0], [3.0, 4.0]])


def test_compute_basic_diagnostics_raises_for_non_2d_input() -> None:
    """Verifica que ``x`` debe ser bidimensional."""
    with pytest.raises(ValueError, match="x must be a 2D array"):
        compute_basic_diagnostics(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "x",
    [
        np.empty((0, 2), dtype=float),
        np.empty((2, 0), dtype=float),
    ],
)
def test_compute_basic_diagnostics_raises_when_any_dimension_is_zero(x: np.ndarray) -> None:
    """Verifica que ambas dimensiones de ``x`` deben ser positivas."""
    with pytest.raises(ValueError, match="x must have positive shape in both dimensions"):
        compute_basic_diagnostics(x)


def test_compute_basic_diagnostics_handles_nan_and_inf_without_failing() -> None:
    """Verifica que NaN o inf no rompen el diagnostico y marcan ``is_finite=False``."""
    x = np.array(
        [
            [1.0, np.nan],
            [np.inf, 2.0],
        ]
    )

    result = compute_basic_diagnostics(x)

    assert result["shape"] == (2, 2)
    assert result["is_finite"] is False
    assert result["column_means"].shape == (2,)
    assert result["column_stds"].shape == (2,)
    assert result["column_mins"].shape == (2,)
    assert result["column_maxs"].shape == (2,)


# === Added in Step 11.2 ===

from tc_synthetic.diagnostics import (
    compute_correlation_diagnostics,
    compute_state_conditioned_diagnostics,
)


def test_compute_correlation_diagnostics_returns_expected_keys_and_values() -> None:
    """Verifica claves, shape de correlacion y valores numericos basicos."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = compute_correlation_diagnostics(x)

    assert list(result.keys()) == ["correlation", "column_means", "column_stds"]
    assert result["correlation"].shape == (2, 2)
    np.testing.assert_allclose(result["correlation"], np.array([[1.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_allclose(result["column_means"], np.array([3.0, 4.0]))
    np.testing.assert_allclose(
        result["column_stds"],
        np.array([np.sqrt(8.0 / 3.0), np.sqrt(8.0 / 3.0)]),
    )


def test_compute_correlation_diagnostics_raises_for_non_array_input() -> None:
    """Verifica que ``x`` debe ser un ``numpy.ndarray``."""
    with pytest.raises(TypeError, match="x must be a numpy.ndarray"):
        compute_correlation_diagnostics([[1.0, 2.0], [3.0, 4.0]])


def test_compute_correlation_diagnostics_raises_for_non_2d_input() -> None:
    """Verifica que ``x`` debe ser bidimensional."""
    with pytest.raises(ValueError, match="x must be a 2D array"):
        compute_correlation_diagnostics(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "x",
    [
        np.empty((0, 2), dtype=float),
        np.empty((2, 0), dtype=float),
    ],
)
def test_compute_correlation_diagnostics_raises_when_any_dimension_is_zero(x: np.ndarray) -> None:
    """Verifica que ambas dimensiones de ``x`` deben ser positivas."""
    with pytest.raises(ValueError, match="x must have positive shape in both dimensions"):
        compute_correlation_diagnostics(x)


def test_compute_state_conditioned_diagnostics_returns_expected_structure() -> None:
    """Verifica el caso valido basico con estados 0 y 1."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    states = np.array([0, 0, 1, 1])

    result = compute_state_conditioned_diagnostics(x, states)

    assert list(result.keys()) == ["state_0", "state_1"]
    assert result["state_0"]["n_obs"] == 2
    assert result["state_1"]["n_obs"] == 2
    assert result["state_0"]["shape"] == (2, 2)
    assert result["state_1"]["shape"] == (2, 2)
    assert result["state_0"]["is_finite"] is True
    assert result["state_1"]["is_finite"] is True
    assert result["state_0"]["column_means"].shape == (2,)
    assert result["state_0"]["column_stds"].shape == (2,)
    assert result["state_0"]["correlation"].shape == (2, 2)
    assert result["state_1"]["column_means"].shape == (2,)
    assert result["state_1"]["column_stds"].shape == (2,)
    assert result["state_1"]["correlation"].shape == (2, 2)


def test_compute_state_conditioned_diagnostics_orders_states_ascending() -> None:
    """Verifica que las claves salen ordenadas por valor de estado."""
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    states = np.array([2, 0, 1, 2, 1, 0])

    result = compute_state_conditioned_diagnostics(x, states)

    assert list(result.keys()) == ["state_0", "state_1", "state_2"]


def test_compute_state_conditioned_diagnostics_raises_for_non_array_states() -> None:
    """Verifica que ``states`` debe ser un ``numpy.ndarray``."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError, match="states must be a numpy.ndarray"):
        compute_state_conditioned_diagnostics(x, [0, 1])


def test_compute_state_conditioned_diagnostics_raises_for_non_1d_states() -> None:
    """Verifica que ``states`` debe ser un array unidimensional."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    states = np.array([[0, 1]])

    with pytest.raises(ValueError, match="states must be a 1D array"):
        compute_state_conditioned_diagnostics(x, states)


def test_compute_state_conditioned_diagnostics_raises_for_length_mismatch() -> None:
    """Verifica que ``x`` y ``states`` deben tener igual numero de observaciones."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    states = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="x and states must have the same number of observations",
    ):
        compute_state_conditioned_diagnostics(x, states)


def test_compute_state_conditioned_diagnostics_raises_for_empty_states() -> None:
    """Verifica que ``states`` no puede estar vacio."""
    x = np.array([[1.0, 2.0]])
    states = np.array([], dtype=int)

    with pytest.raises(ValueError, match="states must be non-empty"):
        compute_state_conditioned_diagnostics(x, states)


def test_compute_state_conditioned_diagnostics_detects_non_finite_values_within_state() -> None:
    """Verifica que ``is_finite`` se calcula correctamente por estado."""
    x = np.array(
        [
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ]
    )
    states = np.array([0, 0, 1, 1])

    result = compute_state_conditioned_diagnostics(x, states)

    assert result["state_0"]["is_finite"] is False
    assert result["state_1"]["is_finite"] is True


# === Added in Step RBIG.2 ===

pytest.importorskip("rbig")

from tc_synthetic.diagnostics import (
    compute_information_diagnostics,
    compute_state_conditioned_information_diagnostics,
)


def test_compute_information_diagnostics_returns_expected_structure() -> None:
    """Verifica claves, shape y escalares finitos del diagnostico de informacion."""
    x = np.random.default_rng(123).standard_normal((300, 2))

    result = compute_information_diagnostics(x, max_layers=50)

    assert list(result.keys()) == ["shape", "total_correlation", "joint_entropy", "units"]
    assert result["shape"] == x.shape
    assert result["units"] == "nats"
    assert isinstance(result["total_correlation"], float)
    assert isinstance(result["joint_entropy"], float)
    assert np.isfinite(result["total_correlation"])
    assert np.isfinite(result["joint_entropy"])


def test_compute_information_diagnostics_supports_bits_units() -> None:
    """Verifica conversion a bits con salida finita."""
    x = np.random.default_rng(123).standard_normal((300, 2))

    result = compute_information_diagnostics(x, max_layers=50, units="bits")

    assert result["units"] == "bits"
    assert np.isfinite(result["total_correlation"])
    assert np.isfinite(result["joint_entropy"])


def test_compute_information_diagnostics_raises_for_non_array_input() -> None:
    """Verifica que ``x`` debe ser un ``numpy.ndarray``."""
    with pytest.raises(TypeError, match="x must be a numpy.ndarray"):
        compute_information_diagnostics([[1.0, 2.0], [3.0, 4.0]])


def test_compute_information_diagnostics_raises_for_non_2d_input() -> None:
    """Verifica que ``x`` debe ser bidimensional."""
    with pytest.raises(ValueError, match="x must be a 2D array"):
        compute_information_diagnostics(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "x",
    [
        np.empty((0, 2), dtype=float),
        np.empty((2, 0), dtype=float),
    ],
)
def test_compute_information_diagnostics_raises_when_any_dimension_is_zero(x: np.ndarray) -> None:
    """Verifica que ambas dimensiones de ``x`` deben ser positivas."""
    with pytest.raises(ValueError, match="x must have positive shape in both dimensions"):
        compute_information_diagnostics(x)


def test_compute_state_conditioned_information_diagnostics_returns_expected_structure() -> None:
    """Verifica el caso valido basico con estados 0 y 1."""
    x = np.random.default_rng(123).standard_normal((300, 2))
    states = np.array([0] * 150 + [1] * 150)

    result = compute_state_conditioned_information_diagnostics(x, states, max_layers=50)

    assert list(result.keys()) == ["state_0", "state_1"]
    for key in ["state_0", "state_1"]:
        assert list(result[key].keys()) == ["shape", "total_correlation", "joint_entropy", "units"]


def test_compute_state_conditioned_information_diagnostics_orders_states_ascending() -> None:
    """Verifica que las claves salen ordenadas por valor de estado."""
    x = np.random.default_rng(123).standard_normal((300, 2))
    states = np.array([2] * 100 + [0] * 100 + [1] * 100)

    result = compute_state_conditioned_information_diagnostics(x, states, max_layers=50)

    assert list(result.keys()) == ["state_0", "state_1", "state_2"]


def test_compute_state_conditioned_information_diagnostics_raises_for_non_array_states() -> None:
    """Verifica que ``states`` debe ser un ``numpy.ndarray``."""
    x = np.random.default_rng(123).standard_normal((300, 2))

    with pytest.raises(TypeError, match="states must be a numpy.ndarray"):
        compute_state_conditioned_information_diagnostics(x, [0, 1, 0], max_layers=50)


def test_compute_state_conditioned_information_diagnostics_raises_for_non_1d_states() -> None:
    """Verifica que ``states`` debe ser un array unidimensional."""
    x = np.random.default_rng(123).standard_normal((300, 2))
    states = np.array([[0, 1, 0]])

    with pytest.raises(ValueError, match="states must be a 1D array"):
        compute_state_conditioned_information_diagnostics(x, states, max_layers=50)


def test_compute_state_conditioned_information_diagnostics_raises_for_empty_states() -> None:
    """Verifica que ``states`` no puede estar vacio."""
    x = np.random.default_rng(123).standard_normal((1, 2))
    states = np.array([], dtype=int)

    with pytest.raises(ValueError, match="states must be non-empty"):
        compute_state_conditioned_information_diagnostics(x, states, max_layers=50)


def test_compute_state_conditioned_information_diagnostics_raises_for_length_mismatch() -> None:
    """Verifica que ``x`` y ``states`` deben tener igual numero de observaciones."""
    x = np.random.default_rng(123).standard_normal((300, 2))
    states = np.array([0] * 299)

    with pytest.raises(
        ValueError,
        match="x and states must have the same number of observations",
    ):
        compute_state_conditioned_information_diagnostics(x, states, max_layers=50)


# === Added in Step RBIG.3 ===

pytest.importorskip("rbig")

from tc_synthetic.diagnostics import (
    compute_pairwise_information_correlation,
    compute_pairwise_mutual_information,
    compute_state_conditioned_mutual_information,
)


def test_compute_pairwise_mutual_information_returns_valid_symmetric_matrix() -> None:
    """Verifica shape, diagonal nula y simetria de la matriz MI."""
    x = np.random.default_rng(123).standard_normal((200, 3))

    mi_matrix = compute_pairwise_mutual_information(x, max_layers=30)

    assert mi_matrix.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mi_matrix), np.zeros(3))
    np.testing.assert_allclose(mi_matrix, mi_matrix.T)


def test_compute_pairwise_mutual_information_is_small_for_independence() -> None:
    """Verifica que la MI fuera de diagonal es pequena para variables independientes."""
    x = np.random.default_rng(123).standard_normal((200, 3))

    mi_matrix = compute_pairwise_mutual_information(x, max_layers=30)
    off_diagonal = mi_matrix[~np.eye(mi_matrix.shape[0], dtype=bool)]

    assert np.all(off_diagonal < 0.1)


def test_compute_pairwise_mutual_information_detects_positive_dependence() -> None:
    """Verifica que la MI aumenta claramente cuando hay dependencia."""
    rng = np.random.default_rng(123)
    base = rng.standard_normal((200, 1))
    x = np.hstack([base, base + 0.2 * rng.standard_normal((200, 1))])

    mi_matrix = compute_pairwise_mutual_information(x, max_layers=30)

    assert mi_matrix[0, 1] > 0.1


def test_compute_pairwise_information_correlation_returns_values_between_zero_and_one() -> None:
    """Verifica rango y simetria de la correlacion equivalente basada en MI."""
    rng = np.random.default_rng(123)
    base = rng.standard_normal((200, 1))
    x = np.hstack(
        [
            base,
            base + 0.2 * rng.standard_normal((200, 1)),
            rng.standard_normal((200, 1)),
        ]
    )

    rho = compute_pairwise_information_correlation(x, max_layers=30)

    assert rho.shape == (3, 3)
    np.testing.assert_allclose(rho, rho.T)
    assert np.all(rho >= 0.0)
    assert np.all(rho <= 1.0)


def test_compute_state_conditioned_mutual_information_returns_expected_structure() -> None:
    """Verifica claves correctas y shape de matrices por estado."""
    x = np.random.default_rng(123).standard_normal((200, 3))
    states = np.array([0] * 100 + [1] * 100)

    result = compute_state_conditioned_mutual_information(x, states, max_layers=30)

    assert list(result.keys()) == ["state_0", "state_1"]
    assert result["state_0"].shape == (3, 3)
    assert result["state_1"].shape == (3, 3)


# === Added in Step D-EXT.1 ===

from tc_synthetic.diagnostics import compute_marginal_distribution_diagnostics


def test_marginal_diagnostics_structure() -> None:
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    result = compute_marginal_distribution_diagnostics(x)

    expected_keys = [
        "mean",
        "std",
        "skew",
        "kurtosis",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "tail_2sigma",
        "tail_3sigma",
    ]
    assert list(result.keys()) == expected_keys
    for value in result.values():
        assert value.shape == (2,)


def test_marginal_diagnostics_gaussian() -> None:
    x = np.random.default_rng(123).standard_normal((10000, 1))
    result = compute_marginal_distribution_diagnostics(x)

    assert abs(result["mean"][0]) < 0.05
    assert abs(result["std"][0] - 1.0) < 0.05
    assert 2.5 < result["kurtosis"][0] < 3.5


def test_marginal_diagnostics_student_t() -> None:
    x = np.random.default_rng(123).standard_t(df=5, size=(10000, 1))
    result = compute_marginal_distribution_diagnostics(x)

    assert result["kurtosis"][0] > 3.0


def test_tail_ratio_gaussian() -> None:
    x = np.random.default_rng(123).standard_normal((10000, 1))
    result = compute_marginal_distribution_diagnostics(x)

    assert result["tail_3sigma"][0] < 0.01


def test_marginal_diagnostics_errors() -> None:
    with pytest.raises(TypeError):
        compute_marginal_distribution_diagnostics([[1, 2]])

    with pytest.raises(ValueError):
        compute_marginal_distribution_diagnostics(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        compute_marginal_distribution_diagnostics(np.empty((0, 2)))
