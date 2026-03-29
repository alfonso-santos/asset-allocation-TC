"""Pruebas para los recorridos smoke de ``tc_synthetic``."""

import numpy as np

from tc_synthetic.smoke import (
    run_dynamic_smoke,
    run_special_smoke,
    run_static_smoke,
)


def test_run_static_smoke_returns_expected_structure() -> None:
    """Verifica claves y coherencia basica del smoke estatico."""
    result = run_static_smoke()

    assert list(result.keys()) == [
        "x",
        "basic_diagnostics",
        "correlation_diagnostics",
    ]
    x = result["x"]
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert result["basic_diagnostics"]["shape"] == x.shape
    assert result["correlation_diagnostics"]["correlation"].shape == (x.shape[1], x.shape[1])


def test_run_static_smoke_is_reproducible() -> None:
    """Verifica reproducibilidad del smoke estatico con la misma seed."""
    result_a = run_static_smoke(seed=123)
    result_b = run_static_smoke(seed=123)

    assert np.array_equal(result_a["x"], result_b["x"])


def test_run_dynamic_smoke_returns_expected_structure() -> None:
    """Verifica claves y coherencia basica del smoke dinamico."""
    result = run_dynamic_smoke()

    assert list(result.keys()) == [
        "x",
        "states",
        "basic_diagnostics",
        "state_conditioned_diagnostics",
    ]
    x = result["x"]
    states = result["states"]
    assert isinstance(x, np.ndarray)
    assert isinstance(states, np.ndarray)
    assert x.ndim == 2
    assert states.ndim == 1
    assert x.shape[0] == states.shape[0]
    assert any(state_key in result["state_conditioned_diagnostics"] for state_key in ["state_0", "state_1"])


def test_run_dynamic_smoke_is_reproducible() -> None:
    """Verifica reproducibilidad del smoke dinamico con la misma seed."""
    result_a = run_dynamic_smoke(seed=123)
    result_b = run_dynamic_smoke(seed=123)

    assert np.array_equal(result_a["x"], result_b["x"])
    assert np.array_equal(result_a["states"], result_b["states"])


def test_run_special_smoke_returns_expected_structure() -> None:
    """Verifica claves y coherencia basica del smoke especial."""
    result = run_special_smoke()

    assert list(result.keys()) == [
        "x",
        "basic_diagnostics",
        "correlation_diagnostics",
    ]
    x = result["x"]
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert result["basic_diagnostics"]["shape"] == x.shape
    assert result["correlation_diagnostics"]["correlation"].shape == (x.shape[1], x.shape[1])


def test_run_special_smoke_is_reproducible() -> None:
    """Verifica reproducibilidad del smoke especial con la misma seed."""
    result_a = run_special_smoke(seed=123)
    result_b = run_special_smoke(seed=123)

    assert np.array_equal(result_a["x"], result_b["x"])
