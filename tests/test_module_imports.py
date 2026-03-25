"""Pruebas minimas de importacion para los modulos de ``tc_synthetic``."""

import importlib

import pytest
import tc_synthetic


MODULE_NAMES = [
    "tc_synthetic",
    "tc_synthetic.specs",
    "tc_synthetic.utils",
    "tc_synthetic.marginals",
    "tc_synthetic.structures",
    "tc_synthetic.copulas",
    "tc_synthetic.states",
    "tc_synthetic.special_generators",
    "tc_synthetic.generator",
    "tc_synthetic.scenarios",
    "tc_synthetic.diagnostics",
    "tc_synthetic.plots",
    "tc_synthetic.smoke",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports(module_name: str) -> None:
    """Verifica que cada modulo puede importarse."""
    module = importlib.import_module(module_name)
    assert module is not None


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_exposes_all(module_name: str) -> None:
    """Verifica que cada modulo expone ``__all__``."""
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__")


def test_package_exposes_version_string() -> None:
    """Verifica que la version del paquete existe y es una cadena."""
    assert hasattr(tc_synthetic, "__version__")
    assert isinstance(tc_synthetic.__version__, str)
