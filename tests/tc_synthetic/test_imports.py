"""Pruebas minimas de importacion para el paquete ``tc_synthetic``."""

import tc_synthetic


def test_package_imports() -> None:
    """Verifica que el paquete puede importarse."""
    assert tc_synthetic.__name__ == "tc_synthetic"


def test_module_exposes_all() -> None:
    """Verifica que el modulo expone ``__all__``."""
    assert hasattr(tc_synthetic, "__all__")


def test_module_exposes_version_string() -> None:
    """Verifica que la version existe y es una cadena."""
    assert hasattr(tc_synthetic, "__version__")
    assert isinstance(tc_synthetic.__version__, str)
