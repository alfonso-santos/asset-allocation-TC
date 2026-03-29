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


# === Added in Step Final.1 ===

@pytest.mark.parametrize(
    "name",
    [
        "MarginalSpec",
        "ScenarioSpec",
        "generate_static_scenario_data",
        "generate_two_state_scenario_data",
        "generate_special_structure_data",
        "compute_basic_diagnostics",
        "plot_sample_paths",
        "run_static_smoke",
    ],
)
def test_package_exposes_main_api_attributes(name: str) -> None:
    """Verifica que el paquete raiz expone atributos principales."""
    assert hasattr(tc_synthetic, name)


@pytest.mark.parametrize(
    "name",
    [
        "MarginalSpec",
        "ScenarioSpec",
        "generate_static_scenario_data",
        "generate_two_state_scenario_data",
        "generate_special_structure_data",
        "compute_basic_diagnostics",
        "plot_sample_paths",
        "run_static_smoke",
    ],
)
def test_package_all_contains_main_api_names(name: str) -> None:
    """Verifica que ``__all__`` incluye nombres principales del paquete."""
    assert name in tc_synthetic.__all__


def test_package_version_matches_expected_public_version() -> None:
    """Verifica que la version publica del paquete coincide con la esperada."""
    assert tc_synthetic.__version__ == "0.1.0"


# === Added in Step Final.2 ===

@pytest.mark.parametrize(
    "name",
    [
        "estimate_rbig_total_correlation",
        "estimate_rbig_joint_entropy",
        "estimate_rbig_mutual_information",
        "compute_information_diagnostics",
        "compute_state_conditioned_information_diagnostics",
        "compute_pairwise_mutual_information",
        "compute_pairwise_information_correlation",
        "compute_state_conditioned_mutual_information",
    ],
)
def test_package_exposes_rbig_api_attributes(name: str) -> None:
    """Verifica que el paquete raiz expone los helpers y diagnosticos RBIG."""
    assert hasattr(tc_synthetic, name)


@pytest.mark.parametrize(
    "name",
    [
        "estimate_rbig_total_correlation",
        "estimate_rbig_joint_entropy",
        "estimate_rbig_mutual_information",
        "compute_information_diagnostics",
        "compute_state_conditioned_information_diagnostics",
        "compute_pairwise_mutual_information",
        "compute_pairwise_information_correlation",
        "compute_state_conditioned_mutual_information",
    ],
)
def test_package_all_contains_rbig_api_names(name: str) -> None:
    """Verifica que ``__all__`` incluye los nombres RBIG expuestos."""
    assert name in tc_synthetic.__all__
