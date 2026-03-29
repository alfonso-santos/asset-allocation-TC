"""Paquete que contendra el toolbox de generacion de datos sinteticos."""

from tc_synthetic.diagnostics import (
    compute_basic_diagnostics,
    compute_correlation_diagnostics,
    compute_information_diagnostics,
    compute_pairwise_information_correlation,
    compute_pairwise_mutual_information,
    compute_state_conditioned_information_diagnostics,
    compute_state_conditioned_diagnostics,
    compute_state_conditioned_mutual_information,
)
from tc_synthetic.generator import (
    generate_static_scenario_data,
    generate_two_state_scenario_data,
)
from tc_synthetic.information import (
    estimate_rbig_joint_entropy,
    estimate_rbig_mutual_information,
    estimate_rbig_total_correlation,
)
from tc_synthetic.plots import (
    plot_correlation_heatmap,
    plot_marginal_histograms,
    plot_sample_paths,
    plot_state_conditioned_correlation_heatmaps,
    plot_state_conditioned_histograms,
    plot_state_path,
)
from tc_synthetic.smoke import (
    run_dynamic_smoke,
    run_special_smoke,
    run_static_smoke,
)
from tc_synthetic.special_generators import (
    generate_nonlinear_redundancy_data,
    generate_special_structure_data,
)
from tc_synthetic.specs import (
    CopulaSpec,
    MarginalSpec,
    ScenarioSpec,
    StateProcessSpec,
    StructureSpec,
)

__all__ = [
    "MarginalSpec",
    "CopulaSpec",
    "StructureSpec",
    "StateProcessSpec",
    "ScenarioSpec",
    "generate_static_scenario_data",
    "generate_two_state_scenario_data",
    "generate_special_structure_data",
    "generate_nonlinear_redundancy_data",
    "compute_basic_diagnostics",
    "compute_correlation_diagnostics",
    "compute_information_diagnostics",
    "compute_state_conditioned_information_diagnostics",
    "compute_pairwise_mutual_information",
    "compute_pairwise_information_correlation",
    "compute_state_conditioned_mutual_information",
    "compute_state_conditioned_diagnostics",
    "estimate_rbig_total_correlation",
    "estimate_rbig_joint_entropy",
    "estimate_rbig_mutual_information",
    "plot_sample_paths",
    "plot_marginal_histograms",
    "plot_correlation_heatmap",
    "plot_state_path",
    "plot_state_conditioned_histograms",
    "plot_state_conditioned_correlation_heatmaps",
    "run_static_smoke",
    "run_dynamic_smoke",
    "run_special_smoke",
]
__version__ = "0.1.0"
