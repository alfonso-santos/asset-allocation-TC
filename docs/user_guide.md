# User Guide

## 1. Generating Data
Most workflows in `tc_synthetic` start with declarative specifications. In the static setting, data generation is driven by three ingredients:

- a structure, which defines the target dependence pattern
- a copula, which defines how dependence is coupled across marginals
- a marginal, which defines the univariate distribution of each variable

This separation makes it easy to change one layer without rewriting the rest of the configuration.

```python
import numpy as np

from tc_synthetic import (
    StructureSpec,
    CopulaSpec,
    MarginalSpec,
    generate_static_scenario_data,
)

rng = np.random.default_rng(123)

structure = StructureSpec(kind="block", params={
    "block_sizes": [2, 2],
    "rho_within": 0.6,
    "rho_between": 0.1,
})
copula = CopulaSpec(kind="gaussian")
marginal = MarginalSpec(kind="student_t", params={"df": 5.0})

x = generate_static_scenario_data(
    n_obs=1000,
    n_assets=4,
    structure=structure,
    copula=copula,
    marginal=marginal,
    rng=rng,
)
```

## 2. Static Scenarios
Use `generate_static_scenario_data(...)` when the full sample should be generated under a single dependence regime. This is the simplest entry point and is appropriate for controlled experiments where the data-generating process does not switch over time.

```python
import numpy as np

from tc_synthetic import (
    StructureSpec,
    CopulaSpec,
    MarginalSpec,
    generate_static_scenario_data,
)

rng = np.random.default_rng(123)

x = generate_static_scenario_data(
    n_obs=500,
    n_assets=3,
    structure=StructureSpec(kind="equicorrelation", params={"rho": 0.5}),
    copula=CopulaSpec(kind="gaussian"),
    marginal=MarginalSpec(kind="gaussian"),
    rng=rng,
)
```

## 3. Dynamic Scenarios
Dynamic generation is useful when the sample should alternate between distinct regimes, such as a calm market and a crisis market. In this workflow, each regime is represented by a `ScenarioSpec`, and the switching logic is controlled by a `StateProcessSpec`.

`generate_two_state_scenario_data(...)` returns both the generated sample and the state path used to route observations through the corresponding scenario.

```python
import numpy as np

from tc_synthetic import (
    ScenarioSpec,
    StructureSpec,
    CopulaSpec,
    MarginalSpec,
    StateProcessSpec,
    generate_two_state_scenario_data,
)

rng = np.random.default_rng(123)

calm = ScenarioSpec(
    name="calm",
    n_obs=500,
    n_assets=3,
    structure=StructureSpec(kind="equicorrelation", params={"rho": 0.2}),
    copula=CopulaSpec(kind="gaussian"),
    marginal=MarginalSpec(kind="gaussian"),
)

crisis = ScenarioSpec(
    name="crisis",
    n_obs=500,
    n_assets=3,
    structure=StructureSpec(kind="equicorrelation", params={"rho": 0.7}),
    copula=CopulaSpec(kind="gaussian"),
    marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
)

state_process = StateProcessSpec(
    kind="markov",
    params={
        "transition_matrix": np.array([[0.97, 0.03], [0.05, 0.95]]),
        "initial_state": 0,
    },
    enabled=True,
)

x, states = generate_two_state_scenario_data(calm, crisis, state_process, rng)
```

## 4. Special Generators
`nonlinear_redundancy` is designed for experiments where linear correlation is not enough to describe the structure of dependence. It is especially useful for testing feature-selection methods, redundancy detection, and nonlinear information metrics.

Use `generate_special_structure_data(...)` when the structure itself requires a dedicated generator rather than the standard structure-copula-marginal pipeline.

```python
import numpy as np

from tc_synthetic import (
    StructureSpec,
    MarginalSpec,
    generate_special_structure_data,
)

rng = np.random.default_rng(123)

x = generate_special_structure_data(
    n_obs=1000,
    n_assets=4,
    structure=StructureSpec(
        kind="nonlinear_redundancy",
        params={"group_sizes": [2, 2]},
    ),
    marginal=MarginalSpec(kind="gaussian"),
    rng=rng,
)
```

## 5. Core Information Functions
The package also provides core information-theoretic functions that can be used directly in algorithms, not only in diagnostics. RBIG-based functions estimate nonlinear dependence, while Gaussian benchmark functions provide a reference based on rank-gaussianized correlation.

The most important functions in this layer are:

- `estimate_rbig_total_correlation`
- `estimate_rbig_mutual_information`
- `compute_gaussian_total_correlation`
- `compute_excess_total_correlation`

```python
from tc_synthetic import (
    estimate_rbig_total_correlation,
    compute_gaussian_total_correlation,
    compute_excess_total_correlation,
)

tc = estimate_rbig_total_correlation(x, max_layers=50)
tc_gauss = compute_gaussian_total_correlation(x)
xi = compute_excess_total_correlation(x, max_layers=50)
```

## 6. Diagnostics
`tc_synthetic` includes several layers of diagnostics, each aimed at a different validation question.

- Basic diagnostics summarize shape, finiteness, means, spreads, and ranges.
- Marginal diagnostics add skewness, kurtosis, percentiles, and tail ratios.
- Correlation diagnostics summarize linear dependence.
- Information diagnostics estimate nonlinear dependence metrics through RBIG.

```python
from tc_synthetic import (
    compute_basic_diagnostics,
    compute_marginal_distribution_diagnostics,
    compute_correlation_diagnostics,
    compute_information_diagnostics,
)

basic = compute_basic_diagnostics(x)
marginal = compute_marginal_distribution_diagnostics(x)
corr = compute_correlation_diagnostics(x)
info = compute_information_diagnostics(x, max_layers=50)
```

## 7. Plots
The plotting layer covers the most common inspection tasks: sample paths, marginal histograms, correlation heatmaps, and state-aware visualizations for dynamic samples.

```python
from tc_synthetic import (
    plot_sample_paths,
    plot_marginal_histograms,
    plot_correlation_heatmap,
    plot_state_path,
)

fig1, ax1 = plot_sample_paths(x)
fig2, axes2 = plot_marginal_histograms(x)
fig3, ax3 = plot_correlation_heatmap(x)
fig4, ax4 = plot_state_path(states)
```

## 8. Smoke Utilities
Smoke utilities provide short end-to-end runs that exercise the package with predefined configurations. They are useful as sanity checks during development, environment setup, or quick regression testing.

The three entry points are:

- `run_static_smoke()`
- `run_dynamic_smoke()`
- `run_special_smoke()`

```python
from tc_synthetic import (
    run_static_smoke,
    run_dynamic_smoke,
    run_special_smoke,
)

static_result = run_static_smoke()
dynamic_result = run_dynamic_smoke()
special_result = run_special_smoke()
```
