# tc_synthetic

## Overview
`tc_synthetic` is a modular Python toolbox for generating synthetic multivariate data with controlled marginals, dependence structures, copulas, dynamic regime changes, and special nonlinear redundancy patterns.

It is designed for methodological experiments, dependence analysis, portfolio and asset-selection research, and information-theoretic diagnostics. The package emphasizes declarative configuration, composable building blocks, and reproducible end-to-end workflows.

## Installation
Install the toolbox from a local clone of the repository. A typical workflow is:

1. Clone the repository.
2. Move into the project directory.
3. Install the package in editable mode.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -e .
```

## Optional RBIG Installation
Some advanced information-theoretic diagnostics depend on RBIG. This dependency is optional and only required if you want to use total correlation, joint entropy, mutual information, or Gaussian excess dependence metrics.

Install RBIG with:

```bash
pip install "git+https://github.com/ipl-uv/rbig.git"
```

## Quick Example
```python
import numpy as np

from tc_synthetic import (
    StructureSpec,
    CopulaSpec,
    MarginalSpec,
    generate_static_scenario_data,
    compute_basic_diagnostics,
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
    n_obs=500,
    n_assets=4,
    structure=structure,
    copula=copula,
    marginal=marginal,
    rng=rng,
)

diagnostics = compute_basic_diagnostics(x)
```

## Architecture
The toolbox is organized around small, focused modules. `specs.py` defines declarative configuration objects. `structures.py`, `copulas.py`, `marginals.py`, and `states.py` implement the atomic components used to build synthetic data models.

`generator.py` and `special_generators.py` assemble these components into static, dynamic, and specialized data-generation pipelines. `diagnostics.py` and `information.py` provide descriptive and information-theoretic analysis tools, while `plots.py` and `smoke.py` support inspection and end-to-end validation.

## Public API
The package exposes the main user-facing classes and functions directly from `tc_synthetic`, so common workflows can be written without importing many submodules. The root API covers specifications, generators, diagnostics, information metrics, plots, and smoke helpers.

More specialized functionality remains available in the underlying submodules when finer control is needed.

## Diagnostics
The toolbox includes several layers of diagnostics: basic descriptive summaries, marginal distribution diagnostics, correlation diagnostics, state-conditioned diagnostics, and information diagnostics. These are intended to help verify both the generated samples and the structural assumptions used to produce them.

## Information Metrics (RBIG)
RBIG-based metrics are available as optional advanced tools for nonlinear dependence analysis. These include total correlation, mutual information, joint entropy, Gaussian benchmark total correlation, and excess dependence relative to the Gaussian case.

They are especially useful when linear correlation is not sufficient to describe the dependence structure of the data.

## Limitations
Some information metrics are computationally expensive, especially on larger samples or when repeated across many variable pairs. RBIG is an optional external dependency and must be installed separately for advanced information diagnostics.

Pairwise mutual information can become slow in higher dimensions, and the synthetic generators are designed for controlled experiments rather than perfect market realism.
