# Architecture

## Repository Structure

The repository combines source code, tests, and research-facing working directories.
The relevant technical structure is:

```text
src/tc_synthetic/
    __init__.py
    specs.py
    utils.py
    structures.py
    marginals.py
    copulas.py
    states.py
    generator.py
    scenarios.py
    diagnostics.py
    plots.py
    smoke.py
    special_generators.py

tests/
    test_specs.py
    test_utils.py
    test_structures.py
    test_marginals.py
    test_module_imports.py
    test_smoke.py

docs/
    ...
```

Some modules already contain implemented logic.
Other modules exist as placeholders to reserve the intended architecture.

## Module Responsibilities

### `specs.py`

Defines the declarative configuration layer.
Its role is to express what should be generated, not how generation happens.

### `utils.py`

Defines reusable validation and matrix helpers.
This is the lowest-level shared module in the current implementation.

### `structures.py`

Defines structural representations for cross-sectional assumptions.
Some outputs are correlation matrices.
Others are explicit metadata containers when a matrix would be misleading.

### `marginals.py`

Defines univariate generators and related normalization helpers.
This module is the current bridge between declarative marginal specifications and actual sampling.

### Placeholder Modules

The following modules are present but intentionally minimal at this stage:

- `copulas.py`
- `states.py`
- `generator.py`
- `scenarios.py`
- `diagnostics.py`
- `plots.py`
- `smoke.py`
- `special_generators.py`

They indicate the planned layering of the project, but they should not be documented as fully implemented systems.

## General Flow of the Generator

The intended generator flow is:

1. Define a `ScenarioSpec`.
2. Validate structural configuration through dataclass construction.
3. Build univariate marginals according to the marginal layer.
4. Build structural assumptions according to the structure layer.
5. Add dependence and state logic in later phases.
6. Assemble final synthetic outputs.

At the current stage, steps 1 through 4 are partially implemented.
Steps 5 and 6 are not yet implemented.

## Current Dependency Structure

The implemented modules follow a simple dependency direction.

- `specs.py` depends only on the Python standard library.
- `utils.py` depends on `numpy`.
- `structures.py` depends on `numpy` and `tc_synthetic.utils`.
- `marginals.py` depends on `numpy`, `scipy.stats.skewnorm`, `tc_synthetic.utils`, and `tc_synthetic.specs`.

This matters because it keeps lower layers stable and reusable.
The foundational modules do not depend on higher-level orchestration code.

## Design Principles

### Explicitness over hidden magic

The code favors small explicit functions over large generic dispatch systems.
Current examples include dedicated functions for each marginal family and each structure family.

### Layer separation

Each module is expected to own one concern.
Validation helpers live in `utils`, scenario declarations live in `specs`, structural assumptions live in `structures`, and univariate sampling lives in `marginals`.

### Honest representations

The project avoids using a mathematically convenient representation when it would distort the intended meaning.
The clearest example is nonlinear redundancy, which is represented as group metadata rather than as a fabricated correlation matrix.

### Theoretical normalization when available

For Student-t and skew-normal marginals, the code uses theoretical moment corrections instead of standardizing the realized sample ex post.
That preserves the interpretation of the chosen distribution family.

### Small testable units

Implemented functions are intentionally pure and narrow.
This supports direct unit testing and reduces coupling across modules.

## Architectural Consequences

The current architecture has several practical consequences.

- It is easy to test each layer independently.
- It is easy to inspect the meaning of each function.
- It avoids committing too early to a generic engine abstraction.
- It pushes some semantic validation into runtime wrappers rather than centralizing all of it in the spec layer.

That last point is intentional for now.
For example, `MarginalSpec(kind="student_t")` is structurally valid in `specs.py`, while the requirement for `params["df"]` is enforced where sampling actually happens.

## Why the Architecture Matters for Research

Synthetic financial data generation is sensitive to assumptions about tails, asymmetry, clustering, and redundancy.
By splitting these assumptions into separate layers, the project makes it easier to answer questions such as:

- Which part of the pipeline introduces heavy tails?
- Which part creates structural similarity between assets?
- Which assumptions are represented exactly, and which are only approximated?
- Which extensions can be added without changing the entire system?

That separation is the main architectural objective of the repository.
