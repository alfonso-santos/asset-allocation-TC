# Roadmap

## Completed Phases

### Phase 1: Specifications

Implemented in `specs.py`.
The project has typed dataclass-based specifications for:

- marginals
- copulas
- structures
- state processes
- full scenarios

This phase establishes the configuration vocabulary of the toolbox.

### Phase 2: Shared Utilities

Implemented in `utils.py`.
This phase provides reusable validation and matrix-checking helpers used by higher layers.

### Phase 3: Structures

Implemented in `structures.py`.
This phase currently covers:

- equicorrelation
- block correlation
- near-duplicates as a specialized block representation
- one-factor and multi-factor correlation structures
- nonlinear redundancy groups represented explicitly as metadata

### Phase 4: Marginals

Implemented in `marginals.py`.
This phase currently covers:

- standard normal marginals
- standardized Student-t marginals
- standardized skew-normal marginals
- heterogeneous marginal matrices driven by `MarginalSpec`

## Near-Term Work

### Phase 5: Copulas

The repository already reserves a `copulas.py` module and a `CopulaSpec` class, but copula sampling logic is not implemented yet.
Likely next work includes:

- independence copula behavior
- Gaussian copula sampling
- t copula sampling
- grouped or structured heavy-tail dependence
- parameter validation at the copula layer

### Phase 6: State Processes

The repository includes `StateProcessSpec` and a placeholder `states.py` module.
A natural next step is a minimal regime process, likely starting from the already reserved `markov` kind.

### Phase 7: Scenario Assembly

Once marginals, copulas, and state processes exist, the toolbox can add scenario assembly logic that maps `ScenarioSpec` into a complete synthetic dataset.
This belongs in `generator.py` and `scenarios.py`.

## Medium-Term Work

The following capabilities are structurally anticipated by the repository but not implemented yet.

- diagnostics for checking generated outputs against intended assumptions
- plotting helpers for inspecting synthetic scenarios
- specialized generator families built on top of the base layers
- more structured integration between scenario definitions and generation outputs

## Work Explicitly Deferred

Some features are intentionally not part of the current phase.

- generic dispatch systems spanning the whole toolbox
- hidden orchestration layers that bypass the explicit module boundaries
- fake linear representations for nonlinear structural effects
- multivariate dependence logic inside the marginal layer

## Documentation Expectations for Future Phases

As the toolbox grows, each completed phase should add:

- unit tests for its public API
- at least one ADR when a durable design decision is introduced
- module-level documentation aligned with the real implementation state

## Current Development Baseline

As of the current codebase state:

- `specs`, `utils`, `structures`, and `marginals` are implemented and tested
- higher-level modules are placeholders
- the project is ready for the first dependence-layer implementation

That makes the next architectural milestone the transition from independent or column-wise generation to explicit multivariate dependence.
