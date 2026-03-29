# ADR-005: Nonlinear Redundancy Groups

## Status

Accepted

## Context

The project includes a structural concept called `nonlinear_redundancy`.
This concept is not equivalent to simple linear correlation.
A naive implementation could force it into a correlation matrix, but that would imply a stronger and more specific representation than the project can currently justify.

The design goal of the toolbox is to remain explicit and honest about representation limits.

## Decision

Represent nonlinear redundancy as validated group metadata instead of a fabricated matrix.
The current representation is a dictionary with:

- `groups`
- `n_assets`
- `strength`

The builder validates index ranges, uniqueness across groups, and the strength parameter in `[0, 1]`.
It does not attempt to construct synthetic data or a correlation matrix.

## Alternatives Considered

### Correlation matrix approximation

A matrix approximation would make the interface superficially uniform with other structures, but it would also incorrectly suggest that nonlinear redundancy is fully captured by linear second-order dependence.

### Deferring the feature entirely

The project could have postponed `nonlinear_redundancy` until a full generative mechanism existed.
This was rejected because a validated structural placeholder is still useful and documents the architectural intent.

### Embedding nonlinear redundancy as arbitrary free-form params

That would reduce implementation effort but would lose a clean explicit representation and weaken testing.

## Consequences

### Positive

- The representation is honest about current model scope.
- Validation is still possible even without generation logic.
- Future nonlinear generators can consume the same explicit group structure.

### Negative

- The output shape differs from matrix-based structure builders.
- Downstream code must branch on representation type until a higher-level structure interface is introduced.

### Neutral

This decision favors semantic clarity over uniform return types.
That is consistent with the project philosophy at the current stage.
