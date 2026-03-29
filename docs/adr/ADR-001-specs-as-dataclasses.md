# ADR-001: Specs As Dataclasses

## Status

Accepted

## Context

The project needs a stable way to describe synthetic data scenarios without coupling configuration to runtime generation logic.
Those descriptions must be:

- explicit
- typed
- easy to validate
- easy to test
- readable in both development and research contexts

The toolbox also distinguishes several conceptually different layers such as marginals, structures, copulas, and state processes.
Each layer requires its own specification object.

## Decision

Use small `dataclass` objects with `slots=True` as the core configuration interface.
The implemented specification classes are:

- `MarginalSpec`
- `CopulaSpec`
- `StructureSpec`
- `StateProcessSpec`
- `ScenarioSpec`

Each class performs structural validation in `__post_init__`.
Validation is intentionally limited to shape and allowed-kind checks at this layer.
More specific semantic validation is deferred to the module that actually consumes the spec.

## Alternatives Considered

### Plain dictionaries

Plain dictionaries would be flexible but would provide weaker guarantees, less discoverability, and less stable structure.
They would also encourage hidden validation rules spread across the codebase.

### A single generic spec object

A single generic spec object would reduce the number of classes but would blur the architectural boundaries between layers.
It would also make validation rules harder to reason about.

### Pydantic or another external schema library

A schema library could provide richer validation and serialization support.
However, it would add a dependency and push the project toward a broader framework before the core modeling decisions are stable.

## Consequences

### Positive

- Configuration is explicit and typed.
- Each layer has a named vocabulary.
- Structural validation happens early.
- Tests can target spec objects directly.
- The configuration layer stays independent from generation code.

### Negative

- Some semantic constraints are not enforced at spec construction time.
- The project must keep spec validation and runtime validation consistent across modules.

### Neutral

This decision keeps the configuration layer deliberately conservative.
Richer schema logic can still be introduced later if the project needs serialization or more centralized validation.
