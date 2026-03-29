# ADR-002: Layered Architecture

## Status

Accepted

## Context

Synthetic financial return generation combines several different concerns:

- scenario declaration
- input validation
- structural modeling
- marginal distribution choice
- dependence modeling
- state or regime logic
- reporting and diagnostics

If these concerns are merged too early, the code becomes difficult to test and difficult to extend.
The repository was designed from the beginning as a multi-module toolbox rather than a single generator function.

## Decision

Adopt a layered module architecture in which each module owns one primary concern.
The current implemented layers are:

- `specs`
- `utils`
- `structures`
- `marginals`

Additional layers are reserved by placeholder modules, including:

- `copulas`
- `states`
- `generator`
- `scenarios`
- `diagnostics`
- `plots`
- `special_generators`

Dependencies are kept directional.
Lower layers do not depend on higher-level orchestration modules.

## Alternatives Considered

### Monolithic generator module

A single module could appear simpler at the beginning.
However, it would quickly accumulate mixed responsibilities and make testing less precise.

### Framework-style dispatcher from day one

A generic dispatcher layer could unify all sampling paths early.
This was rejected because the project does not yet need that abstraction, and premature dispatch systems tend to hide design mistakes.

### Separate repositories per concern

Splitting layers into multiple repositories would make boundaries explicit but would add packaging and coordination overhead too early.

## Consequences

### Positive

- Individual layers are easy to test in isolation.
- Design decisions are easier to document.
- Extensions can be added without refactoring unrelated modules.
- Mathematical assumptions remain local to the relevant module.

### Negative

- Some cross-layer workflows are not implemented yet.
- The repository contains placeholder modules that may look incomplete until later phases are added.

### Neutral

The architecture favors clarity over early convenience.
As the project matures, orchestration logic can be added without changing the role of foundational modules.
