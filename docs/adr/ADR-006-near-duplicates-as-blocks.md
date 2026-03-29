# ADR-006: Near-Duplicates As Blocks

## Status

Accepted

## Context

The project includes a structure called `near_duplicates`, meant to represent groups of assets that are almost redundant in a linear sense.
Mathematically, this is closely aligned with a block correlation structure in which:

- within-group correlation is high
- between-group correlation is lower

The project already has a validated block correlation builder.

## Decision

Implement near-duplicates as a thin specialized wrapper around the block correlation builder.
The near-duplicate API uses domain-specific parameter names:

- `group_sizes`
- `rho_duplicate`
- `rho_background`

Internally it validates the domain-specific relation `rho_duplicate >= rho_background` and delegates matrix construction to `build_block_correlation_matrix`.

## Alternatives Considered

### Separate full implementation

A dedicated implementation would duplicate matrix-building logic and validation that already exists in the block builder.

### Reusing block correlation without a separate public function

This would reduce API surface, but it would remove a useful domain term from the toolbox vocabulary.
The project wants the code to express research concepts directly where appropriate.

### Modeling near-duplicates as exact duplicates

This would be too rigid and would remove the ability to control background correlation explicitly.

## Consequences

### Positive

- The code reuses an already tested structural primitive.
- The public API preserves a domain-relevant concept.
- Validation rules remain small and explicit.

### Negative

- The near-duplicate builder is conceptually separate while being mathematically identical to a block wrapper in its current form.

### Neutral

This decision keeps the project vocabulary close to the research domain while avoiding unnecessary implementation duplication.
Future phases can still add specialized behavior if near-duplicates require more than a simple block representation.
