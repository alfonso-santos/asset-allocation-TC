# ADR-003: Standardized Student-t Marginal

## Status

Accepted

## Context

The project needs a heavy-tailed univariate marginal family as part of its closed domain of supported marginals.
Student-t is a natural choice because it preserves symmetry while allowing explicit tail thickness through `df`.

However, a raw Student-t distribution with `df > 2` does not have unit variance.
For `T ~ t_df`:

`Var(T) = df / (df - 2)`

The toolbox wants standardized marginals with comparable scale across families.

## Decision

Implement the Student-t marginal using theoretical variance correction instead of empirical sample standardization.
The sampling rule is:

1. Generate `raw = rng.standard_t(df, size=n_obs)`.
2. Compute `scale = sqrt((df - 2) / df)`.
3. Return `raw * scale`.

The function requires `df > 2` and rejects invalid `rng` inputs.

## Alternatives Considered

### Empirical standardization with sample mean and sample standard deviation

This would force every realized sample to have approximately mean zero and variance one.
It was rejected because it changes the distribution in a sample-dependent way and hides the theoretical relationship between the chosen family and its moments.

### Leaving the raw Student-t scale unchanged

This would preserve the raw distribution but would make cross-family comparison harder because the variance would depend on `df`.

### Allowing `df <= 2`

This would make the theoretical variance undefined or infinite, which is incompatible with the project goal of standardized marginals.

## Consequences

### Positive

- The Student-t marginal has unit variance by construction.
- The transformation is deterministic and theoretically motivated.
- The distribution family remains interpretable through `df`.

### Negative

- The standardization only addresses variance, not realized sample moments.
- Users who expect empirical standardization must apply it separately and explicitly.

### Neutral

This decision establishes a pattern for other marginals: when closed-form moments are known and useful, the toolbox prefers theoretical normalization over ex post sample correction.
