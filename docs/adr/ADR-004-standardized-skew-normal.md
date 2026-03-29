# ADR-004: Standardized Skew-Normal Marginal

## Status

Accepted

## Context

The project needs an asymmetric univariate marginal family that remains analytically tractable.
The skew-normal distribution is a good fit because it extends the Gaussian family with a shape parameter while preserving closed-form expressions for mean and variance.

A raw skew-normal with parameters `a = shape`, `loc = 0`, and `scale = 1` does not in general have mean zero or variance one.

## Decision

Use `scipy.stats.skewnorm` as the base generator and apply theoretical moment correction.
For a given `shape`, define:

- `delta = shape / sqrt(1 + shape**2)`
- `mu = delta * sqrt(2 / pi)`
- `sigma = sqrt(1 - 2 * delta**2 / pi)`

The sampling rule is:

1. Generate `raw_sample = skewnorm.rvs(a=shape, loc=0, scale=1, size=n_obs, random_state=rng)`.
2. Return `(raw_sample - mu) / sigma`.

The function requires a finite numeric `shape` and a valid `np.random.Generator`.

## Alternatives Considered

### Empirical standardization after sampling

This would produce a standardized realized sample, but it would remove the distinction between theoretical distribution choice and sample-dependent rescaling.
It was rejected for the same reason as empirical standardization of Student-t.

### Custom skew-normal implementation without SciPy

A custom implementation was unnecessary because SciPy already provides a robust and well-known reference implementation.

### Using a non-standardized skew-normal output

This would preserve the raw family but would make comparisons with other standardized marginals less direct.

## Consequences

### Positive

- The marginal keeps the skew-normal shape while being centered and scaled analytically.
- The implementation is concise and grounded in closed-form moments.
- The project gets an asymmetric standardized marginal without inventing a custom distribution.

### Negative

- The implementation introduces a dependency on `scipy.stats.skewnorm`.
- The standardized output is no longer the raw SciPy parameterization.

### Neutral

The use of SciPy is limited and explicit.
The toolbox still exposes a small local API rather than leaking the SciPy interface directly.
