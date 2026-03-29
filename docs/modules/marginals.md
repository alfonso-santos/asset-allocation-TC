# Module: `marginals`

## Purpose

The `tc_synthetic.marginals` module implements the univariate sampling layer of the toolbox.
It provides both direct marginal generators and a wrapper that builds heterogeneous marginal matrices from `MarginalSpec` objects.

## Scope

The module currently implements:

- empirical standardization of a 1D sample
- standard normal marginal sampling
- standardized Student-t marginal sampling
- standardized skew-normal marginal sampling
- heterogeneous column-wise marginal assembly

It does not implement copula-driven dependence.
It does not perform scenario-level orchestration.

## Public API

- `standardize_1d_sample(sample: np.ndarray) -> np.ndarray`
- `sample_standard_normal_marginal(n_obs: int, rng: np.random.Generator) -> np.ndarray`
- `sample_standardized_student_t_marginal(n_obs: int, df: float, rng: np.random.Generator) -> np.ndarray`
- `sample_standardized_skew_normal_marginal(n_obs: int, shape: float, rng: np.random.Generator) -> np.ndarray`
- `sample_heterogeneous_marginals(n_obs: int, specs: list[MarginalSpec], rng: np.random.Generator) -> np.ndarray`

## Design Decisions

### Separate function per marginal family

Each supported marginal has a dedicated function.
The module avoids a hidden general dispatcher and keeps the distribution-specific logic visible.

### Theoretical standardization when moments are known

Student-t and skew-normal samples are standardized through analytical formulas, not by standardizing the realized sample.
This keeps the sampled family interpretable.

### Heterogeneous wrapper is local and explicit

`sample_heterogeneous_marginals` is the first place where `MarginalSpec` is consumed by a generation function.
The wrapper supports only:

- `gaussian`
- `student_t`
- `skew_normal`

It explicitly rejects nested `heterogeneous` specs.

### Shared RNG across columns

The heterogeneous wrapper uses the same `np.random.Generator` object sequentially for all columns.
This makes the generated matrix reproducible and consistent with explicit column-wise generation.

## Mathematical Assumptions

### Empirical standardization

`standardize_1d_sample` uses:

- sample mean
- population standard deviation with `ddof=0`

### Standardized Student-t

For `T ~ t_df` with `df > 2`:

`Var(T) = df / (df - 2)`

The standardized output is:

`X = T * sqrt((df - 2) / df)`

### Standardized skew-normal

For skew-normal shape parameter `shape`, define:

- `delta = shape / sqrt(1 + shape**2)`
- `mu = delta * sqrt(2 / pi)`
- `sigma = sqrt(1 - 2 * delta**2 / pi)`

If `raw_sample` is drawn from `skewnorm(a=shape, loc=0, scale=1)`, the standardized output is:

`(raw_sample - mu) / sigma`

## Examples

### Standard normal marginal

```python
import numpy as np
from tc_synthetic.marginals import sample_standard_normal_marginal

rng = np.random.default_rng(123)
sample = sample_standard_normal_marginal(1000, rng)
```

### Standardized Student-t

```python
import numpy as np
from tc_synthetic.marginals import sample_standardized_student_t_marginal

rng = np.random.default_rng(123)
sample = sample_standardized_student_t_marginal(1000, df=5.0, rng=rng)
```

### Standardized skew-normal

```python
import numpy as np
from tc_synthetic.marginals import sample_standardized_skew_normal_marginal

rng = np.random.default_rng(123)
sample = sample_standardized_skew_normal_marginal(1000, shape=4.0, rng=rng)
```

### Heterogeneous marginal matrix

```python
import numpy as np
from tc_synthetic.marginals import sample_heterogeneous_marginals
from tc_synthetic.specs import MarginalSpec

rng = np.random.default_rng(123)
specs = [
    MarginalSpec(kind="gaussian"),
    MarginalSpec(kind="student_t", params={"df": 5.0}),
    MarginalSpec(kind="skew_normal", params={"shape": 4.0}),
]

matrix = sample_heterogeneous_marginals(500, specs, rng)
```

## Testing

The test suite checks:

- input validation for every public function
- exact reproducibility under fixed seeds
- exact equality with theoretical transformations for Student-t and skew-normal
- exact equality between the heterogeneous wrapper and explicit column-wise generation

Typical command:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_marginals.py
```

## Extension Guidelines

When adding a new marginal family:

1. Implement a dedicated public generator first.
2. Use theoretical normalization when closed-form moments are available.
3. Keep empirical standardization separate from family-specific generation.
4. Extend `sample_heterogeneous_marginals` only after the dedicated generator exists and is tested.
5. Reject unsupported nested wrappers explicitly.

Do not move dependence logic into this module.
The marginal layer should remain univariate or column-wise.
