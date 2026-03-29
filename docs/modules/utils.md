# Module: `utils`

## Purpose

The `tc_synthetic.utils` module provides low-level helpers shared by the rest of the toolbox.
It is the current validation foundation of the implemented codebase.

## Scope

This module currently provides:

- random generator creation
- validation of positive integer counts for assets and observations
- validation of square matrices
- validation of symmetry
- validation of unit diagonal
- validation of positive semidefiniteness

It does not contain domain-specific scenario logic.

## Public API

- `make_rng(seed: int | None = None) -> np.random.Generator`
- `validate_n_assets(n_assets: int) -> int`
- `validate_n_obs(n_obs: int) -> int`
- `validate_square_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray`
- `validate_symmetric_matrix(matrix: np.ndarray, name: str = "matrix", atol: float = 1e-8) -> np.ndarray`
- `validate_unit_diagonal(matrix: np.ndarray, name: str = "matrix", atol: float = 1e-8) -> np.ndarray`
- `validate_positive_semidefinite_matrix(matrix: np.ndarray, name: str = "matrix", atol: float = 1e-8) -> np.ndarray`

## Design Decisions

### Validation helpers return the validated input

Matrix validators return the original `np.ndarray` so they can be composed inline and used in assertions.

### Bool values are rejected for integer-like parameters

Inputs such as `True` are explicitly rejected for `n_assets`, `n_obs`, and similar fields, even though `bool` is a subclass of `int` in Python.

### Numerical tolerance is explicit

Matrix validations use a configurable absolute tolerance and reject invalid tolerance values through `_validate_atol`.

## Mathematical Assumptions

### Symmetry

Symmetry is checked with `numpy.allclose` using absolute tolerance only.
This avoids mixing absolute and relative error logic in a context where exact matrix structure matters.

### Positive semidefiniteness

Positive semidefiniteness is checked through the eigenvalues returned by `numpy.linalg.eigvalsh`.
The matrix is considered invalid if any eigenvalue is below `-atol`.

## Examples

### Reproducible generator

```python
from tc_synthetic.utils import make_rng

rng = make_rng(123)
```

### Observation count validation

```python
from tc_synthetic.utils import validate_n_obs

n_obs = validate_n_obs(250)
```

### Matrix validation chain

```python
import numpy as np
from tc_synthetic.utils import (
    validate_positive_semidefinite_matrix,
    validate_square_matrix,
    validate_symmetric_matrix,
    validate_unit_diagonal,
)

corr = np.array([[1.0, 0.3], [0.3, 1.0]])
validate_square_matrix(corr, name="correlation")
validate_symmetric_matrix(corr, name="correlation")
validate_unit_diagonal(corr, name="correlation")
validate_positive_semidefinite_matrix(corr, name="correlation")
```

## Testing

The test suite checks:

- random generator construction
- type validation for positive integer helpers
- matrix-shape validation
- tolerance validation
- PSD detection

Typical command:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_utils.py
```

## Extension Guidelines

When adding helpers to `utils.py`:

1. Keep them generic and reusable.
2. Avoid embedding domain-specific terminology unless it is truly cross-cutting.
3. Reject ambiguous Python values such as booleans when they would weaken validation.
4. Add focused unit tests for every branch.

Do not move higher-level scenario or sampling logic into this module.
