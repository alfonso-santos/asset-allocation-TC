# Module: `structures`

## Purpose

The `tc_synthetic.structures` module defines cross-sectional structural assumptions used by the toolbox.
Its outputs are either:

- validated correlation matrices
- validated metadata representations when a matrix would be misleading

## Scope

The module currently implements:

- equicorrelation matrices
- block correlation matrices
- near-duplicate correlation structures
- one-factor correlation matrices
- multi-factor correlation matrices
- nonlinear redundancy groups

It does not yet implement a generic `StructureSpec` dispatcher.
It does not generate synthetic returns by itself.

## Public API

- `build_equicorrelation_matrix(n_assets: int, rho: float) -> np.ndarray`
- `build_block_correlation_matrix(block_sizes: list[int], rho_within: float, rho_between: float) -> np.ndarray`
- `build_near_duplicate_correlation_matrix(group_sizes: list[int], rho_duplicate: float, rho_background: float = 0.0) -> np.ndarray`
- `build_factor_correlation_matrix(loadings: np.ndarray) -> np.ndarray`
- `build_one_factor_correlation_matrix(loadings: np.ndarray) -> np.ndarray`
- `build_nonlinear_redundancy_groups(groups: list[list[int]], n_assets: int, strength: float) -> dict[str, object]`

## Design Decisions

### Matrix builders validate the matrix they produce

Builders that return correlation matrices validate:

- square shape
- symmetry
- unit diagonal
- positive semidefiniteness

This turns the output into a checked representation rather than a raw construction.

### Near-duplicates reuse the block model

Near-duplicate groups are treated as a specialized case of block correlation.
The public API preserves domain vocabulary while reusing the block implementation.

### Nonlinear redundancy is not forced into a matrix

The module represents nonlinear redundancy as group metadata because the current code cannot justify a faithful matrix representation for that concept.

### One-factor is a specialization of multi-factor

`build_one_factor_correlation_matrix` delegates to the general factor builder after reshaping the loadings.
This avoids duplication.

## Mathematical Assumptions

### Equicorrelation

For `n_assets > 1`, the lower bound on `rho` is `-1 / (n_assets - 1)`.
This is the standard condition for the equicorrelation matrix to remain positive semidefinite.

### Factor correlation model

The factor builders assume that each row of loadings satisfies:

`sum(loadings_i**2) <= 1`

The returned matrix is:

`beta @ beta.T + diag(1 - row_norms_sq)`

This guarantees a unit diagonal when the row norm condition holds.

### Near-duplicate blocks

Near-duplicates are modeled as blocks with high within-group correlation and lower background correlation.
This is a linear approximation of similarity, not a nonlinear redundancy model.

## Examples

### Equicorrelation

```python
from tc_synthetic.structures import build_equicorrelation_matrix

corr = build_equicorrelation_matrix(n_assets=4, rho=0.25)
```

### Block structure

```python
from tc_synthetic.structures import build_block_correlation_matrix

corr = build_block_correlation_matrix(block_sizes=[2, 3], rho_within=0.6, rho_between=0.1)
```

### Near-duplicates

```python
from tc_synthetic.structures import build_near_duplicate_correlation_matrix

corr = build_near_duplicate_correlation_matrix(group_sizes=[2, 2], rho_duplicate=0.95, rho_background=0.1)
```

### Nonlinear redundancy groups

```python
from tc_synthetic.structures import build_nonlinear_redundancy_groups

structure = build_nonlinear_redundancy_groups(
    groups=[[0, 1], [3, 4]],
    n_assets=6,
    strength=0.8,
)
```

## Testing

The test suite checks:

- value and type validation
- exact matrix construction in representative cases
- consistency between specialized and general builders
- PSD enforcement for matrix-based builders
- explicit nonlinear redundancy group validation

Typical command:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_structures.py
```

## Extension Guidelines

When adding a new structure:

1. Decide whether the structure is honestly representable as a matrix.
2. If it is matrix-based, validate the produced matrix before returning it.
3. If it is not matrix-based, return an explicit structured representation instead of forcing a matrix abstraction.
4. Keep builder functions small and testable.

Do not introduce a generic structure dispatcher until multiple higher-level modules need it.
