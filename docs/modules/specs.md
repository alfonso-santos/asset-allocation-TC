# Module: `specs`

## Purpose

The `tc_synthetic.specs` module defines the declarative configuration layer of the toolbox.
Its job is to describe what a synthetic scenario should contain, not to generate data directly.

## Scope

This module currently provides:

- allowed-kind registries for the main conceptual layers
- dataclass-based specification objects
- basic structural validation in `__post_init__`

It does not perform generation.
It does not validate all semantic details required by downstream modules.

## Public API

### Constants

- `ALLOWED_MARGINAL_KINDS`
- `ALLOWED_COPULA_KINDS`
- `ALLOWED_STRUCTURE_KINDS`
- `ALLOWED_STATE_PROCESS_KINDS`

### Dataclasses

- `MarginalSpec`
- `CopulaSpec`
- `StructureSpec`
- `StateProcessSpec`
- `ScenarioSpec`

## Design Decisions

### Dataclasses with slots

The module uses `dataclass(slots=True)` to keep specification objects lightweight and explicit.

### Structural validation only

Validation at this layer is intentionally limited to:

- type checks
- non-empty string checks
- allowed-kind checks
- basic positive-integer checks for scenario dimensions

For example, `MarginalSpec(kind="student_t")` is structurally valid even if `params` does not yet contain `df`.
That requirement is enforced where the spec is actually consumed.

### Separate classes per conceptual layer

Marginals, copulas, structures, and state processes are represented by different classes instead of one generic spec object.
This keeps module boundaries explicit.

## Mathematical Assumptions

The module itself does not encode mathematical formulas.
Its role is categorical: it defines the allowed conceptual vocabulary used by other layers.

## Examples

### Marginal specification

```python
from tc_synthetic.specs import MarginalSpec

spec = MarginalSpec(kind="student_t", params={"df": 5.0})
```

### Structure specification

```python
from tc_synthetic.specs import StructureSpec

spec = StructureSpec(kind="equicorrelation", params={"rho": 0.3})
```

### Scenario specification

```python
from tc_synthetic.specs import (
    CopulaSpec,
    MarginalSpec,
    ScenarioSpec,
    StateProcessSpec,
    StructureSpec,
)

scenario = ScenarioSpec(
    name="baseline",
    n_assets=5,
    n_obs=1000,
    marginal=MarginalSpec(kind="gaussian"),
    copula=CopulaSpec(kind="independence"),
    structure=StructureSpec(kind="equicorrelation"),
    state_process=StateProcessSpec(kind="markov", enabled=False),
)
```

## Testing

The test suite checks:

- allowed-kind enforcement
- type validation
- empty-string rejection
- scenario-level field validation
- module import behavior

Typical command:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_specs.py
```

## Extension Guidelines

When adding a new spec kind:

1. Add the new kind to the relevant allowed-kind set.
2. Decide whether only structural validation belongs in `specs.py`.
3. Keep semantic validation in the consuming module unless there is a strong reason to centralize it.
4. Extend tests for both the allowed-kind registry and the corresponding dataclass.

Do not add generation logic to this module.
