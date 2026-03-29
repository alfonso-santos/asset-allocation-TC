# Testing

## Test Philosophy

The current project favors small, direct unit tests over broad integration tests.
This matches the present architecture, where the implemented layers are:

- specifications
- utilities
- structures
- marginals

Each public function is expected to have focused tests covering both valid behavior and input validation.

## Main Command

Run the full test suite with the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest
```

## Run Individual Test Files

### Specs

```powershell
.venv\Scripts\python.exe -m pytest tests\test_specs.py
```

### Utils

```powershell
.venv\Scripts\python.exe -m pytest tests\test_utils.py
```

### Structures

```powershell
.venv\Scripts\python.exe -m pytest tests\test_structures.py
```

### Marginals

```powershell
.venv\Scripts\python.exe -m pytest tests\test_marginals.py
```

## Import and Smoke Checks

The repository also includes lightweight checks for:

- package importability
- module importability
- package version exposure

Useful commands:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_module_imports.py
.venv\Scripts\python.exe -m pytest tests\test_smoke.py
```

## Syntax Check

For a quick syntax pass without executing tests:

```powershell
python -m py_compile src\tc_synthetic\specs.py src\tc_synthetic\utils.py src\tc_synthetic\structures.py src\tc_synthetic\marginals.py
```

## What the Tests Currently Emphasize

The existing tests focus on:

- input validation and error types
- exact output shape and dtype
- reproducibility under fixed seeds
- exact equality with known formulas where the implementation is deterministic
- consistency between specialized wrappers and the more general building blocks they reuse

## Interpreting Current Warnings

In the current Windows environment, `pytest` may emit a warning related to `.pytest_cache` creation.
That warning does not indicate a logic error in the package itself.
It should be treated separately from functional test failures.

## Expectations for New Code

When adding a new public function:

1. Add positive-path tests.
2. Add validation tests for wrong types and invalid values.
3. Add exact equality tests when the implementation follows a closed-form formula.
4. Run the full suite before considering the change complete.
