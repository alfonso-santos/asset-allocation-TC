# Contributing

## Scope of This Guide

This guide describes how to contribute to `tc_synthetic` in its current stage.
The project is still building out its core layers, so contributions should prioritize clarity, testability, and explicit design decisions.

## Development Workflow

### 1. Sync your environment

```powershell
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

### 2. Make a focused change

Prefer narrow changes that affect one layer at a time.
Examples:

- add a utility validator
- add a new structure builder
- add a new marginal family
- extend tests for an existing public function
- add or update documentation for a design decision

### 3. Run tests

```powershell
.venv\Scripts\python.exe -m pytest
```

### 4. Run linting when appropriate

```powershell
.venv\Scripts\python.exe -m ruff check .
```

## Contribution Principles

### Keep layers separate

Do not move cross-sectional logic into `marginals.py`.
Do not move sampling logic into `specs.py`.
Do not move scenario orchestration into `utils.py`.

### Prefer explicit functions over hidden abstraction

If a new feature can be expressed as a small public function with a clear name, prefer that over introducing a generic dispatcher too early.

### Be honest about mathematical representation

Do not force a concept into a correlation matrix or a standardized distribution if that would misrepresent what the code actually models.

### Reject ambiguous inputs explicitly

Follow the existing validation style:

- reject `bool` where a numeric or integer value is required
- use `TypeError` for wrong types
- use `ValueError` for invalid values
- keep messages short and clear

### Document durable decisions

If a change introduces a durable design choice, add or update an ADR.
The ADR should explain why the decision was taken, not just what changed.

## Coding Expectations

- Use English for code-facing documentation.
- Keep public APIs small and testable.
- Add full type hints.
- Add docstrings in the style already used by the codebase.
- Import only what is needed.
- Avoid adding dependencies unless they are justified by the mathematical requirement.

## Testing Expectations

A contribution is not complete if it changes public behavior without updating tests.
At minimum, add tests for:

- successful behavior
- invalid inputs
- exact formulas when applicable
- reproducibility under fixed seeds for stochastic code

## Documentation Expectations

Update documentation when a contribution changes:

- the public API of an implemented module
- the roadmap
- a durable architectural or mathematical decision

## What Not To Do

- Do not document placeholder modules as fully implemented systems.
- Do not add features that bypass the existing module boundaries.
- Do not introduce broad framework abstractions unless there is a clear repeated need.
- Do not invent future capabilities in user-facing documentation.

## Suggested Commit Scope

Good contribution scopes are small and reviewable.
Examples:

- one new marginal plus its tests and documentation
- one new structure builder plus its tests and ADR if needed
- one packaging or documentation cleanup

This repository benefits from incremental progress rather than large unstructured refactors.
