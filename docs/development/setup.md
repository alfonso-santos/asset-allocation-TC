# Development Setup

## Prerequisites

The current project setup assumes:

- Python 3.11 or newer
- `venv`
- PowerShell on Windows for the commands shown below

The package is installed from the local repository in editable mode.

## Create and Activate the Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Upgrade Packaging Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## Install the Project and Development Dependencies

```powershell
pip install -e .[dev]
```

## Verify the Installation

Check that the package imports correctly:

```powershell
python -c "import tc_synthetic; print(tc_synthetic.__version__)"
```

Check that the test runner works:

```powershell
python -m pytest
```

## Repository Layout Relevant for Development

- `src/tc_synthetic/`: package source code
- `tests/`: unit and smoke tests
- `docs/`: project documentation
- `pyproject.toml`: packaging and test configuration

## Optional: Run a Single Test File

```powershell
python -m pytest tests\test_marginals.py
```

## Notes

The editable installation is important during development because it allows the test suite to use the live source tree under `src/`.
The test configuration in `pyproject.toml` also adds `src` to `pythonpath`.
