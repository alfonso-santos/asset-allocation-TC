# asset-allocation-TC

Base de proyecto Python con estructura moderna, entorno virtual local y
estructura orientada a experimentacion reproducible para un paper.

## Estructura

```text
.
|-- .venv/
|-- configs/
|   `-- README.md
|-- data/
|   |-- external/
|   |   `-- README.md
|   |-- processed/
|   |   `-- README.md
|   |-- raw/
|   |   `-- README.md
|   `-- README.md
|-- docs/
|   `-- README.md
|-- experiments/
|   `-- README.md
|-- notebooks/
|   `-- README.md
|-- paper/
|   |-- drafts/
|   |   `-- README.md
|   |-- figures/
|   |   `-- README.md
|   |-- references/
|   |   `-- README.md
|   `-- README.md
|-- results/
|   |-- figures/
|   |   `-- README.md
|   |-- logs/
|   |   `-- README.md
|   |-- tables/
|   |   `-- README.md
|   `-- README.md
|-- scripts/
|   `-- README.md
|-- src/
|   `-- asset_allocation_tc/
|       |-- __init__.py
|       |-- cli.py
|       `-- __main__.py
|-- tests/
|   |-- __init__.py
|   `-- test_smoke.py
`-- pyproject.toml
```

## Carpetas de investigacion

- `data/raw`: datos originales sin modificar.
- `data/processed`: datos limpios o transformados para los experimentos.
- `data/external`: datos obtenidos de fuentes externas.
- `configs`: configuraciones de experimentos, seeds y variantes.
- `experiments`: scripts o definiciones de ejecucion.
- `notebooks`: exploracion, analisis y prototipos.
- `results/figures`: figuras generadas automaticamente por los experimentos.
- `results/tables`: tablas exportadas desde los analisis.
- `results/logs`: logs y trazas de ejecucion.
- `paper/figures`: figuras finales seleccionadas para el manuscrito.
- `paper/drafts`: borradores del paper.
- `paper/references`: bibliografia, archivos `.bib` y notas de citas.

La separacion entre `results/figures` y `paper/figures` ayuda a mantener la
reproducibilidad: una carpeta guarda lo generado por el codigo y la otra lo que
termina entrando en el articulo.

## Primeros pasos

### PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
pytest
python -m asset_allocation_tc
```
