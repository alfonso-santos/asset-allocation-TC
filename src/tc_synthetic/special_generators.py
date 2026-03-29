"""Modulo de generadores especiales del toolbox de datos sinteticos."""

import numpy as np

from tc_synthetic.generator import apply_marginal_transform
from tc_synthetic.specs import MarginalSpec, StructureSpec
from tc_synthetic.utils import validate_n_assets, validate_n_obs

__all__ = ["generate_nonlinear_redundancy_data", "generate_special_structure_data"]



def _validate_group_sizes(group_sizes: object, n_assets: int) -> list[int]:
    """Valida una lista de tamanos de grupos positivos."""
    if not isinstance(group_sizes, list):
        raise TypeError("group_sizes must be a list")
    if not group_sizes:
        raise ValueError("group_sizes must not be empty")

    validated_group_sizes: list[int] = []
    for group_size in group_sizes:
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise TypeError("group_sizes must contain integers")
        if group_size <= 0:
            raise ValueError("group_sizes must contain positive integers")
        validated_group_sizes.append(group_size)

    if sum(validated_group_sizes) != n_assets:
        raise ValueError("sum of group_sizes must match n_assets")
    return validated_group_sizes



def _column_to_uniform(values: np.ndarray) -> np.ndarray:
    """Convierte una columna continua en uniforms mediante ranks empiricos."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    return (ranks - 0.5) / values.shape[0]



def generate_nonlinear_redundancy_data(
    n_obs: int,
    n_assets: int,
    structure: StructureSpec,
    marginal: MarginalSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera datos con redundancia no lineal por grupos y ajusta la marginal por columna."""
    n_obs = validate_n_obs(n_obs)
    n_assets = validate_n_assets(n_assets)
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be a StructureSpec")
    if not isinstance(marginal, MarginalSpec):
        raise TypeError("marginal must be a MarginalSpec")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if structure.kind != "nonlinear_redundancy":
        raise ValueError("special generator requires a nonlinear_redundancy structure")
    if "group_sizes" not in structure.params:
        raise ValueError("nonlinear_redundancy structure requires 'group_sizes'")

    group_sizes = _validate_group_sizes(structure.params["group_sizes"], n_assets)
    transforms = [
        lambda z: z,
        lambda z: z**2,
        lambda z: np.tanh(z),
        lambda z: np.sin(z),
    ]
    columns: list[np.ndarray] = []

    for group_size in group_sizes:
        z = rng.standard_normal(size=n_obs)
        for column_index in range(group_size):
            transform = transforms[column_index % len(transforms)]
            noise = 0.05 * rng.standard_normal(size=n_obs)
            columns.append(transform(z) + noise)

    base = np.column_stack(columns).astype(float, copy=False)
    if base.shape != (n_obs, n_assets):
        raise ValueError("generated data shape must match (n_obs, n_assets)")

    uniforms = np.column_stack(
        [_column_to_uniform(base[:, column_index]) for column_index in range(n_assets)]
    ).astype(float, copy=False)
    x = apply_marginal_transform(marginal, uniforms)
    if x.shape != (n_obs, n_assets):
        raise ValueError("generated data shape must match (n_obs, n_assets)")
    return x.astype(float, copy=False)


def generate_special_structure_data(
    n_obs: int,
    n_assets: int,
    structure: StructureSpec,
    marginal: MarginalSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Despacha la generacion para estructuras especiales soportadas."""
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be a StructureSpec")
    if not isinstance(marginal, MarginalSpec):
        raise TypeError("marginal must be a MarginalSpec")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if structure.kind == "nonlinear_redundancy":
        return generate_nonlinear_redundancy_data(n_obs, n_assets, structure, marginal, rng)
    raise ValueError(f"unsupported special structure kind: {structure.kind}")
