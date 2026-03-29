"""Modulo generador del toolbox de datos sinteticos."""

import numpy as np

from tc_synthetic.copulas import (
    sample_clayton_copula,
    sample_gaussian_copula,
    sample_grouped_t_copula,
    sample_independence_copula,
    sample_t_copula,
)
from tc_synthetic.marginals import (
    apply_standard_normal_inverse_cdf,
    apply_standardized_skew_normal_inverse_cdf,
    apply_standardized_student_t_inverse_cdf,
)
from tc_synthetic.scenarios import validate_two_state_scenario
from tc_synthetic.specs import CopulaSpec, MarginalSpec, ScenarioSpec, StateProcessSpec, StructureSpec
from tc_synthetic.states import sample_markov_states
from tc_synthetic.structures import (
    build_block_correlation_matrix,
    build_equicorrelation_matrix,
    build_factor_correlation_matrix,
    build_near_duplicate_correlation_matrix,
)
from tc_synthetic.utils import validate_n_assets, validate_n_obs

__all__ = [
    "resolve_structure_correlation",
    "sample_uniform_from_copula",
    "apply_marginal_transform",
    "generate_static_scenario_data",
    "generate_two_state_scenario_data",
]


def _sample_state_path(
    state_spec: StateProcessSpec,
    n_obs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Muestrea una trayectoria de estados a partir de un ``StateProcessSpec``."""
    if not isinstance(state_spec, StateProcessSpec):
        raise TypeError("state_spec must be a StateProcessSpec")
    if not state_spec.enabled:
        raise ValueError("state process must be enabled")
    if state_spec.kind != "markov":
        raise ValueError("only 'markov' state processes are supported")
    if "transition_matrix" not in state_spec.params:
        raise ValueError("state_spec.params must include 'transition_matrix'")
    if "initial_state" not in state_spec.params:
        raise ValueError("state_spec.params must include 'initial_state'")

    return sample_markov_states(
        n_obs,
        state_spec.params["transition_matrix"],
        state_spec.params["initial_state"],
        rng,
    )



def _generate_two_state_observationwise_data(
    calm_spec: ScenarioSpec,
    crisis_spec: ScenarioSpec,
    state_spec: StateProcessSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera datos de dos estados observacion a observacion segun la trayectoria de estados."""
    validate_two_state_scenario(calm_spec, crisis_spec, state_spec)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if calm_spec.n_obs != crisis_spec.n_obs:
        raise ValueError("calm and crisis scenarios must have the same n_obs")

    n_obs = calm_spec.n_obs
    n_assets = calm_spec.n_assets
    states = _sample_state_path(state_spec, n_obs, rng)
    x = np.empty((n_obs, n_assets), dtype=float)

    for t in range(n_obs):
        scenario = calm_spec if states[t] == 0 else crisis_spec
        observation = generate_static_scenario_data(
            1,
            scenario.n_assets,
            scenario.structure,
            scenario.copula,
            scenario.marginal,
            rng,
        )
        x[t] = observation[0]

    return x.astype(float, copy=False), states.astype(int, copy=False)




def generate_two_state_scenario_data(
    calm_spec: ScenarioSpec,
    crisis_spec: ScenarioSpec,
    state_spec: StateProcessSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera datos dinamicos sinteticos de dos estados y devuelve tambien la trayectoria."""
    return _generate_two_state_observationwise_data(calm_spec, crisis_spec, state_spec, rng)


def resolve_structure_correlation(
    structure: StructureSpec,
    n_assets: int,
) -> np.ndarray:
    """Resuelve una especificacion estructural a una matriz base de correlacion."""
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be a StructureSpec")
    n_assets = validate_n_assets(n_assets)

    if structure.kind == "equicorrelation":
        if "rho" not in structure.params:
            raise ValueError("equicorrelation structure requires 'rho'")
        matrix = build_equicorrelation_matrix(n_assets, structure.params["rho"])
    elif structure.kind == "block":
        if "block_sizes" not in structure.params:
            raise ValueError("block structure requires 'block_sizes'")
        if "rho_within" not in structure.params:
            raise ValueError("block structure requires 'rho_within'")
        if "rho_between" not in structure.params:
            raise ValueError("block structure requires 'rho_between'")
        matrix = build_block_correlation_matrix(
            structure.params["block_sizes"],
            structure.params["rho_within"],
            structure.params["rho_between"],
        )
    elif structure.kind == "near_duplicates":
        if "group_sizes" not in structure.params:
            raise ValueError("near_duplicates structure requires 'group_sizes'")
        if "rho_duplicate" not in structure.params:
            raise ValueError("near_duplicates structure requires 'rho_duplicate'")
        matrix = build_near_duplicate_correlation_matrix(
            structure.params["group_sizes"],
            structure.params["rho_duplicate"],
            structure.params.get("rho_background", 0.0),
        )
    elif structure.kind == "factor":
        if "loadings" not in structure.params:
            raise ValueError("factor structure requires 'loadings'")
        matrix = build_factor_correlation_matrix(structure.params["loadings"])
    elif structure.kind == "nonlinear_redundancy":
        raise ValueError("nonlinear_redundancy does not define a linear correlation matrix")
    else:
        raise ValueError(f"unsupported structure kind: {structure.kind}")

    if matrix.shape != (n_assets, n_assets):
        raise ValueError("resolved correlation shape must match n_assets")
    return matrix



def sample_uniform_from_copula(
    copula: CopulaSpec,
    n_obs: int,
    n_assets: int,
    rng: np.random.Generator,
    correlation: np.ndarray | None = None,
) -> np.ndarray:
    """Resuelve una especificacion de copula a una muestra uniforme."""
    if not isinstance(copula, CopulaSpec):
        raise TypeError("copula must be a CopulaSpec")
    n_obs = validate_n_obs(n_obs)
    n_assets = validate_n_assets(n_assets)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    if copula.kind == "independence":
        u = sample_independence_copula(n_obs, n_assets, rng)
    elif copula.kind == "gaussian":
        if correlation is None:
            raise ValueError("gaussian copula requires 'correlation'")
        u = sample_gaussian_copula(n_obs, correlation, rng)
    elif copula.kind == "t":
        if correlation is None:
            raise ValueError("t copula requires 'correlation'")
        if "df" not in copula.params:
            raise ValueError("t copula requires 'df'")
        u = sample_t_copula(n_obs, correlation, copula.params["df"], rng)
    elif copula.kind == "grouped_t":
        if correlation is None:
            raise ValueError("grouped_t copula requires 'correlation'")
        if "group_assignments" not in copula.params:
            raise ValueError("grouped_t copula requires 'group_assignments'")
        if "group_dfs" not in copula.params:
            raise ValueError("grouped_t copula requires 'group_dfs'")
        u = sample_grouped_t_copula(
            n_obs,
            correlation,
            copula.params["group_assignments"],
            copula.params["group_dfs"],
            rng,
        )
    elif copula.kind == "clayton":
        if "theta" not in copula.params:
            raise ValueError("clayton copula requires 'theta'")
        u = sample_clayton_copula(n_obs, n_assets, copula.params["theta"], rng)
    else:
        raise ValueError(f"unsupported copula kind: {copula.kind}")

    if u.shape != (n_obs, n_assets):
        raise ValueError("resolved copula sample shape must match (n_obs, n_assets)")
    return u.astype(float, copy=False)



def apply_marginal_transform(
    marginal: MarginalSpec,
    u: np.ndarray,
) -> np.ndarray:
    """Resuelve una especificacion marginal sobre un array de uniforms."""
    if not isinstance(marginal, MarginalSpec):
        raise TypeError("marginal must be a MarginalSpec")
    if not isinstance(u, np.ndarray):
        raise TypeError("u must be a numpy.ndarray")
    if u.ndim not in (1, 2):
        raise ValueError("u must be a 1D or 2D array")

    if marginal.kind == "gaussian":
        x = apply_standard_normal_inverse_cdf(u)
    elif marginal.kind == "student_t":
        if "df" not in marginal.params:
            raise ValueError("student_t marginal requires 'df'")
        x = apply_standardized_student_t_inverse_cdf(u, marginal.params["df"])
    elif marginal.kind == "skew_normal":
        if "shape" not in marginal.params:
            raise ValueError("skew_normal marginal requires 'shape'")
        x = apply_standardized_skew_normal_inverse_cdf(u, marginal.params["shape"])
    elif marginal.kind == "heterogeneous":
        if u.ndim != 2:
            raise ValueError("heterogeneous marginal requires a 2D array")
        if "specs" not in marginal.params:
            raise ValueError("heterogeneous marginal requires 'specs'")

        specs = marginal.params["specs"]
        if not isinstance(specs, list):
            raise TypeError("heterogeneous marginal 'specs' must be a list")
        if u.shape[1] != len(specs):
            raise ValueError("heterogeneous marginal specs length must match the number of columns in u")

        columns: list[np.ndarray] = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, MarginalSpec):
                raise TypeError("heterogeneous marginal specs must contain MarginalSpec instances")
            columns.append(apply_marginal_transform(spec, u[:, index]))

        x = np.column_stack(columns).astype(float, copy=False)
    else:
        raise ValueError(f"unsupported marginal kind: {marginal.kind}")

    if x.shape != u.shape:
        raise ValueError("resolved marginal transform shape must match input shape")
    return x.astype(float, copy=False)


def generate_static_scenario_data(
    n_obs: int,
    n_assets: int,
    structure: StructureSpec,
    copula: CopulaSpec,
    marginal: MarginalSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera datos estaticos sinteticos a partir de estructura, copula y marginal."""
    n_obs = validate_n_obs(n_obs)
    n_assets = validate_n_assets(n_assets)
    if not isinstance(structure, StructureSpec):
        raise TypeError("structure must be a StructureSpec")
    if not isinstance(copula, CopulaSpec):
        raise TypeError("copula must be a CopulaSpec")
    if not isinstance(marginal, MarginalSpec):
        raise TypeError("marginal must be a MarginalSpec")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    if structure.kind == "nonlinear_redundancy":
        raise ValueError("static generator does not support nonlinear_redundancy structures")

    correlation = resolve_structure_correlation(structure, n_assets)
    if copula.kind in {"gaussian", "t", "grouped_t"}:
        u = sample_uniform_from_copula(
            copula,
            n_obs,
            n_assets,
            rng,
            correlation=correlation,
        )
    else:
        u = sample_uniform_from_copula(
            copula,
            n_obs,
            n_assets,
            rng,
            correlation=None,
        )

    x = apply_marginal_transform(marginal, u)
    if x.shape != (n_obs, n_assets):
        raise ValueError("generated data shape must match (n_obs, n_assets)")
    return x.astype(float, copy=False)







