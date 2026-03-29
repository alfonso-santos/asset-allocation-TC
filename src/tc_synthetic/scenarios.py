"""Modulo de escenarios del toolbox de datos sinteticos."""

from tc_synthetic.specs import ScenarioSpec, StateProcessSpec

__all__ = ["classify_two_state_scenario", "validate_two_state_scenario"]



def classify_two_state_scenario(calm_spec: ScenarioSpec, crisis_spec: ScenarioSpec) -> str:
    """Clasifica un par calm/crisis segun cambien marginales, dependencia o ambas."""
    if not isinstance(calm_spec, ScenarioSpec):
        raise TypeError("calm_spec must be a ScenarioSpec")
    if not isinstance(crisis_spec, ScenarioSpec):
        raise TypeError("crisis_spec must be a ScenarioSpec")
    if calm_spec.n_assets != crisis_spec.n_assets:
        raise ValueError("calm and crisis scenarios must have the same n_assets")

    marginals_differ = calm_spec.marginal != crisis_spec.marginal
    dependence_differ = (calm_spec.copula, calm_spec.structure) != (
        crisis_spec.copula,
        crisis_spec.structure,
    )

    if not marginals_differ and not dependence_differ:
        raise ValueError("two-state scenario must differ in marginals, dependence, or both")
    if dependence_differ and not marginals_differ:
        return "S1"
    if marginals_differ and not dependence_differ:
        return "S2"
    return "S3"


def validate_two_state_scenario(
    calm_spec: ScenarioSpec,
    crisis_spec: ScenarioSpec,
    state_spec: StateProcessSpec,
) -> str:
    """Valida un escenario declarativo de dos estados y devuelve su clasificacion."""
    if not isinstance(calm_spec, ScenarioSpec):
        raise TypeError("calm_spec must be a ScenarioSpec")
    if not isinstance(crisis_spec, ScenarioSpec):
        raise TypeError("crisis_spec must be a ScenarioSpec")
    if not isinstance(state_spec, StateProcessSpec):
        raise TypeError("state_spec must be a StateProcessSpec")
    if calm_spec.n_assets != crisis_spec.n_assets:
        raise ValueError("calm and crisis scenarios must have the same n_assets")
    if not state_spec.enabled:
        raise ValueError("state process must be enabled for two-state scenarios")
    if state_spec.kind != "markov":
        raise ValueError("only 'markov' state processes are supported")
    return classify_two_state_scenario(calm_spec, crisis_spec)

