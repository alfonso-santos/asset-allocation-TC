"""Pruebas unitarias para ``tc_synthetic.scenarios``."""

import pytest

from tc_synthetic.scenarios import classify_two_state_scenario, validate_two_state_scenario
from tc_synthetic.specs import CopulaSpec, MarginalSpec, ScenarioSpec, StateProcessSpec, StructureSpec



def _make_scenario(
    *,
    name: str,
    n_assets: int = 3,
    n_obs: int = 100,
    marginal: MarginalSpec | None = None,
    copula: CopulaSpec | None = None,
    structure: StructureSpec | None = None,
) -> ScenarioSpec:
    """Construye un ``ScenarioSpec`` pequeno para los tests."""
    return ScenarioSpec(
        name=name,
        n_assets=n_assets,
        n_obs=n_obs,
        marginal=marginal or MarginalSpec(kind="gaussian"),
        copula=copula or CopulaSpec(kind="independence"),
        structure=structure or StructureSpec(kind="equicorrelation", params={"rho": 0.3}),
    )



def test_classify_two_state_scenario_returns_s1_when_only_copula_changes() -> None:
    """Verifica ``S1`` cuando solo cambia la dependencia por la copula."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        copula=CopulaSpec(kind="gaussian"),
    )

    result = classify_two_state_scenario(calm_spec, crisis_spec)

    assert result == "S1"



def test_classify_two_state_scenario_returns_s1_when_only_structure_changes() -> None:
    """Verifica ``S1`` cuando solo cambia la dependencia por la estructura."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        structure=StructureSpec(kind="block", params={"block_sizes": [1, 2], "rho_within": 0.6, "rho_between": 0.1}),
    )

    result = classify_two_state_scenario(calm_spec, crisis_spec)

    assert result == "S1"



def test_classify_two_state_scenario_returns_s2_when_only_marginals_change() -> None:
    """Verifica ``S2`` cuando solo cambian las marginales."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
    )

    result = classify_two_state_scenario(calm_spec, crisis_spec)

    assert result == "S2"



def test_classify_two_state_scenario_returns_s3_when_marginals_and_dependence_change() -> None:
    """Verifica ``S3`` cuando cambian marginales y dependencia."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        marginal=MarginalSpec(kind="student_t", params={"df": 5.0}),
        copula=CopulaSpec(kind="gaussian"),
    )

    result = classify_two_state_scenario(calm_spec, crisis_spec)

    assert result == "S3"



def test_classify_two_state_scenario_raises_when_n_assets_differs() -> None:
    """Verifica el error exacto si ``n_assets`` difiere."""
    calm_spec = _make_scenario(name="calm", n_assets=3)
    crisis_spec = _make_scenario(name="crisis", n_assets=4)

    try:
        classify_two_state_scenario(calm_spec, crisis_spec)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "calm and crisis scenarios must have the same n_assets"



def test_classify_two_state_scenario_raises_when_scenarios_are_identical() -> None:
    """Verifica el error exacto si los escenarios son identicos."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(name="calm")

    try:
        classify_two_state_scenario(calm_spec, crisis_spec)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "two-state scenario must differ in marginals, dependence, or both"


# === Added in Step 10.2 ===

@pytest.mark.parametrize(
    ("crisis_kwargs", "expected"),
    [
        ({"copula": CopulaSpec(kind="gaussian")}, "S1"),
        ({"marginal": MarginalSpec(kind="student_t", params={"df": 5.0})}, "S2"),
        (
            {
                "marginal": MarginalSpec(kind="student_t", params={"df": 5.0}),
                "copula": CopulaSpec(kind="gaussian"),
            },
            "S3",
        ),
    ],
)
def test_validate_two_state_scenario_returns_expected_classification(
    crisis_kwargs: dict[str, object],
    expected: str,
) -> None:
    """Verifica que la validacion devuelve la clasificacion correcta."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(name="crisis", **crisis_kwargs)
    state_spec = StateProcessSpec(kind="markov", enabled=True)

    result = validate_two_state_scenario(calm_spec, crisis_spec, state_spec)

    assert result == expected



def test_validate_two_state_scenario_raises_when_state_process_is_disabled() -> None:
    """Verifica el error si el proceso de estados no esta habilitado."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        copula=CopulaSpec(kind="gaussian"),
    )
    state_spec = StateProcessSpec(kind="markov", enabled=False)

    with pytest.raises(
        ValueError,
        match="state process must be enabled for two-state scenarios",
    ):
        validate_two_state_scenario(calm_spec, crisis_spec, state_spec)



def test_validate_two_state_scenario_raises_when_state_process_kind_is_not_markov() -> None:
    """Verifica el error si el kind del proceso de estados no es ``markov``."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        copula=CopulaSpec(kind="gaussian"),
    )
    state_spec = StateProcessSpec(kind="markov", enabled=True)
    state_spec.kind = "unknown"

    with pytest.raises(
        ValueError,
        match="only 'markov' state processes are supported",
    ):
        validate_two_state_scenario(calm_spec, crisis_spec, state_spec)



def test_validate_two_state_scenario_raises_for_invalid_state_spec() -> None:
    """Verifica que ``state_spec`` debe ser un ``StateProcessSpec``."""
    calm_spec = _make_scenario(name="calm")
    crisis_spec = _make_scenario(
        name="crisis",
        copula=CopulaSpec(kind="gaussian"),
    )

    with pytest.raises(TypeError, match="state_spec must be a StateProcessSpec"):
        validate_two_state_scenario(calm_spec, crisis_spec, "x")

