"""Pruebas minimas para las dataclasses de ``tc_synthetic.specs``."""

from dataclasses import is_dataclass

import pytest

from tc_synthetic.specs import (
    CopulaSpec,
    MarginalSpec,
    ScenarioSpec,
    StateProcessSpec,
    StructureSpec,
)


def test_marginal_spec_stores_attributes() -> None:
    """Verifica que ``MarginalSpec`` almacena sus atributos."""
    spec = MarginalSpec(kind="gaussian", params={"mu": 0.0})

    assert spec.kind == "gaussian"
    assert spec.params == {"mu": 0.0}


def test_copula_spec_stores_attributes() -> None:
    """Verifica que ``CopulaSpec`` almacena sus atributos."""
    spec = CopulaSpec(kind="gaussian", params={"rho": 0.5})

    assert spec.kind == "gaussian"
    assert spec.params == {"rho": 0.5}


def test_structure_spec_stores_attributes() -> None:
    """Verifica que ``StructureSpec`` almacena sus atributos."""
    spec = StructureSpec(kind="equicorrelation", params={"rho": 0.3})

    assert spec.kind == "equicorrelation"
    assert spec.params == {"rho": 0.3}


def test_state_process_spec_stores_attributes() -> None:
    """Verifica que ``StateProcessSpec`` almacena sus atributos."""
    spec = StateProcessSpec(kind="markov", params={"states": 3}, enabled=True)

    assert spec.kind == "markov"
    assert spec.params == {"states": 3}
    assert spec.enabled is True


def test_valid_marginal_kind_is_accepted() -> None:
    """Verifica que una marginal valida se acepta."""
    spec = MarginalSpec(kind="skew_normal")

    assert spec.kind == "skew_normal"



def test_valid_copula_kind_is_accepted() -> None:
    """Verifica que una copula valida se acepta."""
    spec = CopulaSpec(kind="clayton")

    assert spec.kind == "clayton"



def test_valid_structure_kind_is_accepted() -> None:
    """Verifica que una estructura valida se acepta."""
    spec = StructureSpec(kind="factor")

    assert spec.kind == "factor"



def test_valid_state_process_kind_is_accepted() -> None:
    """Verifica que un proceso de estados valido se acepta."""
    spec = StateProcessSpec(kind="markov", enabled=False)

    assert spec.kind == "markov"
    assert spec.enabled is False


def test_params_are_not_shared_between_instances() -> None:
    """Verifica que ``params`` no comparte estado entre instancias."""
    left = MarginalSpec(kind="student_t")
    right = MarginalSpec(kind="student_t")

    left.params["df"] = 7

    assert left.params == {"df": 7}
    assert right.params == {}


def test_scenario_spec_composes_nested_specs() -> None:
    """Verifica que ``ScenarioSpec`` compone correctamente las demas specs."""
    marginal = MarginalSpec(kind="gaussian", params={"mu": 0.0})
    copula = CopulaSpec(kind="gaussian", params={"rho": 0.2})
    structure = StructureSpec(kind="block", params={"blocks": 2})
    state_process = StateProcessSpec(kind="markov", params={"states": 2}, enabled=True)

    scenario = ScenarioSpec(
        name="baseline",
        n_assets=4,
        n_obs=120,
        seed=123,
        marginal=marginal,
        copula=copula,
        structure=structure,
        state_process=state_process,
    )

    assert scenario.name == "baseline"
    assert scenario.n_assets == 4
    assert scenario.n_obs == 120
    assert scenario.seed == 123
    assert scenario.marginal is marginal
    assert scenario.copula is copula
    assert scenario.structure is structure
    assert scenario.state_process is state_process


def test_specs_are_dataclasses() -> None:
    """Verifica que todas las specs son dataclasses."""
    assert is_dataclass(MarginalSpec)
    assert is_dataclass(CopulaSpec)
    assert is_dataclass(StructureSpec)
    assert is_dataclass(StateProcessSpec)
    assert is_dataclass(ScenarioSpec)


def test_scenario_spec_accepts_seed_none() -> None:
    """Verifica que ``seed=None`` funciona."""
    scenario = ScenarioSpec(
        name="no-seed",
        n_assets=2,
        n_obs=50,
        seed=None,
        marginal=MarginalSpec(kind="gaussian"),
        copula=CopulaSpec(kind="independence"),
        structure=StructureSpec(kind="equicorrelation"),
    )

    assert scenario.seed is None


def test_scenario_spec_accepts_state_process_none() -> None:
    """Verifica que ``state_process=None`` funciona."""
    scenario = ScenarioSpec(
        name="no-state-process",
        n_assets=3,
        n_obs=80,
        marginal=MarginalSpec(kind="gaussian"),
        copula=CopulaSpec(kind="independence"),
        structure=StructureSpec(kind="equicorrelation"),
        state_process=None,
    )

    assert scenario.state_process is None


@pytest.mark.parametrize(
    ("spec_cls", "extra_kwargs"),
    [
        (MarginalSpec, {}),
        (CopulaSpec, {}),
        (StructureSpec, {}),
        (StateProcessSpec, {"enabled": False}),
    ],
)
def test_spec_raises_for_empty_kind(spec_cls, extra_kwargs: dict[str, object]) -> None:
    """Verifica que ``kind`` no puede ser vacio."""
    with pytest.raises(ValueError, match="kind must not be empty"):
        spec_cls(kind="", **extra_kwargs)


@pytest.mark.parametrize(
    ("spec_cls", "extra_kwargs"),
    [
        (MarginalSpec, {}),
        (CopulaSpec, {}),
        (StructureSpec, {}),
        (StateProcessSpec, {"enabled": True}),
    ],
)
def test_spec_raises_for_blank_kind(spec_cls, extra_kwargs: dict[str, object]) -> None:
    """Verifica que ``kind`` no puede tener solo espacios."""
    with pytest.raises(ValueError, match="kind must not be empty"):
        spec_cls(kind="   ", **extra_kwargs)


@pytest.mark.parametrize(
    ("spec_cls", "extra_kwargs"),
    [
        (MarginalSpec, {}),
        (CopulaSpec, {}),
        (StructureSpec, {}),
        (StateProcessSpec, {"enabled": False}),
    ],
)
def test_spec_raises_for_non_dict_params(spec_cls, extra_kwargs: dict[str, object]) -> None:
    """Verifica que ``params`` debe ser un diccionario."""
    with pytest.raises(TypeError, match="params must be a dict"):
        spec_cls(kind="gaussian", params=["invalid"], **extra_kwargs)


def test_scenario_spec_raises_for_empty_name() -> None:
    """Verifica que ``name`` no puede ser vacio."""
    with pytest.raises(ValueError, match="name must not be empty"):
        ScenarioSpec(
            name="",
            n_assets=2,
            n_obs=50,
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_zero_n_assets() -> None:
    """Verifica que ``n_assets`` debe ser positivo."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        ScenarioSpec(
            name="invalid-assets",
            n_assets=0,
            n_obs=50,
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_negative_n_assets() -> None:
    """Verifica que ``n_assets`` no puede ser negativo."""
    with pytest.raises(ValueError, match="n_assets must be greater than 0"):
        ScenarioSpec(
            name="invalid-assets",
            n_assets=-1,
            n_obs=50,
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_zero_n_obs() -> None:
    """Verifica que ``n_obs`` debe ser positivo."""
    with pytest.raises(ValueError, match="n_obs must be greater than 0"):
        ScenarioSpec(
            name="invalid-obs",
            n_assets=2,
            n_obs=0,
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_invalid_seed() -> None:
    """Verifica que ``seed`` debe ser entero o ``None``."""
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        ScenarioSpec(
            name="invalid-seed",
            n_assets=2,
            n_obs=50,
            seed="abc",
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_invalid_marginal() -> None:
    """Verifica que ``marginal`` debe tener el tipo correcto."""
    with pytest.raises(TypeError, match="marginal must be a MarginalSpec"):
        ScenarioSpec(
            name="invalid-marginal",
            n_assets=2,
            n_obs=50,
            marginal="gaussian",
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_invalid_copula() -> None:
    """Verifica que ``copula`` debe tener el tipo correcto."""
    with pytest.raises(TypeError, match="copula must be a CopulaSpec"):
        ScenarioSpec(
            name="invalid-copula",
            n_assets=2,
            n_obs=50,
            marginal=MarginalSpec(kind="gaussian"),
            copula="independence",
            structure=StructureSpec(kind="equicorrelation"),
        )


def test_scenario_spec_raises_for_invalid_structure() -> None:
    """Verifica que ``structure`` debe tener el tipo correcto."""
    with pytest.raises(TypeError, match="structure must be a StructureSpec"):
        ScenarioSpec(
            name="invalid-structure",
            n_assets=2,
            n_obs=50,
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure="equicorrelation",
        )


def test_scenario_spec_raises_for_invalid_state_process() -> None:
    """Verifica que ``state_process`` debe tener el tipo correcto."""
    with pytest.raises(TypeError, match="state_process must be a StateProcessSpec"):
        ScenarioSpec(
            name="invalid-state-process",
            n_assets=2,
            n_obs=50,
            marginal=MarginalSpec(kind="gaussian"),
            copula=CopulaSpec(kind="independence"),
            structure=StructureSpec(kind="equicorrelation"),
            state_process="markov",
        )


@pytest.mark.parametrize("kind", ["lognormal", "normal"])
def test_marginal_spec_raises_for_unsupported_kind(kind: str) -> None:
    """Verifica que una marginal no soportada falla."""
    with pytest.raises(ValueError, match="unsupported marginal kind"):
        MarginalSpec(kind=kind)


@pytest.mark.parametrize("kind", ["gumbel", "frank"])
def test_copula_spec_raises_for_unsupported_kind(kind: str) -> None:
    """Verifica que una copula no soportada falla."""
    with pytest.raises(ValueError, match="unsupported copula kind"):
        CopulaSpec(kind=kind)


@pytest.mark.parametrize("kind", ["static", "clustered"])
def test_structure_spec_raises_for_unsupported_kind(kind: str) -> None:
    """Verifica que una estructura no soportada falla."""
    with pytest.raises(ValueError, match="unsupported structure kind"):
        StructureSpec(kind=kind)


@pytest.mark.parametrize("kind", ["hmm", "regime_switching"])
def test_state_process_spec_raises_for_unsupported_kind(kind: str) -> None:
    """Verifica que un proceso de estados no soportado falla."""
    with pytest.raises(ValueError, match="unsupported state_process kind"):
        StateProcessSpec(kind=kind)


@pytest.mark.parametrize("enabled", [1, "yes"])
def test_state_process_spec_raises_for_invalid_enabled(enabled: object) -> None:
    """Verifica que ``enabled`` debe ser booleano."""
    with pytest.raises(TypeError, match="enabled must be a bool"):
        StateProcessSpec(kind="markov", enabled=enabled)
