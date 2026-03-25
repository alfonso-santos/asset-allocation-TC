"""Modulo de especificaciones base del toolbox de datos sinteticos."""

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ALLOWED_MARGINAL_KINDS",
    "ALLOWED_COPULA_KINDS",
    "ALLOWED_STRUCTURE_KINDS",
    "ALLOWED_STATE_PROCESS_KINDS",
    "MarginalSpec",
    "CopulaSpec",
    "StructureSpec",
    "StateProcessSpec",
    "ScenarioSpec",
]

ALLOWED_MARGINAL_KINDS: frozenset[str] = frozenset(
    {"gaussian", "student_t", "skew_normal", "heterogeneous"}
)
ALLOWED_COPULA_KINDS: frozenset[str] = frozenset(
    {"independence", "gaussian", "t", "grouped_t", "clayton"}
)
ALLOWED_STRUCTURE_KINDS: frozenset[str] = frozenset(
    {"equicorrelation", "block", "factor", "near_duplicates", "nonlinear_redundancy"}
)
ALLOWED_STATE_PROCESS_KINDS: frozenset[str] = frozenset({"markov"})


def _validate_kind(value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("kind must be a string")
    if not value.strip():
        raise ValueError("kind must not be empty")


def _validate_name(value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError("name must be a string")
    if not value.strip():
        raise ValueError("name must not be empty")


def _validate_params(value: Any) -> None:
    if not isinstance(value, dict):
        raise TypeError("params must be a dict")


def _validate_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0")


def _validate_seed(value: Any) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError("seed must be an integer or None")


def _validate_instance(value: Any, expected_type: type[Any], field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be a {expected_type.__name__}")


def _validate_allowed_kind(value: str, allowed_kinds: frozenset[str], label: str) -> None:
    if value not in allowed_kinds:
        raise ValueError(f"unsupported {label} kind: {value}")


def _validate_enabled(value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError("enabled must be a bool")


@dataclass(slots=True)
class MarginalSpec:
    """Especificacion de una marginal sintetica."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Aplica validaciones estructurales."""
        _validate_kind(self.kind)
        _validate_params(self.params)
        _validate_allowed_kind(self.kind, ALLOWED_MARGINAL_KINDS, "marginal")


@dataclass(slots=True)
class CopulaSpec:
    """Especificacion de una copula sintetica."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Aplica validaciones estructurales."""
        _validate_kind(self.kind)
        _validate_params(self.params)
        _validate_allowed_kind(self.kind, ALLOWED_COPULA_KINDS, "copula")


@dataclass(slots=True)
class StructureSpec:
    """Especificacion de una estructura sintetica."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Aplica validaciones estructurales."""
        _validate_kind(self.kind)
        _validate_params(self.params)
        _validate_allowed_kind(self.kind, ALLOWED_STRUCTURE_KINDS, "structure")


@dataclass(slots=True)
class StateProcessSpec:
    """Especificacion de un proceso de estados."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False

    def __post_init__(self) -> None:
        """Aplica validaciones estructurales."""
        _validate_kind(self.kind)
        _validate_params(self.params)
        _validate_allowed_kind(self.kind, ALLOWED_STATE_PROCESS_KINDS, "state_process")
        _validate_enabled(self.enabled)


@dataclass(slots=True)
class ScenarioSpec:
    """Especificacion de un escenario sintetico completo."""

    name: str
    n_assets: int
    n_obs: int
    seed: int | None = field(default=None, kw_only=True)
    marginal: MarginalSpec
    copula: CopulaSpec
    structure: StructureSpec
    state_process: StateProcessSpec | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Aplica validaciones estructurales."""
        _validate_name(self.name)
        _validate_positive_int(self.n_assets, "n_assets")
        _validate_positive_int(self.n_obs, "n_obs")
        _validate_seed(self.seed)
        _validate_instance(self.marginal, MarginalSpec, "marginal")
        _validate_instance(self.copula, CopulaSpec, "copula")
        _validate_instance(self.structure, StructureSpec, "structure")
        if self.state_process is not None:
            _validate_instance(self.state_process, StateProcessSpec, "state_process")
