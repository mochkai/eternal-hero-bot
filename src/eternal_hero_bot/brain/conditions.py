"""Condition predicates and TOML parsing for brain rules.

Conditions are parsed once at load time. Unknown keys are errors. A condition
that reads an unknown value (``value is None``) evaluates to ``False``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from eternal_hero_bot.brain.base import BrainContext
from eternal_hero_bot.core.state import GameState, ScreenKind

__all__ = ["Condition", "parse_condition"]


class Condition(Protocol):
    """A predicate evaluated against the current game state and brain context."""

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        """Evaluate the condition.

        Args:
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            True when the condition holds.
        """
        ...


_KNOWN_KEYS = frozenset(
    {
        "screen",
        "resource_at_max",
        "resource_gte",
        "resource_lt",
        "resource_frac_gte",
        "detected",
        "objective",
        "since_last_run",
        "all",
        "any",
        "not",
    }
)


def parse_condition(raw: object, *, path: str = "when") -> Condition:
    """Parse a condition from a TOML-derived value.

    Args:
        raw: Raw condition value, typically a mapping with one predicate key.
        path: Dotted path for error messages.

    Returns:
        The parsed condition.

    Raises:
        ValueError: If the condition is malformed or uses unknown keys.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: condition must be a table, got {type(raw).__name__}")
    keys = set(raw.keys())
    unknown = keys - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown condition key(s): {sorted(unknown)}")
    if len(keys) != 1:
        raise ValueError(f"{path}: condition must have exactly one key, got {sorted(keys)}")
    key = next(iter(keys))
    value = raw[key]
    if key == "screen":
        return _Screen(value=value, path=path)
    if key == "resource_at_max":
        return _ResourceAtMax(name=_expect_str(value, f"{path}.resource_at_max"))
    if key == "resource_gte":
        name, num = _expect_name_value(value, f"{path}.resource_gte")
        return _ResourceCmp(name=name, op="gte", value=num)
    if key == "resource_lt":
        name, num = _expect_name_value(value, f"{path}.resource_lt")
        return _ResourceCmp(name=name, op="lt", value=num)
    if key == "resource_frac_gte":
        name, frac = _expect_name_frac(value, f"{path}.resource_frac_gte")
        return _ResourceFracGte(name=name, frac=frac)
    if key == "detected":
        return _Detected(name=_expect_str(value, f"{path}.detected"))
    if key == "objective":
        return _ObjectiveKind(kind=_expect_str(value, f"{path}.objective"))
    if key == "since_last_run":
        skill, seconds = _expect_since_last_run(value, f"{path}.since_last_run")
        return _SinceLastRun(skill=skill, seconds=seconds)
    if key == "all":
        return _All(children=_expect_condition_list(value, f"{path}.all"))
    if key == "any":
        return _Any(children=_expect_condition_list(value, f"{path}.any"))
    if key == "not":
        return _Not(child=parse_condition(value, path=f"{path}.not"))
    raise ValueError(f"{path}: unhandled condition key {key!r}")  # pragma: no cover


def _expect_str(value: object, path: str) -> str:
    """Expect a string value.

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The string.

    Raises:
        ValueError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected string, got {type(value).__name__}")
    return value


def _expect_number(value: object, path: str) -> float:
    """Expect a numeric value (int or float, not bool).

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The numeric value as float.

    Raises:
        ValueError: If the value is not numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected number, got {type(value).__name__}")
    return float(value)


def _expect_mapping(value: object, path: str) -> Mapping[str, object]:
    """Expect a mapping with string keys.

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The mapping.

    Raises:
        ValueError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected table, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{path}: expected string keys")
    return value


def _expect_name_value(value: object, path: str) -> tuple[str, float]:
    """Parse ``{name, value}`` for numeric resource comparisons.

    Args:
        value: Raw table.
        path: Dotted path for error messages.

    Returns:
        The resource name and numeric value.

    Raises:
        ValueError: If the table is malformed.
    """
    table = _expect_mapping(value, path)
    unknown = set(table.keys()) - {"name", "value"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {sorted(unknown)}")
    if "name" not in table or "value" not in table:
        raise ValueError(f"{path}: expected keys 'name' and 'value'")
    return _expect_str(table["name"], f"{path}.name"), _expect_number(
        table["value"], f"{path}.value"
    )


def _expect_name_frac(value: object, path: str) -> tuple[str, float]:
    """Parse ``{name, frac}`` for fractional resource comparisons.

    Args:
        value: Raw table.
        path: Dotted path for error messages.

    Returns:
        The resource name and fraction.

    Raises:
        ValueError: If the table is malformed.
    """
    table = _expect_mapping(value, path)
    unknown = set(table.keys()) - {"name", "frac"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {sorted(unknown)}")
    if "name" not in table or "frac" not in table:
        raise ValueError(f"{path}: expected keys 'name' and 'frac'")
    return _expect_str(table["name"], f"{path}.name"), _expect_number(table["frac"], f"{path}.frac")


def _expect_since_last_run(value: object, path: str) -> tuple[str, float]:
    """Parse ``{skill, s}`` for since-last-run conditions.

    Args:
        value: Raw table.
        path: Dotted path for error messages.

    Returns:
        The skill name and seconds threshold.

    Raises:
        ValueError: If the table is malformed.
    """
    table = _expect_mapping(value, path)
    unknown = set(table.keys()) - {"skill", "s"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {sorted(unknown)}")
    if "skill" not in table or "s" not in table:
        raise ValueError(f"{path}: expected keys 'skill' and 's'")
    seconds = _expect_number(table["s"], f"{path}.s")
    if seconds < 0:
        raise ValueError(f"{path}.s: must be >= 0")
    return _expect_str(table["skill"], f"{path}.skill"), seconds


def _expect_condition_list(value: object, path: str) -> tuple[Condition, ...]:
    """Parse a list of conditions.

    Args:
        value: Raw list.
        path: Dotted path for error messages.

    Returns:
        Parsed conditions.

    Raises:
        ValueError: If the value is not a list of conditions.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path}: expected list, got {type(value).__name__}")
    return tuple(parse_condition(item, path=f"{path}[{i}]") for i, item in enumerate(value))


@dataclass(frozen=True)
class _Screen:
    """Screen-kind predicate."""

    kind: ScreenKind

    def __init__(self, value: object, path: str) -> None:
        kind_str = _expect_str(value, path)
        try:
            kind = ScreenKind(kind_str)
        except ValueError as exc:
            valid = ", ".join(sorted(k.value for k in ScreenKind))
            raise ValueError(
                f"{path}: unknown screen {kind_str!r}; expected one of {valid}"
            ) from exc
        object.__setattr__(self, "kind", kind)

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del ctx
        return state.screen == self.kind


@dataclass(frozen=True)
class _ResourceAtMax:
    """True when a named resource is at or above its cap."""

    name: str

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del ctx
        reading = state.resources.get(self.name)
        return reading is not None and reading.at_max


@dataclass(frozen=True)
class _ResourceCmp:
    """Numeric comparison against a resource's current value."""

    name: str
    op: str  # "gte" | "lt"
    value: float

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del ctx
        reading = state.resources.get(self.name)
        if reading is None or reading.value is None:
            return False
        current = float(reading.value)
        if self.op == "gte":
            return current >= self.value
        return current < self.value


@dataclass(frozen=True)
class _ResourceFracGte:
    """True when value/max >= frac for a named resource."""

    name: str
    frac: float

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del ctx
        reading = state.resources.get(self.name)
        if reading is None or reading.value is None or reading.max is None:
            return False
        if reading.max == 0:
            return False
        return (reading.value / reading.max) >= self.frac


@dataclass(frozen=True)
class _Detected:
    """True when at least one detection of the named template exists."""

    name: str

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del ctx
        return len(state.detections.get(self.name, ())) > 0


@dataclass(frozen=True)
class _ObjectiveKind:
    """True when at least one objective of the given kind exists."""

    kind: str

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del ctx
        return any(obj.kind == self.kind for obj in state.objectives)


@dataclass(frozen=True)
class _SinceLastRun:
    """True when a skill has not finished in `s` seconds, or never finished."""

    skill: str
    seconds: float

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        del state
        last = ctx.last_run.get(self.skill)
        if last is None:
            return True
        return (ctx.now - last) >= self.seconds


@dataclass(frozen=True)
class _All:
    """All children must hold."""

    children: tuple[Condition, ...]

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        return all(child.evaluate(state, ctx) for child in self.children)


@dataclass(frozen=True)
class _Any:
    """Any child must hold."""

    children: tuple[Condition, ...]

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        return any(child.evaluate(state, ctx) for child in self.children)


@dataclass(frozen=True)
class _Not:
    """Negation of a child condition."""

    child: Condition

    def evaluate(self, state: GameState, ctx: BrainContext) -> bool:
        return not self.child.evaluate(state, ctx)
