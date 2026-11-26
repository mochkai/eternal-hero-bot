"""The skill framework: protocol, context and the shared step vocabulary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol

from eternal_hero_bot.action.navigation import Navigator
from eternal_hero_bot.core.outcomes import StepResult
from eternal_hero_bot.core.state import GameState


@dataclass
class SkillContext:
    """Services the runtime hands to a skill on every tick.

    Attributes:
        navigator: Movement policy used to turn objectives into actions.
        clock: Returns the current time in seconds (injectable for tests).
        options: Options table of the running skill (the ``[skills.<name>]`` config).
    """

    navigator: Navigator
    clock: Callable[[], float]
    options: Mapping[str, object]


class Skill(Protocol):
    """Anything that can run inside the runtime's decision loop.

    Attributes:
        name: Registered name of the skill.
        timeout_s: Wall-clock budget per activation, or ``None`` for unlimited.
        max_steps: Step budget per activation, or ``None`` for unlimited.
    """

    name: ClassVar[str]
    timeout_s: float | None
    max_steps: int | None

    @property
    def interruptible(self) -> bool:
        """Whether a higher-priority rule may preempt this skill mid-run."""
        ...

    def can_run(self, state: GameState) -> bool:
        """Whether the skill may run on this game state.

        Args:
            state: Current immutable game state.

        Returns:
            True when the skill applies to this frame.
        """
        ...

    def start(self, state: GameState, ctx: SkillContext, args: Mapping[str, object]) -> None:
        """Start (or restart) the skill with a fresh activation budget.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.
            args: Arguments for the skill, as produced by the deciding rule.
        """
        ...

    def is_done(self, state: GameState) -> bool:
        """Whether the skill considers itself finished on this game state.

        Args:
            state: Current immutable game state.

        Returns:
            True when the current activation should end.
        """
        ...

    def step(self, state: GameState, ctx: SkillContext) -> StepResult:
        """Advance the skill by one step and return the resulting actions.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.

        Returns:
            The step outcome, carrying the actions to execute (if any).
        """
        ...

    def abort(self, reason: str) -> None:
        """Abort the activation, releasing state without emitting actions.

        Args:
            reason: Machine-readable abort reason (for example ``"preempted"`` or ``"error"``).
        """
        ...
