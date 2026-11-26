"""Brain protocol and context: the decision layer's public interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from eternal_hero_bot.core.outcomes import Decision, SkillStatus
from eternal_hero_bot.core.state import GameState

__all__ = ["Brain", "BrainContext", "Decision"]


@dataclass(frozen=True)
class BrainContext:
    """Context the runtime hands to the brain on every tick.

    Attributes:
        now: Current time in seconds.
        current: Running skill name, if any.
        current_rule: Rule that started the running skill.
        current_interruptible: Whether the running skill may be preempted.
        can_run: Whether a skill may run on a given game state.
        last_run: Skill name to last finish time (for cooldown conditions).
    """

    now: float
    current: str | None
    current_rule: str | None
    current_interruptible: bool
    can_run: Callable[[str, GameState], bool]
    last_run: Mapping[str, float]


class Brain(Protocol):
    """Decides which skill runs on a game state."""

    def decide(self, state: GameState, ctx: BrainContext) -> Decision:
        """Decide the next skill for the current game state.

        Args:
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            The decision to run (possibly the running skill again).
        """
        ...

    def notify(self, decision: Decision, outcome: SkillStatus, now: float) -> None:
        """Record a finished or failed activation, e.g. for cooldown bookkeeping.

        Args:
            decision: The decision that produced the outcome.
            outcome: Skill status the activation ended with.
            now: Current time in seconds.
        """
        ...
