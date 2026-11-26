"""The revive skill: death handling through a forced brain rule."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from eternal_hero_bot.core.actions import TapNamed, Wait
from eternal_hero_bot.core.outcomes import SkillStatus, StepResult
from eternal_hero_bot.core.state import GameState, ScreenKind
from eternal_hero_bot.skills.base import SkillContext
from eternal_hero_bot.skills.registry import register

REVIVE_WAIT_S = 3.0
"""Seconds to wait after tapping Revive."""


@register("revive")
class ReviveSkill:
    """Revive: tap the Revive button after a death and wait for the hero to come back.

    The skill is pure: it reads the game state and returns the tap/wait actions, which the
    controller turns into device calls. Death memory cleanup happens on start because the
    world position is meaningless right after a respawn.
    """

    name: ClassVar[str] = "revive"
    timeout_s: float | None = None
    max_steps: int | None = None

    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        """Create the revive skill, rejecting unrecognised options.

        Args:
            options: Options table for the skill; none are recognised yet.

        Raises:
            ValueError: If the options table is not empty.
        """
        table = dict(options) if options else {}
        if table:
            raise ValueError(f"revive: unknown options {sorted(table)}")

    @property
    def interruptible(self) -> bool:
        """Revive never yields: death handling is forced by the brain."""
        return False

    def can_run(self, state: GameState) -> bool:
        """Revive runs only on dead screens.

        Args:
            state: Current immutable game state.

        Returns:
            True when the death dialog is showing.
        """
        return state.screen == ScreenKind.DEAD

    def start(self, state: GameState, ctx: SkillContext, args: Mapping[str, object]) -> None:
        """Release remembered world state before the revive flow starts.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.
            args: Arguments for the skill, as produced by the deciding rule.
        """
        ctx.navigator.reset()

    def is_done(self, state: GameState) -> bool:
        """Revive is done once the hero is back (the screen leaves the dead state).

        Args:
            state: Current immutable game state.

        Returns:
            True when the hero is alive again.
        """
        return state.screen != ScreenKind.DEAD

    def step(self, state: GameState, ctx: SkillContext) -> StepResult:
        """Tap Revive and wait for the respawn; the whole flow ends as DONE.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.

        Returns:
            A ``DONE`` result carrying the tap and wait actions.
        """
        if state.screen == ScreenKind.DEAD:
            return StepResult(
                status=SkillStatus.DONE,
                actions=(TapNamed(name="revive_button"), Wait(seconds=REVIVE_WAIT_S)),
            )
        return StepResult(status=SkillStatus.DONE, actions=())

    def abort(self, reason: str) -> None:
        """Release revive state without emitting actions.

        Args:
            reason: Machine-readable abort reason (for example ``"preempted"``).
        """
        del reason
