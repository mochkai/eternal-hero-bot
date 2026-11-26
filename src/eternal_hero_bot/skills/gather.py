"""The gather skill: steer toward the nearest visible node and push the joystick."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

from eternal_hero_bot.core.actions import Move
from eternal_hero_bot.core.outcomes import SkillStatus, StepResult
from eternal_hero_bot.core.state import GameState, ScreenKind
from eternal_hero_bot.skills.base import SkillContext
from eternal_hero_bot.skills.registry import register


@register("gather")
class GatherSkill:
    """Gather: steer toward the nearest visible node and push the joystick.

    The skill is pure: it reads a :class:`~eternal_hero_bot.core.state.GameState` and
    returns the resulting :class:`~eternal_hero_bot.core.actions.Move` through the
    :class:`~eternal_hero_bot.action.navigation.Navigator`. It never touches the device,
    the perception layer or the controller.
    """

    name: ClassVar[str] = "gather"
    timeout_s: float | None = None
    max_steps: int | None = None

    _ALLOWED_OPTIONS: ClassVar[frozenset[str]] = frozenset({"timeout_s", "max_steps"})

    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        """Create the gather skill, applying the ``[skills.gather]`` option overrides.

        Args:
            options: Options table for the skill; recognised keys are ``timeout_s`` and
                ``max_steps``. ``timeout_s = 0`` means unlimited.

        Raises:
            ValueError: If the options table holds unrecognised keys.
        """
        table = dict(options) if options else {}
        unknown = sorted(set(table) - self._ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(f"gather: unknown options {unknown}")
        raw_timeout = table.get("timeout_s")
        seconds = float(cast(float, raw_timeout)) if raw_timeout is not None else 0.0
        self.timeout_s = seconds if seconds > 0 else None
        raw_max = table.get("max_steps")
        self.max_steps = int(cast(int, raw_max)) if raw_max is not None else None

    @property
    def interruptible(self) -> bool:
        """Gather stays interruptible; higher-priority rules may preempt it."""
        return True

    def can_run(self, state: GameState) -> bool:
        """Gather runs only when the minimap is visible and the hero controllable.

        Args:
            state: Current immutable game state.

        Returns:
            True on world screens, False on dead or unknown screens.
        """
        return state.screen == ScreenKind.WORLD

    def start(self, state: GameState, ctx: SkillContext, args: Mapping[str, object]) -> None:
        """Start a gather activation.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.
            args: Arguments for the skill, as produced by the deciding rule.
        """
        del state, ctx, args  # gather keeps no per-activation state

    def is_done(self, state: GameState) -> bool:
        """Gather never completes on its own; it runs until preempted, failed or timed out.

        Args:
            state: Current immutable game state.

        Returns:
            False while gathering is possible.
        """
        return False

    def step(self, state: GameState, ctx: SkillContext) -> StepResult:
        """Return the next move toward the nearest node, or an explore move when none.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.

        Returns:
            A ``RUNNING`` result carrying the move action (if any).
        """
        if state.minimap is None:
            return StepResult(status=SkillStatus.RUNNING, actions=())
        targets = [(obj.dx, obj.dy) for obj in state.objectives]
        action = ctx.navigator.step_toward(state.minimap, targets, now=ctx.clock())
        actions: tuple[Move, ...] = (action,) if action is not None else ()
        return StepResult(status=SkillStatus.RUNNING, actions=actions)

    def abort(self, reason: str) -> None:
        """Abort gathering without emitting actions.

        Args:
            reason: Machine-readable abort reason (for example ``"preempted"``).
        """
        del reason
