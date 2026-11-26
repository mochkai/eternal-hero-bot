"""Immutable game-state snapshots produced by perception."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from eternal_hero_bot.core.arrays import Mask
from eternal_hero_bot.core.geometry import Box


class ScreenKind(StrEnum):
    """Coarse classification of what the screen is showing."""

    WORLD = "world"
    """Minimap visible, hero controllable."""
    DEAD = "dead"
    """Death dialog visible."""
    UNKNOWN = "unknown"
    """Anything else (popups, menus); future skills handle it."""


@dataclass(frozen=True)
class Objective:
    """A target the bot can navigate to or interact with.

    Attributes:
        kind: Target type, such as ``"node"``.
        dx: Hero-relative x offset in minimap pixels.
        dy: Hero-relative y offset in minimap pixels.
        size: Detector-specific weight, such as node handle pixel count.
    """

    kind: str
    dx: float
    dy: float
    size: int = 0


@dataclass(frozen=True)
class ResourceReading:
    """A HUD resource reading.

    Attributes:
        name: Resource identifier, such as ``"gold"``.
        value: Current value, or ``None`` when unreadable this frame.
        max: Maximum/cap value, or ``None`` when unknown.
        confidence: Reader confidence in ``[0, 1]``.
    """

    name: str
    value: int | None
    max: int | None
    confidence: float

    @property
    def at_max(self) -> bool:
        """Whether the resource is at or above its cap."""
        return self.value is not None and self.max is not None and self.value >= self.max


@dataclass(frozen=True)
class Detection:
    """A template detection on a frame.

    Attributes:
        name: Template name.
        box: Bounding box in frame pixels.
        score: Match score.
    """

    name: str
    box: Box
    score: float


@dataclass(frozen=True)
class MinimapView:
    """Perception derived from the minimap on a world frame.

    Attributes:
        walk: Walkable mask in minimap-crop coordinates. Read-only.
        hero_pos: Tracked hero world position ``(x, y)`` in minimap pixels.
    """

    walk: Mask
    hero_pos: tuple[float, float]

    def __post_init__(self) -> None:
        """Mark the walkable mask as read-only."""
        self.walk.flags.writeable = False


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of what the bot knows about the current frame.

    Attributes:
        t: Frame capture time in seconds.
        frame_index: Monotonic frame index within the runtime.
        screen: Coarse screen classification.
        map_epoch: Bumps when the tracker resets (death/respawn).
        minimap: Minimap-derived view, or ``None`` outside world screens.
        objectives: Visible objectives.
        resources: HUD resource readings keyed by resource name.
        detections: Template detections keyed by template name.
    """

    t: float
    frame_index: int
    screen: ScreenKind
    map_epoch: int
    minimap: MinimapView | None
    objectives: tuple[Objective, ...]
    resources: Mapping[str, ResourceReading]
    detections: Mapping[str, tuple[Detection, ...]]
