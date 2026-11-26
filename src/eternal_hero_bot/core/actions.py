"""Actions a skill can request; the controller turns them into device calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    """Hold the virtual joystick in a direction.

    Attributes:
        dx: Direction x component.
        dy: Direction y component.
        ms: Hold duration in milliseconds.
    """

    dx: float
    dy: float
    ms: int


@dataclass(frozen=True)
class Tap:
    """Tap a screen pixel.

    Attributes:
        x: Screen x coordinate.
        y: Screen y coordinate.
    """

    x: int
    y: int


@dataclass(frozen=True)
class TapNamed:
    """Tap a named manifest region/template centre.

    Attributes:
        name: Manifest region or template name.
    """

    name: str


@dataclass(frozen=True)
class Swipe:
    """Swipe between two screen pixels.

    Attributes:
        x1: Start x coordinate.
        y1: Start y coordinate.
        x2: End x coordinate.
        y2: End y coordinate.
        ms: Duration in milliseconds.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    ms: int


@dataclass(frozen=True)
class Wait:
    """Pause for a duration.

    Attributes:
        seconds: Duration in seconds.
    """

    seconds: float


type Action = Move | Tap | TapNamed | Swipe | Wait
"""Union of all actions skills may emit."""
