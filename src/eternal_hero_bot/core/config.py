"""Tuned constants for the bot, grouped into frozen dataclasses.

All values were calibrated against BlueStacks at 960x1080 portrait with the in-game
minimap at max size and default zoom. Changing them changes the bot's behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_ADB = "/mnt/c/Program Files/BlueStacks_nxt/HD-Adb.exe"
DEFAULT_DEVICE = "emulator-5554"
ADB_ENV = "EHBOT_ADB"
DEVICE_ENV = "EHBOT_DEVICE"


@dataclass(frozen=True)
class DeviceConfig:
    """How to reach the emulator over adb.

    Attributes:
        adb: Path to the adb executable (BlueStacks ships its own).
        serial: adb device serial of the BlueStacks instance.
        timeout: Seconds before an adb call is abandoned.
    """

    adb: str = DEFAULT_ADB
    serial: str = DEFAULT_DEVICE
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> DeviceConfig:
        """Build a config, letting ``EHBOT_ADB`` / ``EHBOT_DEVICE`` override the defaults.

        Returns:
            The device configuration.
        """
        return cls(
            adb=os.environ.get(ADB_ENV) or DEFAULT_ADB,
            serial=os.environ.get(DEVICE_ENV) or DEFAULT_DEVICE,
        )


@dataclass(frozen=True)
class ScreenGeometry:
    """Screen geometry for 960x1080 portrait with the minimap at max size.

    Attributes:
        mm_cx: Minimap centre x (the hero sits here).
        mm_cy: Minimap centre y.
        mm_r: Usable minimap radius in pixels.
        joy_x: Virtual joystick centre x.
        joy_y: Virtual joystick centre y.
        joy_r: Joystick push distance in pixels.
        player_r: Radius of the hero arrow on the minimap.
        revive: Screen (x, y) of the Revive button.
        death_probe: Screen (x, y) sampled to detect the green Revive button.
    """

    mm_cx: int = 127
    mm_cy: int = 292
    mm_r: int = 98
    joy_x: int = 480
    joy_y: int = 825
    joy_r: int = 90
    player_r: int = 11
    revive: tuple[int, int] = (480, 651)
    death_probe: tuple[int, int] = (400, 651)

    @property
    def mm_size(self) -> int:
        """Side length of the square minimap crop (``2 * mm_r``)."""
        return 2 * self.mm_r


@dataclass(frozen=True)
class Colours:
    """Minimap palette used for walkable-ground classification.

    Attributes:
        ground_rgb: Walkable minimap colour (mid grey-blue).
        ground_tol: Max RGB distance from ``ground_rgb`` to count as ground.
        cone_rgb: Walkable colour under the hero's view cone.
        cone_tol: Max RGB distance from ``cone_rgb`` to count as ground.
    """

    ground_rgb: tuple[int, int, int] = (81, 97, 114)
    ground_tol: int = 13
    cone_rgb: tuple[int, int, int] = (69, 84, 100)
    cone_tol: int = 9


@dataclass(frozen=True)
class PathingConfig:
    """Path planning and steering parameters (minimap pixels unless noted).

    Attributes:
        wall_margin: Clearance kept from walls.
        goal_r: The path may end this close to a node (nodes sit at ground edges).
        lookahead: Distance along the path to aim the joystick at.
        px_per_sec: Rough hero speed in minimap pixels per second.
        skip_match_r: Radius used by ``plan`` to match nodes against a skip list.
    """

    wall_margin: int = 1
    goal_r: int = 12
    lookahead: int = 12
    px_per_sec: int = 14
    skip_match_r: int = 10


@dataclass(frozen=True)
class StuckConfig:
    """Stuck detection and world-memory tuning.

    Attributes:
        stuck_px: Moving less than this per step counts as no progress.
        stuck_steps: Steps of little net movement before the way ahead is blocked.
        max_detours: Blocked detours toward one node area before skipping it.
        skip_radius: Skipping a node also skips nodes this close to it.
        skip_seconds: How long a skipped area is ignored.
        block_seconds: How long learned invisible walls are remembered.
        max_step_px: Larger per-frame tracker jumps are treated as errors.
        area_cell: Size of the coarse grid cell used to count detours per area.
    """

    stuck_px: float = 2.0
    stuck_steps: int = 3
    max_detours: int = 5
    skip_radius: int = 18
    skip_seconds: int = 180
    block_seconds: int = 300
    max_step_px: int = 25
    area_cell: int = 15


SCREEN = ScreenGeometry()
COLOURS = Colours()
PATHING = PathingConfig()
STUCK = StuckConfig()
