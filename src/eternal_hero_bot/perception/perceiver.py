"""Perception: turn a captured frame into a :class:`~eternal_hero_bot.core.state.GameState`."""

from __future__ import annotations

from typing import Protocol

from eternal_hero_bot.core.config import (
    PATHING,
    SCREEN,
    STUCK,
    PathingConfig,
    ScreenGeometry,
    StuckConfig,
)
from eternal_hero_bot.core.state import GameState, MinimapView, Objective, ScreenKind
from eternal_hero_bot.perception.frame import Frame
from eternal_hero_bot.perception.tracker import Tracker
from eternal_hero_bot.perception.vision import find_nodes, is_dead, walkable_mask


class Perceiver(Protocol):
    """Produces immutable game-state snapshots from frames."""

    def perceive(self, frame: Frame) -> GameState:
        """Perceive a frame into a game state.

        Args:
            frame: Captured screenshot.

        Returns:
            The perceived game state.
        """
        ...

    def reset(self) -> None:
        """Reset tracking state (new tracker, epoch += 1)."""
        ...


class DefaultPerceiver:
    """Default perceiver using the existing vision and tracker logic.

    It preserves gather parity: death detection uses the probe pixel, world frames
    update the phase-correlation tracker and expose node objectives.
    """

    def __init__(
        self,
        screen: ScreenGeometry = SCREEN,
        pathing: PathingConfig = PATHING,
        stuck: StuckConfig = STUCK,
        assets: object | None = None,
        ocr: object | None = None,
    ) -> None:
        """Create a perceiver with a fresh tracker.

        Args:
            screen: Screen geometry.
            pathing: Path planning parameters (wall margin).
            stuck: Stuck/tracker parameters (max step).
            assets: Reserved for template/manifest assets (step 7+).
            ocr: Reserved for OCR reader (step 8+).
        """
        self._screen = screen
        self._pathing = pathing
        self._stuck = stuck
        self._assets = assets
        self._ocr = ocr
        self._tracker = Tracker(screen, stuck.max_step_px)
        self._map_epoch = 0
        self._dead_seen = False

    @property
    def tracker(self) -> Tracker:
        """The underlying phase-correlation tracker."""
        return self._tracker

    @property
    def map_epoch(self) -> int:
        """Current map epoch, bumped on tracker resets."""
        return self._map_epoch

    def reset(self) -> None:
        """Reset tracking state: new tracker and bump map epoch."""
        self._tracker = Tracker(self._screen, self._stuck.max_step_px)
        self._map_epoch += 1
        self._dead_seen = False

    def perceive(self, frame: Frame) -> GameState:
        """Perceive a frame into a :class:`GameState`.

        On the first DEAD frame, resets the tracker and bumps the map epoch, returning
        no minimap and no objectives (tracker is not updated). Non-dead frames are
        classified as WORLD in v1 parity mode.

        Args:
            frame: Captured screenshot.

        Returns:
            The immutable game-state snapshot.
        """
        img = frame.image
        if is_dead(img, self._screen):
            if not self._dead_seen:
                self._tracker = Tracker(self._screen, self._stuck.max_step_px)
                self._map_epoch += 1
                self._dead_seen = True
            return GameState(
                t=frame.t,
                frame_index=frame.index,
                screen=ScreenKind.DEAD,
                map_epoch=self._map_epoch,
                minimap=None,
                objectives=(),
                resources={},
                detections={},
            )

        self._dead_seen = False
        walk_raw = walkable_mask(img, self._screen, wall_margin=self._pathing.wall_margin)
        self._tracker.update(walk_raw.astype(float))
        pos = self._tracker.pos
        minimap_view = MinimapView(
            walk=walk_raw,
            hero_pos=(float(pos[0]), float(pos[1])),
        )
        objectives = tuple(
            Objective(kind="node", dx=float(dx), dy=float(dy), size=int(size))
            for dx, dy, size in find_nodes(img, self._screen)
        )
        return GameState(
            t=frame.t,
            frame_index=frame.index,
            screen=ScreenKind.WORLD,
            map_epoch=self._map_epoch,
            minimap=minimap_view,
            objectives=objectives,
            resources={},
            detections={},
        )
