"""Skill registry: the ``@register`` decorator and the config-driven skill factory."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from eternal_hero_bot.skills.base import Skill

SkillT = TypeVar("SkillT", bound="Skill")
"""Type variable for registered skill classes."""

SkillFactory = Callable[[Mapping[str, object]], Skill]
"""Builds a skill from its ``[skills.<name>]`` options table."""

SKILLS: dict[str, SkillFactory] = {}
"""Registered skill factories, keyed by skill name."""


def register(name: str) -> Callable[[type[SkillT]], type[SkillT]]:
    """Class decorator registering a skill factory under ``name``.

    Args:
        name: Registered skill name (the ``[skills.<name>]`` config table).

    Returns:
        A decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[SkillT]) -> type[SkillT]:
        def factory(options: Mapping[str, object]) -> Skill:
            # The protocol itself cannot be instantiated; each registered class takes options.
            return cls(options)  # type: ignore[call-arg]

        SKILLS[name] = factory
        return cls

    return decorator


def build_skills(config: Mapping[str, Mapping[str, object]]) -> dict[str, Skill]:
    """Build every registered skill with its options table from ``config``.

    Args:
        config: Brain/skill configuration keyed by skill name, each holding that
            skill's ``[skills.<name>]`` options table.

    Returns:
        Mapping of skill name to built skill.

    Raises:
        ValueError: If ``config`` names an unknown skill.
    """
    for name in config:
        if name not in SKILLS:
            raise ValueError(f"unknown skill: {name!r}")
    return {name: factory(config.get(name, {})) for name, factory in SKILLS.items()}
