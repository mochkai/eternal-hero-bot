"""Gather and revive skills, and the registry that builds them."""

from eternal_hero_bot.skills.base import Skill, SkillContext
from eternal_hero_bot.skills.gather import GatherSkill
from eternal_hero_bot.skills.registry import SKILLS, build_skills, register
from eternal_hero_bot.skills.revive import ReviveSkill

__all__ = [
    "SKILLS",
    "GatherSkill",
    "ReviveSkill",
    "Skill",
    "SkillContext",
    "build_skills",
    "register",
]
