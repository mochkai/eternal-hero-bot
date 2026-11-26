"""Gather and revive skills, stubs, and the registry that builds them."""

from eternal_hero_bot.skills.base import Skill, SkillContext
from eternal_hero_bot.skills.gather import GatherSkill
from eternal_hero_bot.skills.registry import SKILLS, build_skills, register
from eternal_hero_bot.skills.revive import ReviveSkill
from eternal_hero_bot.skills.stubs import (
    CompleteAltarSkill,
    DailyTasksSkill,
    FindTreasureSkill,
    HuntSkill,
    SpendResourceSkill,
)

__all__ = [
    "SKILLS",
    "CompleteAltarSkill",
    "DailyTasksSkill",
    "FindTreasureSkill",
    "GatherSkill",
    "HuntSkill",
    "ReviveSkill",
    "Skill",
    "SkillContext",
    "SpendResourceSkill",
    "build_skills",
    "register",
]
