"""
Skill Extraction Subpackage: Section segmentation, taxonomy matching, and skill categorization.
"""
from .taxonomy import SKILL_TAXONOMY, SKILL_ALIASES, normalize_skill_name
from .section_extractor import SectionExtractor
from .skill_extractor import SkillExtractor

__all__ = [
    "SKILL_TAXONOMY",
    "SKILL_ALIASES",
    "normalize_skill_name",
    "SectionExtractor",
    "SkillExtractor"
]
