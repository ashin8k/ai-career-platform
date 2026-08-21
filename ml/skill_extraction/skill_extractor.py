import re
from typing import Dict, List, Set, Any
from ml.preprocessing.text_cleaner import TextCleaner
from ml.skill_extraction.taxonomy import SKILL_TAXONOMY, normalize_skill_name


class SkillExtractor:
    """
    Skill Extraction Engine.
    Combines taxonomy-driven multi-word (n-gram) matching and exact word boundary detection
    to identify technical skills and categorize them cleanly.
    """

    def __init__(self, cleaner: Optional[TextCleaner] = None):
        self.cleaner = cleaner if cleaner is not None else TextCleaner()
        self._build_extraction_patterns()

    def _build_extraction_patterns(self):
        """
        Sort skills by token length (descending) so multi-word terms like
        'natural language processing' match before single tokens like 'language' or 'processing'.
        """
        all_skills = []
        for category, skills in SKILL_TAXONOMY.items():
            for skill in skills:
                all_skills.append((skill.lower(), category))

        # Sort multi-word skills first
        all_skills.sort(key=lambda x: len(x[0].split()), reverse=True)
        self.skill_category_map = {skill: cat for skill, cat in all_skills}
        self.sorted_skills = [skill for skill, _ in all_skills]

    def extract_skills(self, text: str) -> Dict[str, Any]:
        """
        Extracts skills from text, normalizes aliases, and groups them by category.
        """
        if not text or not isinstance(text, str):
            return {
                "extracted_skills": [],
                "categorized": {},
                "total_unique_skills": 0
            }

        cleaned_text = self.cleaner.clean_text(text)
        found_skills: Set[str] = set()

        for skill in self.sorted_skills:
            # Handle special characters in regex patterns (like c++, c#)
            escaped_skill = re.escape(skill)
            
            # Match exact word boundaries
            pattern = rf"(?<!\w){escaped_skill}(?!\w)"
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                canonical_name = normalize_skill_name(skill)
                found_skills.add(canonical_name)

        # Categorize found skills
        categorized: Dict[str, List[str]] = {cat: [] for cat in SKILL_TAXONOMY.keys()}
        for skill in sorted(found_skills):
            category = self._get_skill_category(skill)
            categorized[category].append(skill)

        # Filter out empty categories
        categorized = {k: v for k, v in categorized.items() if v}

        return {
            "extracted_skills": sorted(list(found_skills)),
            "categorized": categorized,
            "total_unique_skills": len(found_skills)
        }

    def _get_skill_category(self, skill: str) -> str:
        """Find category of a canonical skill name."""
        if skill in self.skill_category_map:
            return self.skill_category_map[skill]

        # Check raw taxonomy for match
        for cat, skills in SKILL_TAXONOMY.items():
            if skill in skills or skill in [normalize_skill_name(s) for s in skills]:
                return cat
        return "tools_methods"
