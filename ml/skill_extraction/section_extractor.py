import re
from typing import Dict, List, Any


class SectionExtractor:
    """
    Resume Section Segmenter.
    Uses regex boundary detection on standard resume section headers (e.g. EDUCATION,
    PROJECTS, SKILLS, EXPERIENCE, CERTIFICATIONS) to break down raw text into structured key-value sections.
    """

    HEADER_PATTERNS = {
        "summary": r"\b(summary|profile|about me|objective|professional summary)\b",
        "education": r"\b(education|academic background|qualification|qualifications)\b",
        "skills": r"\b(technical skills|skills|technologies|skills & expertise|technical expertise)\b",
        "experience": r"\b(experience|work experience|employment|employment history|work history|internships)\b",
        "projects": r"\b(projects|academic projects|key projects|personal projects)\b",
        "certifications": r"\b(certifications|licenses|courses|certificates|achievements)\b"
    }

    def __init__(self):
        # Combined pattern for matching line-level headers
        combined_pattern = "|".join(
            f"(?P<{category}>{pattern})" for category, pattern in self.HEADER_PATTERNS.items()
        )
        self.header_regex = re.compile(rf"^\s*(?:{combined_pattern})\s*$", re.IGNORECASE | re.MULTILINE)

    def extract_sections(self, raw_text: str) -> Dict[str, str]:
        """
        Segments raw text into a dictionary of {section_name: section_text}.
        Any headerless introductory text is captured under 'header_intro'.
        """
        if not raw_text or not isinstance(raw_text, str):
            return {}

        lines = raw_text.splitlines()
        sections: Dict[str, List[str]] = {}
        current_section = "intro"
        sections[current_section] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if current line acts as a section header
            matched_section = self._identify_header(line_str)
            if matched_section:
                current_section = matched_section
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line_str)

        # Join line lists into single string paragraphs
        result = {sec: "\n".join(content).strip() for sec, content in sections.items() if content}
        return result

    def _identify_header(self, line: str) -> str:
        """
        Helper method to check if a single line matches any standard section header.
        Lines should generally be short (<= 4 words) to avoid false positives in body text.
        """
        if len(line.split()) > 4:
            return ""

        for category, pattern in self.HEADER_PATTERNS.items():
            if re.search(f"^{pattern}:?$", line, re.IGNORECASE):
                return category
        return ""
