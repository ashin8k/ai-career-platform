from typing import Dict, List, Any, Optional
from ml.preprocessing.text_cleaner import TextCleaner
from ml.skill_extraction.section_extractor import SectionExtractor
from ml.skill_extraction.skill_extractor import SkillExtractor


class ATSAnalyzer:
    """
    ATS (Applicant Tracking System) Resume Analyzer.
    Evaluates resume formatting, structural completeness, keyword density,
    length compliance, and orphan skills to generate an ATS Score (0-100) and actionable fixes.
    """

    REQUIRED_SECTIONS = ["education", "skills", "experience", "projects"]

    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        section_extractor: Optional[SectionExtractor] = None,
        skill_extractor: Optional[SkillExtractor] = None
    ):
        self.cleaner = cleaner if cleaner is not None else TextCleaner()
        self.section_extractor = section_extractor if section_extractor is not None else SectionExtractor()
        self.skill_extractor = skill_extractor if skill_extractor is not None else SkillExtractor()

    def analyze_resume(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Executes complete ATS audit and returns score (0-100) and detailed recommendations.
        """
        improvements: List[str] = []
        deductions: float = 0.0

        # 1. Section Structure Audit
        sections = self.section_extractor.extract_sections(resume_text)
        missing_sections = [sec for sec in self.REQUIRED_SECTIONS if sec not in sections]
        
        if missing_sections:
            deductions += len(missing_sections) * 10.0
            improvements.append(
                f"Missing standard section header(s): {', '.join([s.upper() for s in missing_sections])}. "
                "Use standard bold headers so ATS scanners segment your resume properly."
            )

        # 2. Length Audit (Word Count)
        word_count = len(resume_text.split())
        if word_count < 200:
            deductions += 15.0
            improvements.append("Resume is too short (< 200 words). Add relevant project details and technical achievements.")
        elif word_count > 1200:
            deductions += 10.0
            improvements.append("Resume is excessively long (> 1200 words). Condense bullet points for a 1-page student/early-career resume.")

        # 3. Missing Keyword Audit
        resume_skills = set(self.skill_extractor.extract_skills(resume_text)["extracted_skills"])
        jd_skills = set(self.skill_extractor.extract_skills(job_description_text)["extracted_skills"])
        
        missing_keywords = sorted(list(jd_skills.difference(resume_skills)))
        if missing_keywords:
            missing_penalty = min(30.0, len(missing_keywords) * 4.0)
            deductions += missing_penalty
            improvements.append(
                f"Missing {len(missing_keywords)} key technical keyword(s) from target JD: "
                f"{', '.join(missing_keywords[:6])}. Include these naturally in your skills and project descriptions."
            )

        # 4. Orphan Skills Audit (Skills in list but not in experience/projects body)
        skills_sec_text = sections.get("skills", "")
        body_text = sections.get("experience", "") + " " + sections.get("projects", "")
        
        skills_sec_skills = set(self.skill_extractor.extract_skills(skills_sec_text)["extracted_skills"])
        body_skills = set(self.skill_extractor.extract_skills(body_text)["extracted_skills"])
        
        orphan_skills = sorted(list(skills_sec_skills.difference(body_skills)))
        if orphan_skills:
            deductions += min(15.0, len(orphan_skills) * 3.0)
            improvements.append(
                f"{len(orphan_skills)} skill(s) listed under Skills are not demonstrated in Experience/Projects: "
                f"{', '.join(orphan_skills[:5])}. Add project bullet points proving where you applied them."
            )

        # Compute Final ATS Score
        ats_score = max(0.0, min(100.0, 100.0 - deductions))

        return {
            "ats_score": round(ats_score, 1),
            "word_count": word_count,
            "section_structure": {
                "detected_sections": list(sections.keys()),
                "missing_sections": missing_sections,
                "is_complete": len(missing_sections) == 0
            },
            "missing_keywords": missing_keywords,
            "orphan_skills": orphan_skills,
            "improvements": improvements
        }
