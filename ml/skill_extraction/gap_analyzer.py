from typing import Dict, List, Set, Any
from ml.skill_extraction.skill_extractor import SkillExtractor
from ml.skill_extraction.section_extractor import SectionExtractor


class SkillGapAnalyzer:
    """
    Skill Gap Categorization Engine.
    Categorizes skills into Matching, Missing, Partial, Must-Have vs Good-To-Have,
    and analyzes section depth (e.g. skills mentioned only in Skills section vs supported in Projects/Experience).
    """

    def __init__(
        self,
        skill_extractor: Optional[SkillExtractor] = None,
        section_extractor: Optional[SectionExtractor] = None
    ):
        self.skill_extractor = skill_extractor if skill_extractor is not None else SkillExtractor()
        self.section_extractor = section_extractor if section_extractor is not None else SectionExtractor()

    def analyze_gap(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Calculates skill gaps, categorizes requirements into Must-Have vs Good-To-Have,
        and identifies partial skill evidence.
        """
        # Extract skills from full texts
        resume_skills_res = self.skill_extractor.extract_skills(resume_text)
        jd_skills_res = self.skill_extractor.extract_skills(job_description_text)

        s_resume: Set[str] = set(resume_skills_res["extracted_skills"])
        s_jd: Set[str] = set(jd_skills_res["extracted_skills"])

        matching_skills = sorted(list(s_resume.intersection(s_jd)))
        missing_skills = sorted(list(s_jd.difference(s_resume)))

        # Extract resume sections for depth checking
        sections = self.section_extractor.extract_sections(resume_text)
        skills_section_text = sections.get("skills", "")
        experience_text = sections.get("experience", "") + " " + sections.get("projects", "")

        # Extract skills specifically inside Experience and Projects sections
        exp_skills_set = set(self.skill_extractor.extract_skills(experience_text)["extracted_skills"])
        skills_sec_set = set(self.skill_extractor.extract_skills(skills_section_text)["extracted_skills"])

        # PARTIAL SKILLS: Listed in skills section but NOT backed by projects or experience
        partial_skills = sorted(list(skills_sec_set.difference(exp_skills_set)))
        
        # Categorize Must-Have vs Good-To-Have skills based on JD context keywords
        must_have_skills, good_to_have_skills = self._categorize_jd_priority(job_description_text, s_jd)

        return {
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "partial_skills": partial_skills,
            "must_have_skills": must_have_skills,
            "good_to_have_skills": good_to_have_skills,
            "matching_count": len(matching_skills),
            "missing_count": len(missing_skills),
            "partial_count": len(partial_skills),
            "skill_match_rate": round(len(matching_skills) / len(s_jd) * 100, 2) if s_jd else 100.0
        }

    def _categorize_jd_priority(self, jd_text: str, jd_skills: Set[str]) -> Tuple[List[str], List[str]]:
        """
        Splits job description skills into Must-Have vs Good-To-Have
        by looking for section keywords (e.g. 'preferred', 'nice to have', 'bonus').
        """
        lines = jd_text.splitlines()
        must_have = set()
        good_to_have = set()

        current_priority = "must_have"
        for line in lines:
            line_lower = line.lower()
            if any(k in line_lower for k in ["preferred", "nice to have", "good to have", "bonus", "plus"]):
                current_priority = "good_to_have"
            elif any(k in line_lower for k in ["required", "must have", "minimum qualifications", "requirements"]):
                current_priority = "must_have"

            line_skills = set(self.skill_extractor.extract_skills(line)["extracted_skills"])
            if current_priority == "good_to_have":
                good_to_have.update(line_skills)
            else:
                must_have.update(line_skills)

        # Ensure all JD skills are classified
        for skill in jd_skills:
            if skill not in good_to_have:
                must_have.add(skill)

        return sorted(list(must_have)), sorted(list(good_to_have))
