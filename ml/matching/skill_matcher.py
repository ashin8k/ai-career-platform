from typing import Dict, List, Set, Any
from ml.skill_extraction.skill_extractor import SkillExtractor


class SkillMatcher:
    """
    Skill Set Overlap Matcher.
    Uses canonical skill set intersection and Jaccard similarity to quantify hard-skill coverage,
    missing skill gaps, and bonus skills.
    """

    def __init__(self, skill_extractor: Optional[SkillExtractor] = None):
        self.extractor = skill_extractor if skill_extractor is not None else SkillExtractor()

    def match_skills(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Extracts skills from both resume and job description, then computes set metrics.
        """
        resume_res = self.extractor.extract_skills(resume_text)
        jd_res = self.extractor.extract_skills(job_description_text)

        s_resume: Set[str] = set(resume_res["extracted_skills"])
        s_jd: Set[str] = set(jd_res["extracted_skills"])

        matching_skills = sorted(list(s_resume.intersection(s_jd)))
        missing_skills = sorted(list(s_jd.difference(s_resume)))
        extra_skills = sorted(list(s_resume.difference(s_jd)))

        # 1. Coverage Ratio: matching_skills / required_jd_skills
        coverage_ratio = len(matching_skills) / len(s_jd) if s_jd else 1.0

        # 2. Jaccard Index: matching_skills / total_union_skills
        union_size = len(s_resume.union(s_jd))
        jaccard_index = len(matching_skills) / union_size if union_size > 0 else 1.0

        return {
            "score": round(coverage_ratio, 4),
            "percentage": round(coverage_ratio * 100, 2),
            "jaccard_index": round(jaccard_index, 4),
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "resume_skill_count": len(s_resume),
            "jd_skill_count": len(s_jd),
            "matching_count": len(matching_skills),
            "missing_count": len(missing_skills)
        }
