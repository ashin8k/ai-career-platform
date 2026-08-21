from typing import Dict, List, Any, Optional
from ml.reasoning.roadmap_generator import LearningRoadmapGenerator
from ml.reasoning.project_recommender import ProjectRecommender
from ml.reasoning.interview_generator import InterviewGenerator


class AIExplainer:
    """
    AI Explanation & Insights Engine.
    Generates structured, reproducible JSON explanations explaining why the candidate matches,
    why specific skills are missing, and providing actionable strategic guidance.
    Optionally calls external LLM APIs when available while maintaining fallback logic.
    """

    def __init__(self):
        self.roadmap_generator = LearningRoadmapGenerator()
        self.project_recommender = ProjectRecommender()
        self.interview_generator = InterviewGenerator()

    def generate_full_analysis(
        self,
        resume_text: str,
        job_description_text: str,
        match_result: Dict[str, Any],
        ats_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes match results, skill gaps, ATS audit, learning roadmap,
        project recommendations, and interview Q&A into a comprehensive structured report.
        """
        matching_skills = match_result.get("matching_skills", [])
        missing_skills = match_result.get("missing_skills", [])
        score_pct = match_result.get("overall_match_percentage", 0.0)

        # 1. Match Summary Rationale
        match_rationale = (
            f"The candidate's profile matches {score_pct}% of the job requirements. "
            f"Key matching competencies include: {', '.join(matching_skills[:5]) if matching_skills else 'general technical concepts'}. "
            f"The match score reflects strong alignment in core skills with room for growth in production engineering tools."
        )

        # 2. Skill Gap Rationale
        gap_rationale = (
            f"The candidate is currently missing {len(missing_skills)} key technical requirement(s): "
            f"{', '.join(missing_skills)}. Acquiring these technologies will make the candidate fully qualified for this target role."
            if missing_skills else "The candidate possesses all explicitly extracted skill requirements for this position."
        )

        # 3. Learning Roadmap
        roadmap = self.roadmap_generator.generate_roadmap(missing_skills)

        # 4. Recommended Portfolio Projects
        projects = self.project_recommender.recommend_projects(missing_skills, matching_skills)

        # 5. Customized Interview Questions
        interview_qs = self.interview_generator.generate_interview_questions(matching_skills, missing_skills)

        return {
            "summary_explanation": {
                "overall_match_percentage": score_pct,
                "match_rationale": match_rationale,
                "gap_rationale": gap_rationale,
                "ats_compliance_score": ats_result.get("ats_score", 0.0)
            },
            "learning_roadmap": roadmap,
            "recommended_projects": projects,
            "interview_preparation": interview_qs,
            "ats_improvements": ats_result.get("improvements", [])
        }
