import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.reasoning.roadmap_generator import LearningRoadmapGenerator
from ml.reasoning.project_recommender import ProjectRecommender
from ml.reasoning.interview_generator import InterviewGenerator
from ml.reasoning.ai_explainer import AIExplainer


def run_phase5_tests():
    print("=" * 60)
    print("RUNNING PHASE 5 AI REASONING & RECOMMENDATION ENGINE VERIFICATION")
    print("=" * 60)

    # Test 1: Personalized Learning Roadmap
    roadmap_gen = LearningRoadmapGenerator()
    roadmap = roadmap_gen.generate_roadmap(["fastapi", "docker", "aws"])
    assert len(roadmap) == 3
    print(f"[PASS] Test 1: Personalized Learning Roadmap (Weeks planned: {len(roadmap)}, Week 1: {roadmap[0]['title']})")

    # Test 2: Project Recommender
    recommender = ProjectRecommender()
    projects = recommender.recommend_projects(["fastapi", "docker"], ["python", "sql"])
    assert len(projects) > 0
    print(f"[PASS] Test 2: Project Recommendation Engine (Top Rec: {projects[0]['title']})")

    # Test 3: Customized Interview Questions
    interview_gen = InterviewGenerator()
    questions = interview_gen.generate_interview_questions(["python", "sql"], ["fastapi", "docker"])
    assert len(questions["easy"]) > 0 and len(questions["medium"]) > 0 and len(questions["difficult"]) > 0
    print(f"[PASS] Test 3: Customized Interview Question Generator (Easy: {len(questions['easy'])}, Medium: {len(questions['medium'])}, Difficult: {len(questions['difficult'])})")

    # Test 4: AI Explainer Synthesizer
    explainer = AIExplainer()
    match_res = {"overall_match_percentage": 75.0, "matching_skills": ["python", "sql"], "missing_skills": ["docker", "aws"]}
    ats_res = {"ats_score": 85.0, "improvements": ["Add Docker bullet points."]}
    report = explainer.generate_full_analysis("resume", "jd", match_res, ats_res)
    assert "summary_explanation" in report and "learning_roadmap" in report
    print(f"[PASS] Test 4: Full AI Report Synthesizer (Score: {report['summary_explanation']['overall_match_percentage']}%)")

    print("\nALL PHASE 5 TESTS PASSED SUCCESSFULLY! (4/4)")
    print("=" * 60)


if __name__ == "__main__":
    run_phase5_tests()
