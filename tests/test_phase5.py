import pytest
from ml.reasoning.roadmap_generator import LearningRoadmapGenerator
from ml.reasoning.project_recommender import ProjectRecommender
from ml.reasoning.interview_generator import InterviewGenerator
from ml.reasoning.ai_explainer import AIExplainer


def test_roadmap_generator():
    generator = LearningRoadmapGenerator()
    roadmap = generator.generate_roadmap(["fastapi", "docker", "aws"])

    assert len(roadmap) == 3
    assert roadmap[0]["week"] == 1
    assert "FastAPI" in roadmap[0]["title"]
    assert "Docker" in roadmap[1]["title"]


def test_project_recommender():
    recommender = ProjectRecommender()
    projects = recommender.recommend_projects(["fastapi", "docker"], ["python", "sql"])

    assert len(projects) > 0
    assert "FastAPI" in projects[0]["title"] or "Docker" in projects[0]["title"]
    assert len(projects[0]["gaps_addressed"]) > 0


def test_interview_generator():
    generator = InterviewGenerator()
    questions = generator.generate_interview_questions(["python", "sql"], ["fastapi", "docker"])

    assert "easy" in questions
    assert "medium" in questions
    assert "difficult" in questions
    assert len(questions["easy"]) > 0


def test_ai_explainer():
    explainer = AIExplainer()
    match_res = {
        "overall_match_percentage": 75.0,
        "matching_skills": ["python", "sql"],
        "missing_skills": ["docker", "aws"]
    }
    ats_res = {
        "ats_score": 85.0,
        "improvements": ["Add Docker project details."]
    }

    full_report = explainer.generate_full_analysis("sample resume", "sample jd", match_res, ats_res)

    assert "summary_explanation" in full_report
    assert "learning_roadmap" in full_report
    assert "recommended_projects" in full_report
    assert "interview_preparation" in full_report
