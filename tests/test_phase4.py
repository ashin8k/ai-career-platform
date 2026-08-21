import pytest
from ml.skill_extraction.gap_analyzer import SkillGapAnalyzer
from ml.skill_extraction.ats_analyzer import ATSAnalyzer

SAMPLE_RESUME = """
Alex Mercer
Summary
Passionate Machine Learning Student.

EDUCATION
BS Computer Science

SKILLS
Python, SQL, Machine Learning, Docker, AWS, Kubernetes

PROJECTS
Predictive Customer Churn Engine: Built model using Python, Scikit-learn, SQL.

EXPERIENCE
ML Intern: Data preprocessing using Python and Pandas.
"""

SAMPLE_JD = """
Job Title: Machine Learning & Backend Developer
Requirements:
- Strong skills in Python, SQL, and Machine Learning
- Hands-on experience with FastAPI and Docker
Preferred Qualifications:
- Experience with AWS and PostgreSQL
"""


def test_skill_gap_analyzer():
    analyzer = SkillGapAnalyzer()
    res = analyzer.analyze_gap(SAMPLE_RESUME, SAMPLE_JD)

    assert "python" in res["matching_skills"]
    assert "sql" in res["matching_skills"]
    assert "fastapi" in res["missing_skills"]
    assert "docker" in res["partial_skills"] or "aws" in res["partial_skills"]
    assert res["skill_match_rate"] > 50.0


def test_ats_analyzer():
    analyzer = ATSAnalyzer()
    res = analyzer.analyze_resume(SAMPLE_RESUME, SAMPLE_JD)

    assert 0 <= res["ats_score"] <= 100
    assert "fastapi" in res["missing_keywords"]
    assert "docker" in res["orphan_skills"] or "aws" in res["orphan_skills"] or "kubernetes" in res["orphan_skills"]
    assert len(res["improvements"]) > 0
