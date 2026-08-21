import pytest
from ml.matching.tfidf_matcher import TFIDFMatcher
from ml.matching.skill_matcher import SkillMatcher
from ml.matching.hybrid_matcher import HybridMatcher


SAMPLE_RESUME = """
Alex Mercer - AI & Software Engineering Student
Skills: Python, C++, SQL, PyTorch, Scikit-learn, FastAPI, Docker, PostgreSQL, Machine Learning
Experience: ML Intern building predictive models and FastAPI backends.
Projects: AI Resume Matcher using Scikit-learn, Sentence Transformers, Docker.
"""

SAMPLE_JD = """
Job Title: Machine Learning & Backend Engineer
Requirements:
- 1+ years experience with Python and SQL
- Strong knowledge of Machine Learning algorithms and Scikit-learn
- Hands-on experience with FastAPI, Docker, PostgreSQL, and AWS
- Familiarity with CI/CD and PyTorch
"""


def test_tfidf_matcher():
    matcher = TFIDFMatcher()
    res = matcher.calculate_similarity(SAMPLE_RESUME, SAMPLE_JD)
    
    assert res["score"] > 0.1
    assert "top_shared_terms" in res
    assert len(res["top_shared_terms"]) > 0


def test_skill_matcher():
    matcher = SkillMatcher()
    res = matcher.match_skills(SAMPLE_RESUME, SAMPLE_JD)
    
    assert "python" in res["matching_skills"]
    assert "fastapi" in res["matching_skills"]
    assert "docker" in res["matching_skills"]
    assert "aws" in res["missing_skills"]
    assert res["matching_count"] >= 5


def test_hybrid_matcher():
    # Use lazy loading for fast tests
    matcher = HybridMatcher(lazy_load_embedding=True)
    res = matcher.match(SAMPLE_RESUME, SAMPLE_JD)
    
    assert "overall_match_percentage" in res
    assert 0 <= res["overall_match_score"] <= 1.0
    assert "python" in res["matching_skills"]
    assert "aws" in res["missing_skills"]
    assert "approach_comparison" in res
