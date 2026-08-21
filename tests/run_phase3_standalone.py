import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.matching.tfidf_matcher import TFIDFMatcher
from ml.matching.skill_matcher import SkillMatcher
from ml.matching.hybrid_matcher import HybridMatcher

SAMPLE_RESUME = """
Alex Mercer - Computer Science & AI Engineering Student
Skills: Python, C++, SQL, PyTorch, Scikit-learn, FastAPI, Docker, PostgreSQL, Machine Learning
Experience: ML Intern developing recommendation models and FastAPI endpoints.
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


def run_phase3_tests():
    print("=" * 60)
    print("RUNNING PHASE 3 MATCHING ENGINE VERIFICATION SUITE")
    print("=" * 60)

    # Test 1: TF-IDF Lexical Matcher
    tfidf = TFIDFMatcher()
    res_tfidf = tfidf.calculate_similarity(SAMPLE_RESUME, SAMPLE_JD)
    assert res_tfidf["score"] > 0.1, f"TF-IDF score too low: {res_tfidf}"
    assert len(res_tfidf["top_shared_terms"]) > 0, "No shared TF-IDF terms found"
    print(f"[PASS] Test 1: TF-IDF Cosine Similarity (Score: {res_tfidf['percentage']}%, Shared terms: {res_tfidf['top_shared_terms'][:5]})")

    # Test 2: Skill Set Graph Intersection Matcher
    skill_m = SkillMatcher()
    res_skill = skill_m.match_skills(SAMPLE_RESUME, SAMPLE_JD)
    assert "python" in res_skill["matching_skills"]
    assert "fastapi" in res_skill["matching_skills"]
    assert "docker" in res_skill["matching_skills"]
    assert "aws" in res_skill["missing_skills"]
    print(f"[PASS] Test 2: Skill Set Matcher (Match Score: {res_skill['percentage']}%, Matching: {len(res_skill['matching_skills'])}, Missing: {len(res_skill['missing_skills'])})")

    # Test 3: Hybrid Matcher (Lexical + Skill Graph)
    # Using TFIDF + Skill weights without triggering model download in quick script
    hybrid = HybridMatcher(weight_skill=0.5, weight_embed=0.0, weight_tfidf=0.5)
    res_hybrid = hybrid.match(SAMPLE_RESUME, SAMPLE_JD)
    assert res_hybrid["overall_match_score"] > 0.3
    assert "python" in res_hybrid["matching_skills"]
    assert "aws" in res_hybrid["missing_skills"]
    assert "approach_comparison" in res_hybrid
    print(f"[PASS] Test 3: Ensemble Hybrid Score (Overall Match: {res_hybrid['overall_match_percentage']}%)")

    print("\nALL PHASE 3 MATCHING TESTS PASSED SUCCESSFULLY! (3/3)")
    print("=" * 60)


if __name__ == "__main__":
    run_phase3_tests()
