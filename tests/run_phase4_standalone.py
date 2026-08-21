import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


def run_phase4_tests():
    print("=" * 60)
    print("RUNNING PHASE 4 SKILL GAP & ATS ANALYZER VERIFICATION SUITE")
    print("=" * 60)

    # Test 1: Skill Gap Analysis
    gap_analyzer = SkillGapAnalyzer()
    gap_res = gap_analyzer.analyze_gap(SAMPLE_RESUME, SAMPLE_JD)
    
    assert "python" in gap_res["matching_skills"]
    assert "fastapi" in gap_res["missing_skills"]
    print(f"[PASS] Test 1: Skill Gap Categorization (Matching: {gap_res['matching_skills']}, Missing: {gap_res['missing_skills']}, Partial: {gap_res['partial_skills']})")

    # Test 2: ATS Compliance Audit
    ats_analyzer = ATSAnalyzer()
    ats_res = ats_analyzer.analyze_resume(SAMPLE_RESUME, SAMPLE_JD)
    
    assert ats_res["ats_score"] > 50.0
    assert "fastapi" in ats_res["missing_keywords"]
    assert len(ats_res["improvements"]) > 0
    print(f"[PASS] Test 2: ATS Resume Score (Score: {ats_res['ats_score']}/100, Word Count: {ats_res['word_count']}, Improvements count: {len(ats_res['improvements'])})")

    print("\nALL PHASE 4 TESTS PASSED SUCCESSFULLY! (2/2)")
    print("=" * 60)


if __name__ == "__main__":
    run_phase4_tests()
