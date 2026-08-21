import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def run_phase6_tests():
    print("=" * 60)
    print("RUNNING PHASE 6 FASTAPI BACKEND API & DATABASE VERIFICATION")
    print("=" * 60)

    # Test 1: Root Endpoint
    res = client.get("/")
    assert res.status_code == 200
    print(f"[PASS] Test 1: GET / (Status: {res.json()['status']})")

    # Test 2: Job Analysis Endpoint
    jd_payload = {
        "title": "Machine Learning Engineer",
        "job_description_text": "Required skills include Python, SQL, Machine Learning, FastAPI, Docker, and PostgreSQL."
    }
    jd_res = client.post("/api/job/analyze", json=jd_payload)
    assert jd_res.status_code == 200
    jd_id = jd_res.json()["jd_id"]
    print(f"[PASS] Test 2: POST /api/job/analyze (JD ID: {jd_id[:8]}..., Extracted skills: {jd_res.json()['extracted_skills']})")

    # Test 3: Resume PDF Upload Endpoint
    sample_pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_resume.pdf")
    if not os.path.exists(sample_pdf_path):
        # Create sample pdf if not present
        from data.sample.generate_sample_pdf import create_sample_pdf
        create_sample_pdf(sample_pdf_path)

    with open(sample_pdf_path, "rb") as f:
        upload_res = client.post("/api/resume/upload", files={"file": ("resume.pdf", f, "application/pdf")})
    assert upload_res.status_code == 200
    resume_id = upload_res.json()["resume_id"]
    print(f"[PASS] Test 3: POST /api/resume/upload (Resume ID: {resume_id[:8]}..., Skills count: {upload_res.json()['total_skills_count']})")

    # Test 4: Match Analysis Endpoint
    match_payload = {"resume_id": resume_id, "jd_id": jd_id}
    match_res = client.post("/api/match", json=match_payload)
    assert match_res.status_code == 200
    analysis_id = match_res.json()["analysis_id"]
    print(f"[PASS] Test 4: POST /api/match (Analysis ID: {analysis_id[:8]}..., Overall Score: {match_res.json()['overall_match_percentage']}%)")

    # Test 5: GET Roadmap & Interview Endpoints
    roadmap_res = client.get(f"/api/roadmap/{analysis_id}")
    assert roadmap_res.status_code == 200
    interview_res = client.get(f"/api/interview/{analysis_id}")
    assert interview_res.status_code == 200
    print(f"[PASS] Test 5: GET /api/roadmap and GET /api/interview (Roadmap weeks: {len(roadmap_res.json()['roadmap'])})")

    print("\nALL PHASE 6 BACKEND API TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    run_phase6_tests()
