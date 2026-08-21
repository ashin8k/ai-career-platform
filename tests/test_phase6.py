import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_job_analyze_endpoint():
    payload = {
        "title": "Machine Learning Engineer",
        "job_description_text": "We are seeking a Machine Learning Engineer with 2+ years of experience in Python, PyTorch, FastAPI, Docker, and PostgreSQL."
    }
    response = client.post("/api/job/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "jd_id" in data
    assert "python" in data["extracted_skills"]
    assert "fastapi" in data["extracted_skills"]


def test_full_api_workflow():
    # 1. Generate sample PDF resume
    sample_pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_resume.pdf")
    if not os.path.exists(sample_pdf_path):
        pytest.skip("sample_resume.pdf does not exist.")

    # 2. Upload PDF resume
    with open(sample_pdf_path, "rb") as f:
        upload_res = client.post("/api/resume/upload", files={"file": ("resume.pdf", f, "application/pdf")})
    assert upload_res.status_code == 200
    resume_id = upload_res.json()["resume_id"]

    # 3. Submit Job Description
    jd_payload = {
        "title": "Backend AI Developer",
        "job_description_text": "Required skills: Python, Scikit-learn, FastAPI, Docker, PostgreSQL, AWS, and Machine Learning."
    }
    jd_res = client.post("/api/job/analyze", json=jd_payload)
    assert jd_res.status_code == 200
    jd_id = jd_res.json()["jd_id"]

    # 4. Trigger Resume Match
    match_payload = {"resume_id": resume_id, "jd_id": jd_id}
    match_res = client.post("/api/match", json=match_payload)
    assert match_res.status_code == 200
    analysis_id = match_res.json()["analysis_id"]

    # 5. Fetch Analysis Details
    detail_res = client.get(f"/api/analysis/{analysis_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["overall_match_percentage"] > 0

    # 6. Fetch Roadmap & Interview Questions
    roadmap_res = client.get(f"/api/roadmap/{analysis_id}")
    assert roadmap_res.status_code == 200
    
    interview_res = client.get(f"/api/interview/{analysis_id}")
    assert interview_res.status_code == 200
