from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.models.models import MatchAnalysis
from backend.schemas.schemas import (
    ResumeUploadResponse,
    JobAnalyzeRequest,
    JobAnalyzeResponse,
    MatchRequest,
    MatchResponse,
    AnalysisDetailResponse,
    RoadmapResponse,
    InterviewResponse
)
from backend.services.analysis_service import AnalysisService

router = APIRouter()
service = AnalysisService()


@router.post("/resume/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF resume files are supported."
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty."
        )

    resume_obj = service.process_pdf_resume(db, file.filename, pdf_bytes)
    return ResumeUploadResponse(
        resume_id=resume_obj.id,
        filename=resume_obj.filename,
        num_pages=1,
        extracted_skills=resume_obj.extracted_skills or [],
        total_skills_count=len(resume_obj.extracted_skills or []),
        message="Resume PDF processed and extracted successfully."
    )


@router.post("/job/analyze", response_model=JobAnalyzeResponse)
def analyze_job(
    request: JobAnalyzeRequest,
    db: Session = Depends(get_db)
):
    jd_obj = service.process_job_description(db, request.title or "Target Job Role", request.job_description_text)
    return JobAnalyzeResponse(
        jd_id=jd_obj.id,
        title=jd_obj.title,
        extracted_skills=jd_obj.extracted_skills or [],
        total_skills_count=len(jd_obj.extracted_skills or [])
    )


@router.post("/match", response_model=MatchResponse)
def match_resume_and_jd(
    request: MatchRequest,
    db: Session = Depends(get_db)
):
    try:
        analysis_obj = service.execute_match_analysis(db, request.resume_id, request.jd_id)
        return MatchResponse(
            analysis_id=analysis_obj.id,
            overall_match_percentage=analysis_obj.overall_score,
            skill_match_percentage=analysis_obj.skill_score,
            ats_score=analysis_obj.ats_score,
            matching_skills=analysis_obj.matching_skills or [],
            missing_skills=analysis_obj.missing_skills or [],
            created_at=analysis_obj.created_at.isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/analysis/{id}", response_model=AnalysisDetailResponse)
def get_analysis_details(id: str, db: Session = Depends(get_db)):
    analysis = db.query(MatchAnalysis).filter(MatchAnalysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Match Analysis with id {id} not found.")

    return AnalysisDetailResponse(
        analysis_id=analysis.id,
        overall_match_percentage=analysis.overall_score,
        skill_match_percentage=analysis.skill_score,
        tfidf_match_percentage=analysis.tfidf_score,
        semantic_embedding_percentage=analysis.embedding_score,
        ats_score=analysis.ats_score,
        matching_skills=analysis.matching_skills or [],
        missing_skills=analysis.missing_skills or [],
        partial_skills=analysis.partial_skills or [],
        full_report=analysis.full_report or {}
    )


@router.get("/skills/{id}")
def get_skills_breakdown(id: str, db: Session = Depends(get_db)):
    analysis = db.query(MatchAnalysis).filter(MatchAnalysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    return {
        "analysis_id": analysis.id,
        "matching_skills": analysis.matching_skills or [],
        "missing_skills": analysis.missing_skills or [],
        "partial_skills": analysis.partial_skills or []
    }


@router.get("/roadmap/{id}", response_model=RoadmapResponse)
def get_roadmap(id: str, db: Session = Depends(get_db)):
    analysis = db.query(MatchAnalysis).filter(MatchAnalysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    full_report = analysis.full_report or {}
    roadmap = full_report.get("learning_roadmap", [])

    return RoadmapResponse(
        analysis_id=analysis.id,
        roadmap=roadmap
    )


@router.get("/interview/{id}", response_model=InterviewResponse)
def get_interview_questions(id: str, db: Session = Depends(get_db)):
    analysis = db.query(MatchAnalysis).filter(MatchAnalysis.id == id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    full_report = analysis.full_report or {}
    interview_qs = full_report.get("interview_preparation", {})

    return InterviewResponse(
        analysis_id=analysis.id,
        interview_questions=interview_qs
    )
