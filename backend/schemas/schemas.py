from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ResumeUploadResponse(BaseModel):
    resume_id: str
    filename: str
    num_pages: int
    extracted_skills: List[str]
    total_skills_count: int
    message: str


class JobAnalyzeRequest(BaseModel):
    title: Optional[str] = "Target Job Role"
    job_description_text: str = Field(..., min_length=20, description="Raw text of the target job description")


class JobAnalyzeResponse(BaseModel):
    jd_id: str
    title: str
    extracted_skills: List[str]
    total_skills_count: int


class MatchRequest(BaseModel):
    resume_id: str
    jd_id: str


class MatchResponse(BaseModel):
    analysis_id: str
    overall_match_percentage: float
    skill_match_percentage: float
    ats_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    created_at: str


class AnalysisDetailResponse(BaseModel):
    analysis_id: str
    overall_match_percentage: float
    skill_match_percentage: float
    tfidf_match_percentage: float
    semantic_embedding_percentage: float
    ats_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    partial_skills: List[str]
    full_report: Dict[str, Any]


class RoadmapResponse(BaseModel):
    analysis_id: str
    roadmap: List[Dict[str, Any]]


class InterviewResponse(BaseModel):
    analysis_id: str
    interview_questions: Dict[str, List[Dict[str, str]]]
