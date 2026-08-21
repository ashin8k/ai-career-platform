import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from backend.database.db import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    resumes = relationship("Resume", back_populates="user")


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    filename = Column(String, nullable=False)
    raw_text = Column(Text, nullable=False)
    extracted_skills = Column(JSON, nullable=True)
    sections = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="resumes")
    analyses = relationship("MatchAnalysis", back_populates="resume")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)
    extracted_skills = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("MatchAnalysis", back_populates="job_description")


class MatchAnalysis(Base):
    __tablename__ = "match_analyses"

    id = Column(String, primary_key=True, default=generate_uuid)
    resume_id = Column(String, ForeignKey("resumes.id"), nullable=False)
    jd_id = Column(String, ForeignKey("job_descriptions.id"), nullable=False)

    overall_score = Column(Float, nullable=False)
    skill_score = Column(Float, nullable=False)
    tfidf_score = Column(Float, nullable=False)
    embedding_score = Column(Float, nullable=False)
    ats_score = Column(Float, nullable=False)

    matching_skills = Column(JSON, nullable=True)
    missing_skills = Column(JSON, nullable=True)
    partial_skills = Column(JSON, nullable=True)

    full_report = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("Resume", back_populates="analyses")
    job_description = relationship("JobDescription", back_populates="analyses")
