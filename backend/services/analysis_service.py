from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.models.models import Resume, JobDescription, MatchAnalysis
from ml.preprocessing.pdf_extractor import PDFExtractor
from ml.skill_extraction.skill_extractor import SkillExtractor
from ml.skill_extraction.section_extractor import SectionExtractor
from ml.skill_extraction.gap_analyzer import SkillGapAnalyzer
from ml.skill_extraction.ats_analyzer import ATSAnalyzer
from ml.matching.hybrid_matcher import HybridMatcher
from ml.reasoning.ai_explainer import AIExplainer


class AnalysisService:
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.skill_extractor = SkillExtractor()
        self.section_extractor = SectionExtractor()
        self.gap_analyzer = SkillGapAnalyzer()
        self.ats_analyzer = ATSAnalyzer()
        self.hybrid_matcher = HybridMatcher(lazy_load_embedding=True)
        self.explainer = AIExplainer()

    def process_pdf_resume(self, db: Session, filename: str, pdf_bytes: bytes) -> Resume:
        pdf_res = self.pdf_extractor.extract_text_from_bytes(pdf_bytes)
        raw_text = pdf_res["text"]

        skills_res = self.skill_extractor.extract_skills(raw_text)
        sections = self.section_extractor.extract_sections(raw_text)

        resume_obj = Resume(
            filename=filename,
            raw_text=raw_text,
            extracted_skills=skills_res["extracted_skills"],
            sections=sections
        )
        db.add(resume_obj)
        db.commit()
        db.refresh(resume_obj)
        return resume_obj

    def process_job_description(self, db: Session, title: str, jd_text: str) -> JobDescription:
        skills_res = self.skill_extractor.extract_skills(jd_text)

        jd_obj = JobDescription(
            title=title,
            raw_text=jd_text,
            extracted_skills=skills_res["extracted_skills"]
        )
        db.add(jd_obj)
        db.commit()
        db.refresh(jd_obj)
        return jd_obj

    def execute_match_analysis(self, db: Session, resume_id: str, jd_id: str) -> MatchAnalysis:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()

        if not resume or not jd:
            raise ValueError("Resume or Job Description not found in database.")

        match_res = self.hybrid_matcher.match(resume.raw_text, jd.raw_text)
        gap_res = self.gap_analyzer.analyze_gap(resume.raw_text, jd.raw_text)
        ats_res = self.ats_analyzer.analyze_resume(resume.raw_text, jd.raw_text)

        full_report = self.explainer.generate_full_analysis(
            resume.raw_text,
            jd.raw_text,
            match_res,
            ats_res
        )

        analysis_obj = MatchAnalysis(
            resume_id=resume.id,
            jd_id=jd.id,
            overall_score=match_res["overall_match_percentage"],
            skill_score=match_res["skill_match_percentage"],
            tfidf_score=match_res["tfidf_match_percentage"],
            embedding_score=match_res["semantic_embedding_percentage"],
            ats_score=ats_res["ats_score"],
            matching_skills=gap_res["matching_skills"],
            missing_skills=gap_res["missing_skills"],
            partial_skills=gap_res["partial_skills"],
            full_report=full_report
        )
        db.add(analysis_obj)
        db.commit()
        db.refresh(analysis_obj)
        return analysis_obj
