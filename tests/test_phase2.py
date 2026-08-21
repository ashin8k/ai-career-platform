import pytest
from ml.skill_extraction.section_extractor import SectionExtractor
from ml.skill_extraction.skill_extractor import SkillExtractor
from ml.skill_extraction.taxonomy import normalize_skill_name


def test_section_extractor():
    sample_text = """
Alex Mercer
Summary
Passionate AI Engineer.

EDUCATION
BS in Computer Science from University of Tech

SKILLS
Python, C++, Docker, PostgreSQL, Machine Learning

PROJECTS
Built an AI Resume Matcher using FastAPI and Docker.

EXPERIENCE
ML Intern at TechCorp.
"""

    extractor = SectionExtractor()
    sections = extractor.extract_sections(sample_text)

    assert "education" in sections
    assert "skills" in sections
    assert "projects" in sections
    assert "experience" in sections
    assert "Python" in sections["skills"] or "python" in sections["skills"].lower()


def test_skill_extractor_single_and_multi_word():
    extractor = SkillExtractor()
    sample_text = "Proficient in Python, C++, Machine Learning, Natural Language Processing, Docker, and PostgreSQL."
    
    result = extractor.extract_skills(sample_text)
    skills = result["extracted_skills"]

    assert "python" in skills
    assert "c++" in skills
    assert "machine learning" in skills
    assert "natural language processing" in skills
    assert "docker" in skills
    assert "postgresql" in skills
    assert result["total_unique_skills"] >= 6


def test_skill_categorization():
    extractor = SkillExtractor()
    sample_text = "Skills include Python, PyTorch, PostgreSQL, AWS, and Agile."
    
    result = extractor.extract_skills(sample_text)
    categorized = result["categorized"]

    assert "programming_languages" in categorized
    assert "python" in categorized["programming_languages"]
    assert "frameworks_libraries" in categorized
    assert "pytorch" in categorized["frameworks_libraries"]
    assert "databases" in categorized
    assert "postgresql" in categorized["databases"]
    assert "cloud_devops" in categorized
    assert "aws" in categorized["cloud_devops"]


def test_skill_alias_normalization():
    assert normalize_skill_name("Postgres") == "postgresql"
    assert normalize_skill_name("sklearn") == "scikit-learn"
    assert normalize_skill_name("k8s") == "kubernetes"
