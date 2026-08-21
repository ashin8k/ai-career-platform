import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.preprocessing.text_cleaner import TextCleaner
from ml.skill_extraction.section_extractor import SectionExtractor
from ml.skill_extraction.skill_extractor import SkillExtractor
from ml.skill_extraction.taxonomy import normalize_skill_name, get_all_skills_flat


def run_tests():
    print("=" * 60)
    print("RUNNING PHASE 1 & PHASE 2 STANDALONE VERIFICATION SUITE")
    print("=" * 60)

    # Test 1: Text Cleaner Tech Preservation
    cleaner = TextCleaner()
    raw = "Hands-on experience in C++, C#, .NET, Node.js, scikit-learn, and CI/CD."
    cleaned = cleaner.clean_text(raw)
    assert "c++" in cleaned, f"Failed C++ preservation in: {cleaned}"
    assert "c#" in cleaned, f"Failed C# preservation in: {cleaned}"
    assert ".net" in cleaned, f"Failed .NET preservation in: {cleaned}"
    assert "node.js" in cleaned, f"Failed Node.js preservation in: {cleaned}"
    assert "scikit-learn" in cleaned, f"Failed scikit-learn preservation in: {cleaned}"
    assert "ci/cd" in cleaned, f"Failed CI/CD preservation in: {cleaned}"
    print("[PASS] Test 1: Domain-Aware Text Cleaner (Tech Preservation)")

    # Test 2: Text Cleaner Tokenization & Stats
    raw_tokens = "Built a Machine Learning model using Python and SQL."
    tokens = cleaner.tokenize(raw_tokens)
    assert "python" in tokens and "sql" in tokens and "and" not in tokens
    stats = cleaner.get_corpus_statistics(raw_tokens)
    assert stats["raw_word_count"] == 9
    print("[PASS] Test 2: Text Tokenization & Corpus Statistics")

    # Test 3: Section Extraction
    sample_resume = """
John Doe
Summary
Motivated Software Engineer.

EDUCATION
BS Computer Science - Stanford

SKILLS
Python, SQL, PyTorch, Docker, Kubernetes, AWS, Machine Learning

PROJECTS
Smart Resume Matcher: Built using FastAPI, Scikit-learn, Docker.

EXPERIENCE
Software Intern at Acme Corp.
"""
    sec_extractor = SectionExtractor()
    sections = sec_extractor.extract_sections(sample_resume)
    assert "education" in sections, "Missing education section"
    assert "skills" in sections, "Missing skills section"
    assert "projects" in sections, "Missing projects section"
    assert "experience" in sections, "Missing experience section"
    print("[PASS] Test 3: Resume Section Segmentation")

    # Test 4: Skill Extraction & Taxonomy Categorization
    skill_extractor = SkillExtractor(cleaner=cleaner)
    extraction = skill_extractor.extract_skills(sample_resume)
    extracted = extraction["extracted_skills"]
    categorized = extraction["categorized"]

    assert "python" in extracted, f"Missing python in {extracted}"
    assert "sql" in extracted, f"Missing sql in {extracted}"
    assert "pytorch" in extracted, f"Missing pytorch in {extracted}"
    assert "docker" in extracted, f"Missing docker in {extracted}"
    assert "machine learning" in extracted, f"Missing machine learning in {extracted}"
    
    assert "programming_languages" in categorized
    assert "frameworks_libraries" in categorized
    assert "cloud_devops" in categorized
    assert "ml_ai_concepts" in categorized

    print("[PASS] Test 4: Multi-Word N-gram Skill Extraction & Categorization")

    # Test 5: Alias Resolution
    assert normalize_skill_name("Postgres") == "postgresql"
    assert normalize_skill_name("sklearn") == "scikit-learn"
    assert normalize_skill_name("k8s") == "kubernetes"
    print("[PASS] Test 5: Skill Alias Resolution")

    print("\nALL STANDALONE TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
