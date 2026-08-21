import os
import pytest
from ml.preprocessing.pdf_extractor import PDFExtractor
from ml.preprocessing.text_cleaner import TextCleaner


def test_text_cleaner_tech_preservation():
    cleaner = TextCleaner()
    raw_text = "Experienced in C++, C#, .NET, Node.js, React.js, scikit-learn, and CI/CD pipelines."
    cleaned = cleaner.clean_text(raw_text)
    
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert ".net" in cleaned
    assert "node.js" in cleaned
    assert "scikit-learn" in cleaned
    assert "ci/cd" in cleaned


def test_text_cleaner_tokenization():
    cleaner = TextCleaner()
    raw_text = "Developer with experience in Python and SQL programming languages."
    tokens = cleaner.tokenize(raw_text, remove_stopwords=True)
    
    assert "developer" in tokens
    assert "python" in tokens
    assert "sql" in tokens
    assert "and" not in tokens
    assert "with" not in tokens


def test_text_cleaner_corpus_statistics():
    cleaner = TextCleaner()
    raw_text = "Built a Machine Learning model using Python."
    stats = cleaner.get_corpus_statistics(raw_text)
    
    assert stats["raw_word_count"] == 7
    assert stats["cleaned_word_count"] > 0
    assert stats["vocabulary_size"] > 0


def test_pdf_extractor_sample_file():
    sample_pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_resume.pdf")
    if not os.path.exists(sample_pdf_path):
        pytest.skip("sample_resume.pdf does not exist yet.")
    
    extractor = PDFExtractor()
    result = extractor.extract_text_from_file(sample_pdf_path)
    
    assert result["status"] == "success"
    assert result["num_pages"] > 0
    assert "Computer Science" in result["text"] or "Computer Science".lower() in result["text"].lower()
