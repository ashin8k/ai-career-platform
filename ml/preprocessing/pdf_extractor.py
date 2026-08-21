import io
from typing import Dict, Any, Optional


class PDFExtractor:
    """
    Robust PDF Resume Text Extractor.
    Supports file paths, binary stream inputs, multi-page text extraction,
    and metadata extraction using pdfplumber with PyMuPDF fallback.
    """

    def __init__(self, fallback_to_pymupdf: bool = True):
        self.fallback_to_pymupdf = fallback_to_pymupdf

    def extract_text_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file on disk.
        """
        try:
            return self._extract_with_pdfplumber(file_path)
        except Exception as e:
            if self.fallback_to_pymupdf:
                return self._extract_with_pymupdf(file_path)
            raise RuntimeError(f"Failed to extract text from PDF file {file_path}: {str(e)}")

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text and metadata from raw PDF byte stream (e.g. uploaded via FastAPI stream).
        """
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text(layout=True) or page.extract_text() or ""
                    if text.strip():
                        pages_text.append(text)
                
                full_text = "\n\n".join(pages_text)
                return {
                    "text": full_text,
                    "num_pages": len(pdf.pages),
                    "extractor_used": "pdfplumber",
                    "status": "success" if full_text.strip() else "empty_or_scanned"
                }
        except Exception as e:
            if self.fallback_to_pymupdf:
                return self._extract_bytes_with_pymupdf(pdf_bytes)
            raise RuntimeError(f"Failed to extract text from PDF bytes: {str(e)}")

    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text(layout=True) or page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
            
            full_text = "\n\n".join(pages_text)
            return {
                "text": full_text,
                "num_pages": len(pdf.pages),
                "extractor_used": "pdfplumber",
                "status": "success" if full_text.strip() else "empty_or_scanned"
            }

    def _extract_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages_text = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        doc.close()
        full_text = "\n\n".join(pages_text)
        return {
            "text": full_text,
            "num_pages": len(doc),
            "extractor_used": "PyMuPDF",
            "status": "success" if full_text.strip() else "empty_or_scanned"
        }

    def _extract_bytes_with_pymupdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        doc.close()
        full_text = "\n\n".join(pages_text)
        return {
            "text": full_text,
            "num_pages": len(doc),
            "extractor_used": "PyMuPDF",
            "status": "success" if full_text.strip() else "empty_or_scanned"
        }
