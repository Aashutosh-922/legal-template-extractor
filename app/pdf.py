from dataclasses import dataclass
from typing import List

import fitz


class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""


@dataclass
class PDFTextExtractionResult:
    text: str
    pages: List[str]
    ocr_used: bool


def _ocr_page(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise PDFExtractionError("OCR fallback requires pytesseract and Pillow.") from exc

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise PDFExtractionError(
            "OCR fallback requires a local Tesseract binary. Install `tesseract-ocr` and retry."
        ) from exc

    matrix = fitz.Matrix(2, 2)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, enable_ocr_fallback: bool) -> PDFTextExtractionResult:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_text: List[str] = []
            ocr_used = False
            for page_number, page in enumerate(document, start=1):
                extracted = page.get_text("text").strip()
                if not extracted and enable_ocr_fallback:
                    extracted = _ocr_page(page)
                    if extracted:
                        ocr_used = True
                if extracted:
                    page_text.append(f"[Page {page_number}]\n{extracted}")
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFExtractionError(f"Could not read PDF content: {exc}") from exc

    combined_text = "\n\n".join(page_text)
    if not combined_text:
        raise PDFExtractionError(
            "No extractable text found in PDF. OCR fallback did not return usable text."
        )

    return PDFTextExtractionResult(text=combined_text, pages=page_text, ocr_used=ocr_used)
