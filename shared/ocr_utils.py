"""
RAG Medan v3 - Shared OCR Utils
Utilities untuk OCR dan text extraction dari berbagai format file
"""
import os
import re
import tempfile
import logging
import hashlib
from typing import Dict, Optional

logger = logging.getLogger("ocr_utils")

# Lazy-loaded PaddleOCR engine (singleton)
_ocr_engine: Optional[object] = None


def get_ocr_engine():
    """Lazy load PaddleOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("[OCR] Initializing PaddleOCR engine...")
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(lang="id", use_angle_cls=True)
        logger.info("[OCR] PaddleOCR engine ready")
    return _ocr_engine


def _ocr_image_bytes(img_bytes: bytes) -> str:
    """OCR dari bytes gambar."""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(img_bytes)
            temp_file.flush()
            temp_file_path = temp_file.name

        ocr_engine = get_ocr_engine()
        ocr_result = ocr_engine.ocr(temp_file_path)
        if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
            return "\n".join([line[1][0] for line in ocr_result[0] if line and len(line) > 1])
        return ""
    except Exception as e:
        logger.warning(f"[OCR] Gagal OCR image: {e}")
        return ""
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


def _clean_page_text(page_text: str) -> str:
    """Bersihkan header/footer, spasi dobel, nomor halaman."""
    if not page_text:
        return ""
    cleaned_text = re.sub(r"^\s*\d{1,4}\s*$", "", page_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def clean_ocr_text(text: str) -> str:
    """Membersihkan hasil OCR dari formatting berlebihan."""
    if not text:
        return ""
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    
    cleaned_lines = []
    prev_line = None
    for line in lines:
        if line and line != prev_line:
            cleaned_lines.append(line)
            prev_line = line
        elif not line and prev_line:
            cleaned_lines.append('')
            prev_line = None
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
    
    return text.strip()


def format_for_display(text: str) -> str:
    """Format text untuk display dengan paragraph breaks yang jelas."""
    if not text:
        return ""
    
    paragraphs = re.split(r'\n\n+', text)
    cleaned_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            para = para.replace('\n', ' ')
            para = re.sub(r' {2,}', ' ', para)
            cleaned_paragraphs.append(para)
    
    return '\n\n'.join(cleaned_paragraphs)


def calculate_file_hash(file_path: str) -> str:
    """Hitung SHA256 hash dari file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def calculate_content_hash(text: str) -> str:
    """Hitung SHA256 hash dari content text."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _extract_pdf_pages(pdf_path: str, dpi: int = 180) -> Dict[int, str]:
    """Ekstrak teks PDF per halaman dengan hybrid (vector text + OCR)."""
    import fitz
    
    extracted_pages: Dict[int, str] = {}

    logger.info(f"[PDF] Opening PDF: {pdf_path}")
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"[PDF] Failed to open PDF: {e}")
        raise
    
    total_page_count = len(pdf_document)
    logger.info(f"[PDF] Total pages: {total_page_count}")

    for page_number in range(1, total_page_count + 1):
        logger.info(f"[PDF] Processing page {page_number}/{total_page_count}...")
        current_page = pdf_document[page_number - 1]

        # Coba pakai teks bawaan PDF
        try:
            extracted_text = current_page.get_text("text") or ""
        except Exception as e:
            logger.warning(f"[PDF] Gagal get_text di halaman {page_number}: {e}")
            extracted_text = ""

        extracted_text = extracted_text.strip()
        if extracted_text:
            extracted_pages[page_number] = _clean_page_text(extracted_text)
            continue

        # Jika tidak ada teks -> OCR dari bitmap
        try:
            page_pixmap = current_page.get_pixmap(dpi=dpi)
            image_bytes = page_pixmap.tobytes("png")
            ocr_extracted_text = _ocr_image_bytes(image_bytes)
            extracted_pages[page_number] = _clean_page_text(ocr_extracted_text)
        except Exception as e:
            logger.warning(f"[PDF] Gagal render/OCR halaman {page_number}: {e}")
            extracted_pages[page_number] = ""

    return dict(sorted(extracted_pages.items(), key=lambda x: x[0]))


def extract_text_from_file(file_path: str, lang: str = "id", return_pages: bool = False):
    """
    Ekstraksi teks dari berbagai format file.
    Mendukung: PDF, DOCX, XLSX, TXT, dan gambar (JPG, PNG).
    
    Args:
        file_path: Path ke file
        lang: Bahasa untuk OCR (default: "id")
        return_pages: Jika True return dict, False return string gabungan
    """
    from docx import Document
    import openpyxl
    
    file_extension = os.path.splitext(file_path)[1].lower()
    extracted_pages = {}

    if file_extension == ".pdf":
        extracted_pages = _extract_pdf_pages(file_path, dpi=180)

    elif file_extension in [".jpg", ".jpeg", ".png"]:
        try:
            ocr_engine = get_ocr_engine()
            ocr_result = ocr_engine.ocr(file_path)
        except Exception as e:
            logger.warning(f"[OCR] Gagal OCR image file {file_path}: {e}")
            ocr_result = None

        extracted_text = ""
        if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
            extracted_text = "\n".join([line[1][0] for line in ocr_result[0] if line and len(line) > 1])
        extracted_pages[1] = _clean_page_text(extracted_text)

    elif file_extension == ".docx":
        try:
            docx_document = Document(file_path)
        except Exception as e:
            logger.warning(f"[DOCX] Gagal membuka file DOCX {file_path}: {e}")
            extracted_pages[1] = ""
            if return_pages:
                return extracted_pages
            return ""

        text_parts = []
        for para in docx_document.paragraphs:
            if not para.text.strip():
                continue
            # Heading-aware: tambah double newline sebelum heading (batas chunk alami)
            if para.style.name.startswith("Heading"):
                text_parts.append("\n\n" + para.text.strip())
            else:
                text_parts.append(para.text.strip())
        extracted_text = "\n".join(text_parts)
        extracted_pages[1] = _clean_page_text(extracted_text)

    elif file_extension == ".xlsx":
        try:
            # read_only=True: streaming mode — RAM konstan berapapun ukuran file
            workbook = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
            all_sheets_text = []
            for sheet in workbook.worksheets:
                sheet_text = f"=== Sheet: {sheet.title} ===\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    if row_text.strip():
                        sheet_text += row_text + "\n"
                all_sheets_text.append(sheet_text)
            extracted_text = "\n\n".join(all_sheets_text)
            extracted_pages[1] = _clean_page_text(extracted_text)
        except Exception as e:
            logger.warning(f"[XLSX] Gagal ekstraksi Excel file {file_path}: {e}")
            extracted_pages[1] = ""

    elif file_extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as txt_file:
                extracted_text = txt_file.read()
            extracted_pages[1] = _clean_page_text(extracted_text)
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as txt_file:
                    extracted_text = txt_file.read()
                extracted_pages[1] = _clean_page_text(extracted_text)
            except Exception as e:
                logger.warning(f"[TXT] Gagal ekstraksi TXT file {file_path}: {e}")
                extracted_pages[1] = ""

    else:
        raise ValueError(f"Format file {file_extension} belum didukung untuk OCR.")

    if return_pages:
        return dict(sorted(extracted_pages.items(), key=lambda x: x[0]))

    return "\n\n".join(extracted_pages.values()).strip()
