"""
RAG Document Service - Document Bank
Modular service untuk pencarian dokumen PDF/OCR di document_bank
"""
from services.rag_document.main import app, start_service

__all__ = ["app", "start_service"]
