"""
RAG Text Service - Knowledge Bank
Modular service untuk pencarian Q&A di knowledge_bank
"""
from services.rag_text.main import app, start_service

__all__ = ["app", "start_service"]
