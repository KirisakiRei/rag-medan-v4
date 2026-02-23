from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ============== REQUEST MODELS ==============

class SearchRequest(BaseModel):
    """Request for document search."""
    query: str
    limit: int = 5


class UnifiedSearchRequest(BaseModel):
    """Request for unified search (parallel mode)."""
    question: str
    original_question: str
    wa_number: str = "unknown"
    top_k: int = 3


class SyncRequest(BaseModel):
    """Request for document sync."""
    doc_id: str
    opd_name: Optional[str] = None
    file_url: str


class DeleteRequest(BaseModel):
    """Request for document delete."""
    doc_id: str


# ============== RESPONSE MODELS ==============
