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
    organization_id: Optional[str] = None  # UUID dari sistem wa manajemen
    filename: Optional[str] = None          # Nama file user-friendly (opsional)
    file_url: str
    is_active: bool = True


class DeleteRequest(BaseModel):
    """Request for document delete."""
    doc_id: str


# ============== RESPONSE MODELS ==============
