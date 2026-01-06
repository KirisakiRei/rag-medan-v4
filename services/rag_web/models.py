"""
RAG Web Service - Pydantic Models
Request/Response models untuk rag_web service
"""
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ============== REQUEST MODELS ==============

class SearchRequest(BaseModel):
    """Request dari orchestrator untuk search."""
    query: str
    limit: int = 5


class UnifiedSearchRequest(BaseModel):
    """Request dari orchestrator untuk unified search (parallel mode)."""
    question: str  # Clean question dari orchestrator
    original_question: str  # Pertanyaan asli user
    wa_number: str = "unknown"
    top_k: int = 3  # Number of candidates to return


class TriggerRequest(BaseModel):
    """Request untuk trigger scraping."""
    link_id: str = Field(..., description="Unique ID dari link")
    url: str = Field(..., description="URL yang akan di-scrape")
    callback_url: Optional[str] = Field(None, description="Custom callback URL")
    metadata: Optional[dict] = Field(default_factory=dict)


class SyncRequest(BaseModel):
    """Request untuk sync edited content."""
    link_id: str = Field(..., description="ID link")
    edited_content: str = Field(..., description="Konten hasil edit")


class DeleteRequest(BaseModel):
    """Request untuk delete content."""
    link_id: str = Field(..., description="ID link yang akan dihapus")


class GetContentRequest(BaseModel):
    """Request untuk get content."""
    link_id: str = Field(..., description="ID link")


# ============== RESPONSE MODELS ==============

class SearchResultItem(BaseModel):
    """Single search result item."""
    id: str
    link_id: str
    url: str
    title: str
    content: str
    chunk_index: int
    score: float
    source: str = "web"


class SearchResponse(BaseModel):
    """Response untuk search endpoint."""
    status: str
    message: str
    source: str = "web"
    results: List[SearchResultItem] = []
    timing: Optional[Dict[str, float]] = None


class TriggerResponse(BaseModel):
    """Response untuk trigger endpoint."""
    status: str
    message: str
    link_id: str
    job_id: Optional[str] = None


class SyncResponse(BaseModel):
    """Response untuk sync endpoint."""
    status: str
    link_id: str
    chunks_count: Optional[int] = None


class DeleteResponse(BaseModel):
    """Response untuk delete endpoint."""
    status: str
    link_id: str
    deleted_chunks: Optional[int] = None


class ContentResponse(BaseModel):
    """Response untuk get content endpoint."""
    status: str
    link_id: str
    url: str
    title: str
    clean_content: str
    chunks_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
