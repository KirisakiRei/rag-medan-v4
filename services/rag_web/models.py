"""Pydantic models for rag_web service."""
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ============== REQUEST MODELS ==============

class SearchRequest(BaseModel):
    """Request for search endpoint."""
    query: str
    limit: int = 5


class UnifiedSearchRequest(BaseModel):
    """Request for unified search (parallel mode)."""
    question: str
    original_question: str
    wa_number: str = "unknown"
    top_k: int = 3


class TriggerRequest(BaseModel):
    """Request untuk trigger scraping."""
    web_bank_id: str = Field(..., description="Unique ID dari web bank")
    name: str = Field(..., description="Nama website")
    opd_id: str = Field(..., description="ID OPD pemilik website")
    url: str = Field(..., description="URL yang akan di-scrape")
    css_selector: Optional[str] = Field(None, description="CSS selector untuk target konten spesifik (contoh: div.berita-isi)")
    scrape_interval: Optional[int] = Field(None, description="Interval scrape dalam jam")
    is_active: bool = Field(True, description="Apakah web bank aktif dan searchable")
    metadata: Optional[dict] = Field(default_factory=dict)


class UpdateRequest(BaseModel):
    """Request untuk update metadata web bank."""
    web_bank_id: str = Field(..., description="Unique ID dari web bank")
    name: str = Field(..., description="Nama website")
    opd_id: str = Field(..., description="ID OPD pemilik website")
    url: str = Field(..., description="URL website")
    css_selector: Optional[str] = Field(None, description="CSS selector target konten")
    scrape_interval: Optional[int] = Field(None, description="Interval scrape dalam jam")
    is_active: bool = Field(True, description="Apakah web bank aktif dan searchable")
    metadata: Optional[dict] = Field(default_factory=dict)


class SyncRequest(BaseModel):
    """Request untuk sync edited content."""
    web_bank_id: str = Field(..., description="ID web bank")
    edited_content: str = Field(..., description="Konten hasil edit")


class DeleteRequest(BaseModel):
    """Request untuk delete content."""
    web_bank_id: str = Field(..., description="ID web bank yang akan dihapus")


class GetContentRequest(BaseModel):
    """Request untuk get content."""
    web_bank_id: str = Field(..., description="ID web bank")


# ============== RESPONSE MODELS ==============

class SearchResultItem(BaseModel):
    """Single search result item."""
    id: str
    web_bank_id: str
    name: str
    opd_id: str
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
    web_bank_id: str
    job_id: Optional[str] = None


class SyncResponse(BaseModel):
    """Response untuk sync endpoint."""
    status: str
    web_bank_id: str
    chunks_count: Optional[int] = None


class UpdateResponse(BaseModel):
    """Response untuk update endpoint."""
    status: str
    web_bank_id: str
    message: str
    job_id: Optional[str] = None


class DeleteResponse(BaseModel):
    """Response untuk delete endpoint."""
    status: str
    web_bank_id: str
    deleted_chunks: Optional[int] = None


class ContentResponse(BaseModel):
    """Response untuk get content endpoint."""
    status: str
    web_bank_id: str
    url: str
    title: str
    clean_content: str
    chunks_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
