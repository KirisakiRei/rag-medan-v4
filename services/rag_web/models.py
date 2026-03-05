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
    link_id: str = Field(..., description="Unique ID dari link")
    url: str = Field(..., description="URL yang akan di-scrape")

    # Content extraction options
    content_type: str = Field("general", description="Tipe konten: general | article | faq")
    css_selector: Optional[str] = Field(None, description="CSS selector untuk target konten spesifik (contoh: div.berita-isi)")

    # JavaScript rendering options
    use_js_renderer: Optional[bool] = Field(None, description="True=paksa Playwright, False=paksa httpx, None=auto-detect")
    wait_selector: Optional[str] = Field(None, description="CSS selector yang ditunggu muncul sebelum ambil HTML (hanya untuk JS renderer)")

    # FAQ extraction options
    faq_question_selector: Optional[str] = Field(None, description="CSS selector untuk elemen pertanyaan FAQ")
    faq_answer_selector: Optional[str] = Field(None, description="CSS selector untuk elemen jawaban FAQ")

    # Processing control
    force_rescrape: bool = Field(False, description="Paksa re-scrape meski link_id sudah ada")
    callback_url: Optional[str] = Field(None, description="Custom callback URL (override global config)")
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
