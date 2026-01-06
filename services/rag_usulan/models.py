"""
RAG Usulan Service - Pydantic Models
Request/Response models untuk rag_usulan service
"""
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ============== REQUEST MODELS ==============

class SearchRequest(BaseModel):
    """Request dari orchestrator untuk search."""
    question: str
    wa_number: str = "unknown"


class SyncRequest(BaseModel):
    """Request untuk sync usulan_bank."""
    action: str  # bulk_sync, add, update, delete
    content: Optional[Any] = None


class SyncItemContent(BaseModel):
    """Content item untuk sync."""
    usulan_id: int
    isi_usulan: str
    opd_pelaksana: Optional[str] = ""
    kecamatan: Optional[str] = ""
    kelurahan: Optional[str] = ""


# ============== RESPONSE MODELS ==============

class SearchResultItem(BaseModel):
    """Single search result item."""
    usulan_id: int
    isi_usulan: str
    opd_pelaksana: str
    kecamatan: str
    kelurahan: str
    score: float
    note: str


class SearchMetadata(BaseModel):
    """Metadata dari search."""
    wa_number: str
    original_question: str
    final_question: str
    ai_reason: str
    ai_reformulated: str
    final_score_top: float


class SearchResponseData(BaseModel):
    """Data dalam search response."""
    similar_usulan: List[SearchResultItem]
    metadata: SearchMetadata


class SearchResponse(BaseModel):
    """Response untuk search endpoint."""
    status: str
    message: str
    source: str = "usulan"
    data: Optional[SearchResponseData] = None
    timing: Optional[Dict[str, float]] = None


class SyncResponse(BaseModel):
    """Response untuk sync endpoint."""
    status: str
    message: str
    action: str
    count: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
