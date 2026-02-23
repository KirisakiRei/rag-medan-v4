from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ============== REQUEST MODELS ==============

class SearchRequest(BaseModel):
    """Request for search endpoint."""
    question: str
    wa_number: str = "unknown"
    original_question: Optional[str] = None
    skip_prefilter: bool = False


class UnifiedSearchRequest(BaseModel):
    """Request for unified search (parallel mode)."""
    question: str
    original_question: str
    wa_number: str = "unknown"
    top_k: int = 3


class SyncRequest(BaseModel):
    """Request for sync operations."""
    action: str
    content: Optional[Any] = None


class SyncItemContent(BaseModel):
    """Content item untuk sync."""
    answer_id: int
    question: str
    answer_doc: Optional[str] = ""
    category_id: Optional[str] = "1"


# ============== RESPONSE MODELS ==============

class SearchResultItem(BaseModel):
    """Single search result item."""
    question: str
    question_rag_name: str
    answer_id: int
    answer_doc: str
    category_id: str
    dense_score: float
    overlap_score: float
    final_score: float
    note: str


class SearchMetadata(BaseModel):
    """Search metadata."""
    wa_number: str
    original_question: str
    final_question: str
    category: str
    ai_reason: str
    ai_reformulated: str
    final_score_top: float


class SearchResponseData(BaseModel):
    """Data dalam search response."""
    similar_questions: List[SearchResultItem]
    metadata: SearchMetadata


class SearchResponse(BaseModel):
    """Response untuk search endpoint."""
    status: str
    message: str
    source: str = "text"
    data: Optional[SearchResponseData] = None
    timing: Optional[Dict[str, float]] = None


class SyncResponse(BaseModel):
    """Response untuk sync endpoint."""
    status: str
    message: str
    action: str
    count: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
