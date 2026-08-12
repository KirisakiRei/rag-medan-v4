"""
RAG Medan v4 - LightRAG Adapter — Pydantic Models.

Request/Response models untuk semua endpoint adapter.
Mengikuti konvensi models existing (orchestrator/models.py).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============== SEARCH ==============

class SearchRequest(BaseModel):
    """Request untuk POST /internal/search."""
    query: str
    knowledge_base_id: str = "medan-main"
    mode: str = "mix"                       # naive|local|global|hybrid|mix
    top_k: int = 10
    include_references: bool = True


class ContextItem(BaseModel):
    """Satu unit context dari hasil retrieval."""
    content: str
    source_type: str                        # "text" | "document" | "web"
    source_id: str
    title: str = ""
    source_uri: str = ""
    reference_id: str = ""
    score: Optional[float] = None


class SearchTiming(BaseModel):
    """Breakdown timing untuk observability."""
    retrieval_sec: float = 0.0
    rerank_sec: float = 0.0
    total_sec: float = 0.0


class SearchResponse(BaseModel):
    """Canonical response dari LightRAG search."""
    status: str                             # "success" | "no_results" | "error"
    engine: str = "lightrag"
    query: str = ""
    contexts: List[ContextItem] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    timing: SearchTiming = Field(default_factory=SearchTiming)


# ============== SYNC — TEXT ==============

class SyncTextRequest(BaseModel):
    """Request untuk POST /internal/sync/text."""
    source_id: str
    knowledge_base_id: str = "medan-main"
    title: str
    content: str
    content_hash: str
    is_active: bool = True
    # Extra metadata untuk citation
    category: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


# ============== SYNC — DOCUMENT ==============

class SyncDocumentRequest(BaseModel):
    """Request untuk POST /internal/sync/document."""
    source_id: str
    knowledge_base_id: str = "medan-main"
    title: str
    normalized_content: str
    file_name: Optional[str] = None
    file_hash: Optional[str] = None
    content_hash: str
    is_active: bool = True
    organization_id: Optional[str] = None


# ============== SYNC — WEB ==============

class SyncWebRequest(BaseModel):
    """Request untuk POST /internal/sync/web."""
    source_id: str
    knowledge_base_id: str = "medan-main"
    url: str
    title: str
    clean_content: str
    content_hash: str
    is_active: bool = True


# ============== SYNC RESPONSE ==============

class SyncResponse(BaseModel):
    """Canonical response dari sync operations."""
    status: str                             # "success" | "skipped" | "error"
    source_id: str
    source_type: str
    lightrag_document_id: Optional[str] = None
    message: str = ""


# ============== REINDEX ==============

class ReindexRequest(BaseModel):
    """Request untuk POST /internal/reindex."""
    source_type: Optional[str] = None       # "text" | "document" | "web" | None (all)
    batch_size: int = 50
    dry_run: bool = False


class ReindexResponse(BaseModel):
    """Response dari reindex operation."""
    status: str
    total_processed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    errors: List[Dict[str, str]] = Field(default_factory=list)
