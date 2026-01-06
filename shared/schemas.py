"""
RAG Medan v3 - Shared Schemas/Models
Pydantic models untuk request/response
"""
from datetime import datetime
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


# ============== ENUMS ==============

class SearchSource(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    WEB = "web_scraping"
    NONE = "none"
    COMBINED = "combined"


class SyncAction(str, Enum):
    BULK_SYNC = "bulk_sync"
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


# ============== BASE MODELS ==============

class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: str = ""
    data: Any = None


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = "error"
    error: Dict[str, Any]


# ============== SEARCH MODELS ==============

class SearchRequest(BaseModel):
    """Request untuk unified search."""
    question: str = Field(..., description="Pertanyaan user")
    wa_number: str = Field(default="unknown", description="Nomor WhatsApp user")
    limit: int = Field(default=5, description="Jumlah hasil maksimum per source")
    include_sources: List[str] = Field(
        default=["text", "document", "web"],
        description="Sumber yang akan dicari"
    )


class SimilarQuestion(BaseModel):
    """Hasil pencarian yang mirip."""
    question: str = Field(default="-")
    question_rag_name: str = Field(default="-")
    answer_id: Optional[List[str]] = Field(default=None)
    answer_doc: str = Field(default="")
    category_id: Optional[str] = Field(default=None)
    dense_score: float = Field(default=0.0)
    overlap_score: float = Field(default=0.0)
    final_score: float = Field(default=0.0)
    note: str = Field(default="")
    source: str = Field(default="text")


class DocumentInfo(BaseModel):
    """Info dokumen sumber."""
    filename: Optional[str] = None
    page_number: Optional[int] = None
    opd: Optional[str] = None


class WebInfo(BaseModel):
    """Info web sumber."""
    url: str = ""
    title: str = ""
    link_id: str = ""


class SearchMetadata(BaseModel):
    """Metadata hasil pencarian."""
    wa_number: str
    original_question: str
    final_question: str
    category: str = Field(default="Global")
    ai_reason: str = Field(default="")
    ai_reformulated: str = Field(default="")
    final_score_top: float = Field(default=0.0)
    document_info: Optional[DocumentInfo] = None
    web_info: Optional[WebInfo] = None


class SearchTiming(BaseModel):
    """Timing metrics."""
    ai_domain_sec: float = Field(default=0.0)
    ai_relevance_sec: float = Field(default=0.0)
    embedding_sec: float = Field(default=0.0)
    qdrant_sec: float = Field(default=0.0)
    rerank_sec: float = Field(default=0.0)
    total_sec: float = Field(default=0.0)


class SearchResponseData(BaseModel):
    """Data response pencarian."""
    similar_questions: List[SimilarQuestion]
    metadata: SearchMetadata


class SearchResponse(BaseModel):
    """Response untuk search endpoint."""
    status: str = Field(default="success")
    message: str = Field(default="")
    source: str = Field(default="text")
    data: SearchResponseData
    timing: SearchTiming


# ============== SYNC MODELS (TEXT) ==============

class TextSyncItem(BaseModel):
    """Item untuk sync knowledge_bank."""
    question_rag_id: int
    question_id: int
    answer_id: Any
    category_id: Optional[str] = None
    question: str
    question_rag_name: str


class TextSyncRequest(BaseModel):
    """Request untuk sync text RAG."""
    action: SyncAction
    content: Optional[Any] = None


# ============== SYNC MODELS (DOCUMENT) ==============

class DocSyncRequest(BaseModel):
    """Request untuk sync document."""
    doc_id: str
    opd_name: Optional[str] = None
    file_url: str


class DocDeleteRequest(BaseModel):
    """Request untuk delete document."""
    doc_id: str


# ============== SYNC MODELS (WEB) ==============

class WebTriggerRequest(BaseModel):
    """Request untuk trigger web scraping."""
    link_id: str = Field(..., description="Unique ID dari link")
    url: str = Field(..., description="URL yang akan di-scrape")
    callback_url: Optional[str] = Field(None, description="Custom callback URL")
    metadata: Optional[dict] = Field(default_factory=dict)


class WebSyncRequest(BaseModel):
    """Request untuk sync edited web content."""
    link_id: str = Field(..., description="ID link")
    edited_content: str = Field(..., description="Konten hasil edit")


class WebDeleteRequest(BaseModel):
    """Request untuk delete web content."""
    link_id: str = Field(..., description="ID link yang akan dihapus")


# ============== SYNC MODELS (USULAN) ==============

class UsulanSyncItem(BaseModel):
    """Item untuk sync usulan_bank."""
    request_rag_id: int
    request_id: int
    organization_id: int
    request_name: str
    request_rag_name: str


class UsulanSyncRequest(BaseModel):
    """Request untuk sync usulan."""
    action: SyncAction
    content: Optional[Any] = None


class UsulanSearchRequest(BaseModel):
    """Request untuk search usulan."""
    question: str
    wa_number: str = "unknown"


# ============== INTERNAL SERVICE MODELS ==============

class ServiceSearchRequest(BaseModel):
    """Request internal dari orchestrator ke service."""
    question: str
    normalized_question: str
    category_id: Optional[str] = None
    limit: int = 5


class ServiceSearchResponse(BaseModel):
    """Response internal dari service ke orchestrator."""
    status: str
    results: List[Dict[str, Any]]
    timing: Dict[str, float]
