"""
Orchestrator - Pydantic Models
Request/Response models untuk semua endpoints.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str
    wa_number: str = "unknown"


class SyncRequest(BaseModel):
    action: str
    content: Optional[Any] = None


class DocSearchRequest(BaseModel):
    query: str
    limit: int = 5


class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: Optional[str] = None
    file_url: str


class DocDeleteRequest(BaseModel):
    doc_id: str


class UsulanSyncRequest(BaseModel):
    action: str
    content: Optional[Any] = None


class UsulanSearchRequest(BaseModel):
    question: str
    wa_number: str = "unknown"


class WebTriggerRequest(BaseModel):
    link_id: str
    url: str
    callback_url: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


class WebSyncRequest(BaseModel):
    link_id: str
    edited_content: str


class WebDeleteRequest(BaseModel):
    link_id: str


class WebSearchRequest(BaseModel):
    query: str
    limit: int = 5
