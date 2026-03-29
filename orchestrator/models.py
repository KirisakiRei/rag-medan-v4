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
    organization_id: Optional[str] = None
    filename: Optional[str] = None
    file_url: str
    is_active: bool = True


class DocDeleteRequest(BaseModel):
    doc_id: str


class UsulanSyncRequest(BaseModel):
    action: str
    content: Optional[Any] = None


class UsulanSearchRequest(BaseModel):
    question: str
    wa_number: str = "unknown"


class WebTriggerRequest(BaseModel):
    web_bank_id: str
    name: str
    opd_id: str
    url: str
    css_selector: Optional[str] = None
    scrape_interval: Optional[int] = None
    is_active: bool = True
    metadata: Optional[dict] = Field(default_factory=dict)


class WebUpdateRequest(BaseModel):
    web_bank_id: str
    name: str
    opd_id: str
    url: str
    css_selector: Optional[str] = None
    scrape_interval: Optional[int] = None
    is_active: bool = True
    metadata: Optional[dict] = Field(default_factory=dict)


class WebDeleteRequest(BaseModel):
    web_bank_id: str


class WebSearchRequest(BaseModel):
    query: str
    limit: int = 5
