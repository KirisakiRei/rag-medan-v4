"""
RAG Medan v3 - Orchestrator
Controller utama yang mengatur semua service RAG.

Tugas Orchestrator:
1. Menerima request dari user (search, sync, dll)
2. Meneruskan request ke service yang sesuai
3. Return response ke user (format sama dengan v2)

TIDAK ADA FALLBACK - setiap service independen.
User -> Orchestrator -> Service -> Orchestrator -> User

ENDPOINT SAMA DENGAN V2:
- POST /api/search         -> RAG Text (knowledge_bank)
- POST /api/sync           -> Sync knowledge_bank
- POST /api/doc-search     -> RAG Document (document_bank)
- POST /api/doc-sync       -> Sync document_bank
- GET  /api/doc-sync/status/{task_id}
- DELETE /api/doc-delete   -> Delete document
- POST /api/sync-usulan    -> Sync usulan_bank
- POST /api/search-usulan  -> RAG Usulan (usulan_bank)

NEW IN V3 (endpoint tambahan):
- POST /api/web-trigger    -> Trigger scraping
- POST /api/web-sync       -> Sync edited web content
- DELETE /api/web-delete   -> Delete web content
- POST /api/web-search     -> RAG Web (web_scraping_bank)
"""
import os
import sys
import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from shared.logging_config import setup_logging

# Setup logging
logger = setup_logging("orchestrator")

# HTTP client untuk komunikasi ke services
http_client: httpx.AsyncClient = None


# ============== REQUEST/RESPONSE MODELS (V2 COMPATIBLE) ==============

class SearchRequest(BaseModel):
    """Request untuk /api/search - SAMA DENGAN V2"""
    question: str
    wa_number: str = "unknown"


class SyncRequest(BaseModel):
    """Request untuk /api/sync - SAMA DENGAN V2"""
    action: str
    content: Optional[Any] = None


class DocSearchRequest(BaseModel):
    """Request untuk /api/doc-search - SAMA DENGAN V2"""
    query: str
    limit: int = 5


class DocSyncRequest(BaseModel):
    """Request untuk /api/doc-sync - SAMA DENGAN V2"""
    doc_id: str
    opd_name: Optional[str] = None
    file_url: str


class DocDeleteRequest(BaseModel):
    """Request untuk /api/doc-delete - SAMA DENGAN V2"""
    doc_id: str


class UsulanSyncRequest(BaseModel):
    """Request untuk /api/sync-usulan - SAMA DENGAN V2"""
    action: str
    content: Optional[Any] = None


class UsulanSearchRequest(BaseModel):
    """Request untuk /api/search-usulan - SAMA DENGAN V2"""
    question: str
    wa_number: str = "unknown"


# NEW V3 Models
class WebTriggerRequest(BaseModel):
    """Request untuk /api/web-trigger"""
    link_id: str
    url: str
    callback_url: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


class WebSyncRequest(BaseModel):
    """Request untuk /api/web-sync"""
    link_id: str
    edited_content: str


class WebDeleteRequest(BaseModel):
    """Request untuk /api/web-delete"""
    link_id: str


class WebSearchRequest(BaseModel):
    """Request untuk /api/web-search"""
    query: str
    limit: int = 5


# ============== SERVICE COMMUNICATION ==============

async def call_service(
    service_url: str, 
    endpoint: str, 
    method: str = "POST", 
    data: dict = None
) -> dict:
    """
    Call internal service endpoint.
    
    Args:
        service_url: Base URL service
        endpoint: Endpoint path
        method: HTTP method
        data: Request data
        
    Returns:
        Response dict
    """
    url = f"{service_url}{endpoint}"
    
    try:
        if method == "POST":
            response = await http_client.post(url, json=data, timeout=120.0)
        elif method == "GET":
            response = await http_client.get(url, timeout=60.0)
        elif method == "DELETE":
            response = await http_client.request("DELETE", url, json=data, timeout=60.0)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response.json()
        
    except httpx.TimeoutException:
        logger.error(f"[SERVICE] Timeout calling {url}")
        return {"status": "error", "error": "Service timeout"}
    except Exception as e:
        logger.error(f"[SERVICE] Error calling {url}: {e}")
        return {"status": "error", "error": str(e)}


# ============== INITIALIZATION ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global http_client
    
    logger.info("Starting RAG Medan v3 - Orchestrator...")
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient()
    
    logger.info("Orchestrator Started")
    logger.info(f"  - Text Service: {config.TEXT_SERVICE_URL}")
    logger.info(f"  - Document Service: {config.DOCUMENT_SERVICE_URL}")
    logger.info(f"  - Web Service: {config.WEB_SERVICE_URL}")
    logger.info(f"  - Usulan Service: {config.USULAN_SERVICE_URL}")
    
    yield
    
    await http_client.aclose()
    logger.info("Orchestrator Shutting down...")


app = FastAPI(
    title="RAG Medan v3 - Orchestrator",
    description="Unified RAG Service Orchestrator",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== HEALTH ENDPOINTS ==============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Medan v3 - Orchestrator is running!",
        "version": "3.0.0",
        "services": {
            "text": config.TEXT_SERVICE_URL,
            "document": config.DOCUMENT_SERVICE_URL,
            "web": config.WEB_SERVICE_URL,
            "usulan": config.USULAN_SERVICE_URL
        }
    }


@app.get("/health")
async def health_check():
    """Health check - cek semua services."""
    services_status = {}
    
    # Check text service
    try:
        text_health = await call_service(config.TEXT_SERVICE_URL, "/health", "GET")
        services_status["text_service"] = text_health.get("status") == "healthy"
    except:
        services_status["text_service"] = False
    
    # Check document service
    try:
        doc_health = await call_service(config.DOCUMENT_SERVICE_URL, "/health", "GET")
        services_status["document_service"] = doc_health.get("status") == "healthy"
    except:
        services_status["document_service"] = False
    
    # Check web service
    try:
        web_health = await call_service(config.WEB_SERVICE_URL, "/health", "GET")
        services_status["web_service"] = web_health.get("status") == "healthy"
    except:
        services_status["web_service"] = False
    
    # Check usulan service
    try:
        usulan_health = await call_service(config.USULAN_SERVICE_URL, "/health", "GET")
        services_status["usulan_service"] = usulan_health.get("status") == "healthy"
    except:
        services_status["usulan_service"] = False
    
    overall_status = all(services_status.values())
    
    return {
        "status": "healthy" if overall_status else "degraded",
        "service": "orchestrator",
        "components": services_status
    }


# ============== TEXT RAG ENDPOINTS (V2 COMPATIBLE) ==============

@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search di RAG Text (knowledge_bank).
    TIDAK ADA FALLBACK - hanya search di text service.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SEARCH] Question: {request.question}, wa_number: {request.wa_number}")
    
    result = await call_service(
        config.TEXT_SERVICE_URL,
        "/internal/search",
        "POST",
        {"question": request.question, "wa_number": request.wa_number}
    )
    
    if result.get("status") == "error":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": result.get("error", "Internal error")}
        )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/api/sync")
async def sync_data(request: SyncRequest):
    """
    Sync data ke knowledge_bank.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SYNC] Action: {request.action}")
    
    result = await call_service(
        config.TEXT_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"action": request.action, "content": request.content}
    )
    
    return JSONResponse(status_code=200, content=result)


# ============== DOCUMENT RAG ENDPOINTS (V2 COMPATIBLE) ==============

@app.post("/api/doc-search")
async def doc_search(request: DocSearchRequest):
    """
    Search di RAG Document (document_bank).
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[DOC-SEARCH] Query: {request.query}, limit: {request.limit}")
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/search",
        "POST",
        {"query": request.query, "limit": request.limit}
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Internal error"))
    
    return result


@app.post("/api/doc-sync")
async def doc_sync(request: DocSyncRequest):
    """
    Sync document (trigger OCR).
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[DOC-SYNC] doc_id={request.doc_id}, opd={request.opd_name}")
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/sync",
        "POST",
        {
            "doc_id": request.doc_id,
            "opd_name": request.opd_name,
            "file_url": request.file_url
        }
    )
    
    return result


@app.get("/api/doc-sync/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status task OCR.
    SAMA DENGAN V2.
    """
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        f"/internal/sync/status/{task_id}",
        "GET"
    )
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    
    return result


@app.get("/api/doc-sync/tasks")
async def list_tasks():
    """List semua tasks OCR."""
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/sync/tasks",
        "GET"
    )
    return result


@app.delete("/api/doc-delete")
async def doc_delete(request: DocDeleteRequest):
    """
    Soft delete document.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[DOC-DELETE] doc_id={request.doc_id}")
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/delete",
        "DELETE",
        {"doc_id": request.doc_id}
    )
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Document not found")
    
    return result


# ============== USULAN RAG ENDPOINTS (V2 COMPATIBLE) ==============

@app.post("/api/sync-usulan")
async def sync_usulan(request: UsulanSyncRequest):
    """
    Sync data ke usulan_bank.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SYNC-USULAN] Action: {request.action}")
    
    result = await call_service(
        config.USULAN_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"action": request.action, "content": request.content}
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/api/search-usulan")
async def search_usulan(request: UsulanSearchRequest):
    """
    Search di RAG Usulan (usulan_bank).
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SEARCH-USULAN] Question: {request.question}, wa_number: {request.wa_number}")
    
    result = await call_service(
        config.USULAN_SERVICE_URL,
        "/internal/search",
        "POST",
        {"question": request.question, "wa_number": request.wa_number}
    )
    
    if result.get("status") == "error":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": result.get("error", "Internal error")}
        )
    
    return JSONResponse(status_code=200, content=result)


# ============== WEB RAG ENDPOINTS (NEW IN V3) ==============

@app.post("/api/web-trigger")
async def trigger_web_scraping(request: WebTriggerRequest):
    """
    Trigger web scraping (NEW in v3).
    """
    logger.info(f"[WEB-TRIGGER] link_id={request.link_id}, url={request.url}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/trigger",
        "POST",
        {
            "link_id": request.link_id,
            "url": request.url,
            "callback_url": request.callback_url,
            "metadata": request.metadata
        }
    )
    
    return result


@app.post("/api/web-sync")
async def sync_web_content(request: WebSyncRequest):
    """
    Sync edited web content (NEW in v3).
    """
    logger.info(f"[WEB-SYNC] link_id={request.link_id}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"link_id": request.link_id, "edited_content": request.edited_content}
    )
    
    return result


@app.delete("/api/web-delete")
async def delete_web_content(request: WebDeleteRequest):
    """
    Delete web content (NEW in v3).
    """
    logger.info(f"[WEB-DELETE] link_id={request.link_id}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/delete",
        "DELETE",
        {"link_id": request.link_id}
    )
    
    return result


@app.post("/api/web-search")
async def search_web(request: WebSearchRequest):
    """
    Search di RAG Web (web_scraping_bank) (NEW in v3).
    """
    logger.info(f"[WEB-SEARCH] query={request.query}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/search",
        "POST",
        {"query": request.query, "limit": request.limit}
    )
    
    return result


if __name__ == "__main__":
    uvicorn.run(
        "orchestrator.orchestrator:app",
        host=config.API_HOST,
        port=config.ORCHESTRATOR_PORT,
        reload=False,
        log_config=None
    )
