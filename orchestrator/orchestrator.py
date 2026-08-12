"""
RAG Medan v3 - Orchestrator
Unified controller untuk semua RAG services dengan parallel search dan score-based selection.
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from shared.logging_config import setup_logging
from shared.security import InternalAuthMiddleware

# Import modules
from orchestrator.models import (
    SearchRequest, SyncRequest, DocSearchRequest, DocSyncRequest, 
    DocDeleteRequest, UsulanSyncRequest, UsulanSearchRequest,
    WebTriggerRequest, WebUpdateRequest, WebDeleteRequest, WebSearchRequest
)
from orchestrator.service_client import call_service, set_client, create_optimized_client
from orchestrator.search_handler import unified_search

logger = setup_logging("orchestrator")

http_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    
    logger.info("Starting RAG Medan v3 - Orchestrator...")
    http_client = create_optimized_client()  # Use optimized client
    
    # Set client for service_client module
    set_client(http_client)
    
    logger.info("Orchestrator Started - PARALLEL SEARCH MODE")
    logger.info(f"  Text: {config.TEXT_SERVICE_URL}")
    logger.info(f"  Document: {config.DOCUMENT_SERVICE_URL}")
    logger.info(f"  Web: {config.WEB_SERVICE_URL}")
    logger.info(f"  Usulan: {config.USULAN_SERVICE_URL}")
    
    yield
    
    await http_client.aclose()
    logger.info("Orchestrator Shutting down...")


app = FastAPI(
    title="RAG Medan v3 - Orchestrator",
    description="Unified RAG Service Orchestrator with Parallel Search",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Proteksi seluruh /api/* dengan header X-API-Key (payload tidak berubah).
app.add_middleware(InternalAuthMiddleware)

# ============== HEALTH ENDPOINTS ==============

@app.get("/")
async def root():
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
    """
    Health check gabungan - cek semua services secara paralel.

    Threshold:
    - Core services (text, document, web, usulan) harus semua healthy.
    - Embedding service hanya wajib healthy saat USE_SHARED_EMBEDDING=True
      (mode local tidak bergantung padanya).
    """
    health_timeout = 5.0

    async def _check(name: str, url: str) -> dict:
        try:
            detail = await call_service(url, "/health", "GET", timeout=health_timeout)
            up = detail.get("status") == "healthy"
            return {
                "up": up,
                "status": detail.get("status", "unknown"),
                "detail": detail,
            }
        except Exception:
            return {"up": False, "status": "unreachable", "detail": None}

    targets = [
        ("text_service", config.TEXT_SERVICE_URL),
        ("document_service", config.DOCUMENT_SERVICE_URL),
        ("web_service", config.WEB_SERVICE_URL),
        ("usulan_service", config.USULAN_SERVICE_URL),
        ("embedding_service", config.SHARED_EMBEDDING_URL),
    ]

    results = await asyncio.gather(*(_check(name, url) for name, url in targets))
    components = dict(zip([t[0] for t in targets], results))

    core_up = all(components[name]["up"] for name in [
        "text_service", "document_service", "web_service", "usulan_service"
    ])

    embedding_required = config.USE_SHARED_EMBEDDING
    embedding_up = components["embedding_service"]["up"]

    if core_up and (not embedding_required or embedding_up):
        status = "healthy"
    elif core_up:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "service": "orchestrator",
        "mode": "parallel_search",
        "shared_embedding": embedding_required,
        "components": {
            name: {"up": info["up"], "status": info["status"]}
            for name, info in components.items()
        },
        "details": components,
    }

# ============== UNIFIED SEARCH ==============

@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    """Unified search — parallel fan-out to all services, score-based selection."""
    user_question = (request.question or "").strip()
    wa_number = request.wa_number

    if not user_question:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Field 'question' wajib diisi"}
        )

    logger.info(f"[REQUEST] POST /api/search | wa={wa_number} | q='{user_question[:80]}'")

    result = await unified_search(
        user_question=user_question,
        wa_number=wa_number,
        use_ai_pre_filter=True
    )

    status = result.get("status", "?")
    total_sec = result.get("timing", {}).get("total_sec", 0)
    source = result.get("source", "-")
    logger.info(f"[RESPONSE] status={status} | source={source} | total={total_sec}s | wa={wa_number}")

    return JSONResponse(status_code=200, content=result)

# ============== TEXT RAG SYNC ==============

@app.post("/api/sync")
async def sync_data(request: SyncRequest):
    """Sync data ke knowledge_bank."""
    logger.info(f"[SYNC] Action: {request.action}")
    
    result = await call_service(
        config.TEXT_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"action": request.action, "content": request.content}
    )
    
    return JSONResponse(status_code=200, content=result)

# ============== DOCUMENT RAG ENDPOINTS ==============

@app.post("/api/doc-search")
async def doc_search(request: DocSearchRequest):
    """Search di RAG Document (document_bank) direct mode."""
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
    """Sync document (trigger OCR)."""
    organization_id = request.organization_id or request.opd_name
    logger.info(
        f"[DOC-SYNC] doc_id={request.doc_id}, organization_id={organization_id}, "
        f"filename={request.filename}, is_active={request.is_active}"
    )
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/sync",
        "POST",
        {
            "doc_id": request.doc_id,
            "organization_id": organization_id,
            "filename": request.filename,
            "file_url": request.file_url,
            "is_active": request.is_active,
        }
    )
    
    return result


@app.get("/api/doc-sync/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status task OCR."""
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
    """Soft delete document."""
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

# ============== USULAN RAG ENDPOINTS ==============

@app.post("/api/sync-usulan")
async def sync_usulan(request: UsulanSyncRequest):
    """Sync usulan ke usulan_bank."""
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
    """Search di RAG Usulan (usulan_bank)."""
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

# ============== WEB RAG ENDPOINTS ==============

@app.post("/api/web-trigger")
async def trigger_web_scraping(request: WebTriggerRequest):
    """Trigger web scraping."""
    logger.info(f"[WEB-TRIGGER] web_bank_id={request.web_bank_id}, url={request.url}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/trigger",
        "POST",
        {
            "web_bank_id": request.web_bank_id,
            "name": request.name,
            "opd_id": request.opd_id,
            "url": request.url,
            "css_selector": request.css_selector,
            "scrape_interval": request.scrape_interval,
            "is_active": request.is_active,
            "metadata": request.metadata
        }
    )
    
    return result


@app.put("/api/web-update")
async def update_web_content(request: WebUpdateRequest):
    """Update web bank metadata and optionally rescrape if source settings changed."""
    logger.info(f"[WEB-UPDATE] web_bank_id={request.web_bank_id}, url={request.url}")

    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/update",
        "PUT",
        {
            "web_bank_id": request.web_bank_id,
            "name": request.name,
            "opd_id": request.opd_id,
            "url": request.url,
            "css_selector": request.css_selector,
            "scrape_interval": request.scrape_interval,
            "is_active": request.is_active,
            "metadata": request.metadata,
        }
    )

    return result


@app.delete("/api/web-delete")
async def delete_web_content(request: WebDeleteRequest):
    """Soft delete web content."""
    logger.info(f"[WEB-DELETE] web_bank_id={request.web_bank_id}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/delete",
        "DELETE",
        {"web_bank_id": request.web_bank_id}
    )
    
    return result


@app.post("/api/web-search")
async def search_web(request: WebSearchRequest):
    """Search di RAG Web (web_scraping_bank)."""
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
