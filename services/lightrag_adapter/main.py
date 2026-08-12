"""
RAG Medan v4 - LightRAG Adapter — FastAPI Application.

Port: 5015

Service adapter yang menjembatani Orchestrator dan LightRAG Server.
Menyediakan internal API untuk:
- Unified search via LightRAG
- Knowledge sync (text/document/web) ke LightRAG
- Source deletion dari LightRAG
- Health check

Semua endpoint dilindungi oleh InternalAuthMiddleware (X-API-Key).
Tidak ada endpoint yang terekspos ke publik.
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.logging_config import setup_logging
from shared.security import InternalAuthMiddleware

from services.lightrag_adapter.config import adapter_config
from services.lightrag_adapter.client import lightrag_client
from services.lightrag_adapter import search as search_module
from services.lightrag_adapter import sync as sync_module
from services.lightrag_adapter.health import check_health
from services.lightrag_adapter.models import (
    SearchRequest,
    SyncTextRequest,
    SyncDocumentRequest,
    SyncWebRequest,
)

# Setup logging — konsisten dengan service lainnya
logger = setup_logging("lightrag_adapter")


# ============== LIFESPAN ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LightRAG client on startup, cleanup on shutdown."""
    await lightrag_client.start()

    logger.info(f"LightRAG Adapter Started on port {adapter_config.PORT}")
    logger.info(f"  LightRAG Server : {adapter_config.BASE_URL}")
    logger.info(f"  Workspace       : {adapter_config.WORKSPACE}")
    logger.info(f"  Query Mode      : {adapter_config.QUERY_MODE}")
    logger.info(f"  Fallback Legacy : {adapter_config.FALLBACK_TO_LEGACY}")
    logger.info(f"  Index Text      : {adapter_config.INDEX_TEXT}")
    logger.info(f"  Index Document  : {adapter_config.INDEX_DOCUMENT}")
    logger.info(f"  Index Web       : {adapter_config.INDEX_WEB}")

    yield

    await lightrag_client.stop()
    logger.info("LightRAG Adapter Shutting down...")


# ============== FASTAPI APP ==============

app = FastAPI(
    title="LightRAG Adapter",
    description=(
        "Adapter service antara Orchestrator RAG Medan dan LightRAG Server. "
        "Menyediakan unified search dan knowledge sync API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Proteksi semua endpoint internal dengan X-API-Key
app.add_middleware(InternalAuthMiddleware)


# ============== HEALTH ==============

@app.get("/health")
async def health_check():
    """Health check endpoint — mengecek adapter dan LightRAG Server."""
    result = await check_health()
    return JSONResponse(status_code=200, content=result)


# ============== SEARCH ==============

@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """
    Unified search via LightRAG.

    Memanggil LightRAG Server untuk melakukan retrieval dari
    unified knowledge base (text + document + web).

    Response: canonical format dengan contexts dan references.
    """
    logger.info(
        f"[SEARCH] query='{request.query[:80]}' "
        f"mode={request.mode} top_k={request.top_k}"
    )

    result = await search_module.search(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
    )

    return JSONResponse(status_code=200, content=result)


# ============== SYNC — TEXT ==============

@app.post("/internal/sync/text")
async def sync_text_endpoint(request: SyncTextRequest):
    """
    Sync text/FAQ knowledge ke LightRAG.

    Dipanggil oleh rag_text service setelah normalisasi konten.
    Idempotent — caller bertanggung jawab atas content_hash check.
    """
    logger.info(
        f"[SYNC-TEXT] source_id={request.source_id} "
        f"title='{request.title[:50]}' active={request.is_active}"
    )

    result = await sync_module.sync_text(
        source_id=request.source_id,
        knowledge_base_id=request.knowledge_base_id,
        title=request.title,
        content=request.content,
        content_hash=request.content_hash,
        is_active=request.is_active,
        category=request.category,
        question=request.question,
        answer=request.answer,
    )

    return JSONResponse(status_code=200, content=result)


# ============== SYNC — DOCUMENT ==============

@app.post("/internal/sync/document")
async def sync_document_endpoint(request: SyncDocumentRequest):
    """
    Sync document knowledge ke LightRAG.

    Dipanggil oleh rag_document service setelah OCR/extraction selesai.
    Konten yang dikirim adalah normalized full document text.
    """
    logger.info(
        f"[SYNC-DOC] source_id={request.source_id} "
        f"title='{request.title[:50]}' active={request.is_active}"
    )

    result = await sync_module.sync_document(
        source_id=request.source_id,
        knowledge_base_id=request.knowledge_base_id,
        title=request.title,
        normalized_content=request.normalized_content,
        file_name=request.file_name,
        content_hash=request.content_hash,
        is_active=request.is_active,
        organization_id=request.organization_id,
    )

    return JSONResponse(status_code=200, content=result)


# ============== SYNC — WEB ==============

@app.post("/internal/sync/web")
async def sync_web_endpoint(request: SyncWebRequest):
    """
    Sync web page knowledge ke LightRAG.

    Dipanggil oleh rag_web service setelah scraping dan cleaning selesai.
    """
    logger.info(
        f"[SYNC-WEB] source_id={request.source_id} "
        f"url='{request.url[:80]}' active={request.is_active}"
    )

    result = await sync_module.sync_web(
        source_id=request.source_id,
        knowledge_base_id=request.knowledge_base_id,
        url=request.url,
        title=request.title,
        clean_content=request.clean_content,
        content_hash=request.content_hash,
        is_active=request.is_active,
    )

    return JSONResponse(status_code=200, content=result)


# ============== DELETE ==============

@app.delete("/internal/source/{source_type}/{source_id}")
async def delete_source(source_type: str, source_id: str):
    """
    Delete a source dari LightRAG index.

    Path params:
        source_type: "text" | "document" | "web"
        source_id: Application primary key
    """
    logger.info(f"[DELETE] type={source_type} id={source_id}")

    result = await sync_module.delete_source(
        source_type=source_type,
        source_id=source_id,
    )

    return JSONResponse(status_code=200, content=result)


# ============== REINDEX ==============

@app.post("/internal/reindex/{source_type}/{source_id}")
async def reindex_source(source_type: str, source_id: str):
    """
    Trigger reindex untuk specific source.

    Reindex = delete + re-sync dari SQL source of truth.
    Actual re-sync dilakukan oleh caller (source processor)
    setelah endpoint ini berhasil menghapus data lama.
    """
    logger.info(f"[REINDEX] type={source_type} id={source_id}")

    # Step 1: Delete existing data dari LightRAG
    delete_result = await sync_module.delete_source(
        source_type=source_type,
        source_id=source_id,
    )

    if delete_result.get("status") == "error":
        return JSONResponse(status_code=200, content={
            "status": "error",
            "message": f"Delete failed: {delete_result.get('message', '')}",
            "source_type": source_type,
            "source_id": source_id,
        })

    # Step 2: Return status — caller harus trigger sync ulang
    return JSONResponse(status_code=200, content={
        "status": "deleted_ready_for_resync",
        "source_type": source_type,
        "source_id": source_id,
        "message": (
            f"Source {source_type}:{source_id} deleted from LightRAG. "
            f"Caller should now trigger sync to re-index."
        ),
    })


# ============== ENTRY POINT ==============

def start_service():
    """Start the LightRAG Adapter service."""
    uvicorn.run(
        "services.lightrag_adapter.main:app",
        host="0.0.0.0",
        port=adapter_config.PORT,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    start_service()
