"""FastAPI app for RAG Web Scraping service."""
import os
import sys
import uuid
import gc
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging

from services.rag_web import search as search_module
from services.rag_web import sync as sync_module
from services.rag_web.models import (
    SearchRequest, UnifiedSearchRequest, TriggerRequest, SyncRequest, 
    DeleteRequest, GetContentRequest
)

logger = setup_logging("rag_web")

_model: SentenceTransformer = None
_model_lock = asyncio.Lock()
_last_model_used: float = 0.0
qdrant: AsyncQdrantClient = None


async def get_model() -> SentenceTransformer:
    """Lazy load embedding model with thundering herd protection."""
    global _model, _last_model_used
    
    if _model is not None:
        _last_model_used = time.time()
        return _model
    
    async with _model_lock:
        if _model is not None:
            _last_model_used = time.time()
            return _model
        
        logger.info("Loading embedding model (lazy)...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)
        _last_model_used = time.time()
        logger.info(f"Model loaded: {config.EMBEDDING_MODEL_PATH}")
        
        search_module.set_instances(_model, qdrant)
        sync_module.set_instances(_model, qdrant)
        
        return _model


async def _idle_unload_loop():
    """Background task: unload model after IDLE_TIMEOUT seconds of inactivity."""
    global _model, _last_model_used
    while True:
        await asyncio.sleep(300)
        if _model is not None and _last_model_used > 0:
            idle_seconds = time.time() - _last_model_used
            if idle_seconds > config.MODEL_IDLE_TIMEOUT:
                async with _model_lock:
                    if _model is not None and (time.time() - _last_model_used) > config.MODEL_IDLE_TIMEOUT:
                        logger.info(f"Model idle for {idle_seconds:.0f}s > {config.MODEL_IDLE_TIMEOUT}s, unloading...")
                        del _model
                        _model = None
                        gc.collect()
                        logger.info("Model unloaded, RAM freed")


async def init_qdrant():
    """Initialize Qdrant connection."""
    global qdrant
    
    if config.QDRANT_API_KEY:
        qdrant = AsyncQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
            grpc_port=None,
            prefer_grpc=False,
            timeout=60
        )
    else:
        qdrant = AsyncQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            grpc_port=None,
            prefer_grpc=False,
            timeout=60
        )
    
    try:
        collections = await qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if config.COLLECTION_WEB not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_WEB}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_WEB,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            
            # Create indexes
            await qdrant.create_payload_index(
                collection_name=config.COLLECTION_WEB,
                field_name="link_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD
            )
            await qdrant.create_payload_index(
                collection_name=config.COLLECTION_WEB,
                field_name="is_deleted",
                field_schema=qdrant_models.PayloadSchemaType.BOOL
            )
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")
    
    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - model loaded lazily on first request."""
    await init_qdrant()
    
    asyncio.create_task(_idle_unload_loop())
    
    logger.info(f"RAG Web Service Started on port {config.WEB_SERVICE_PORT} (model: lazy load)")
    
    yield
    
    logger.info("RAG Web Service Shutting down...")

    try:
        from services.rag_web.js_renderer import js_renderer
        await js_renderer.close()
    except Exception as e:
        logger.warning(f"Error saat close js_renderer: {e}")


app = FastAPI(
    title="RAG Web Service",
    description="Service untuk RAG Web Scraping - Web Scraping Bank",
    version="3.0.0",
    lifespan=lifespan
)


# ============== ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        await qdrant.get_collections()
        qdrant_ok = True
    except:
        qdrant_ok = False
    
    status = "healthy" if qdrant_ok else "unhealthy"
    
    return {
        "status": status,
        "service": "rag_web",
        "model_loaded": _model is not None,
        "components": {
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """Internal search endpoint - dipanggil oleh orchestrator (direct mode)."""
    await get_model()
    logger.info(f"[SEARCH] Query: {request.query[:50]}...")
    
    result = await search_module.search_web_bank(
        question=request.query,
        limit=request.limit
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/search-unified")
async def internal_search_unified(request: UnifiedSearchRequest):
    """Internal search endpoint for unified/parallel mode."""
    await get_model()
    logger.info(f"[SEARCH-UNIFIED] Question: {request.question[:50]}...")
    
    result = await search_module.search_web_unified(
        question=request.question,
        original_question=request.original_question,
        wa_number=request.wa_number,
        top_k=request.top_k
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/trigger")
async def internal_trigger(request: TriggerRequest, background_tasks: BackgroundTasks):
    """
    Trigger web scraping in background.
    Semua opsi (CSS selector, JS renderer, FAQ mode, dll.) diteruskan ke sync_module.
    """
    await get_model()
    logger.info(
        f"[TRIGGER] link_id={request.link_id}, url={request.url}, "
        f"content_type={request.content_type}, css_selector={request.css_selector}, "
        f"use_js_renderer={request.use_js_renderer}, force_rescrape={request.force_rescrape}"
    )

    # Rate limit per-domain
    try:
        from services.rag_web.rate_limiter import rate_limiter
        await rate_limiter.wait_for_domain(request.url)
    except Exception as e:
        logger.warning(f"[TRIGGER] Rate limiter error (ignored): {e}")

    job_id = str(uuid.uuid4())

    background_tasks.add_task(
        sync_module.process_url,
        link_id=request.link_id,
        url=request.url,
        css_selector=request.css_selector,
        use_js_renderer=request.use_js_renderer,
        wait_selector=request.wait_selector,
        content_type=request.content_type,
        faq_question_selector=request.faq_question_selector,
        faq_answer_selector=request.faq_answer_selector,
        force_rescrape=request.force_rescrape,
        callback_url=request.callback_url,
        metadata=request.metadata,
    )

    return {
        "status": "processing",
        "message": "Scraping job started",
        "link_id": request.link_id,
        "job_id": job_id,
        "options": {
            "content_type": request.content_type,
            "css_selector": request.css_selector,
            "use_js_renderer": request.use_js_renderer,
            "force_rescrape": request.force_rescrape,
        }
    }


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """Internal sync endpoint - untuk edited content."""
    await get_model()
    logger.info(f"[SYNC] link_id={request.link_id}")
    
    result = await sync_module.sync_edited_content(
        link_id=request.link_id,
        edited_content=request.edited_content
    )
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Content not found")
    
    return result


@app.delete("/internal/delete")
async def internal_delete(request: DeleteRequest):
    """Internal delete endpoint - soft delete."""
    logger.info(f"[DELETE] link_id={request.link_id}")
    
    count = await sync_module.soft_delete_by_link_id(request.link_id)
    
    if count == 0:
        raise HTTPException(status_code=404, detail="Content not found")
    
    return {
        "status": "success",
        "link_id": request.link_id,
        "deleted_chunks": count
    }


@app.post("/internal/content")
async def internal_get_content(request: GetContentRequest):
    """Internal get content endpoint."""
    logger.info(f"[GET-CONTENT] link_id={request.link_id}")
    
    result = await sync_module.get_content(request.link_id)
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Content not found")
    
    return result


def start_service():
    """Start the service."""
    uvicorn.run(
        "services.rag_web.main:app",
        host="0.0.0.0",
        port=config.WEB_SERVICE_PORT,
        reload=False,
        log_config=None
    )


if __name__ == "__main__":
    start_service()
