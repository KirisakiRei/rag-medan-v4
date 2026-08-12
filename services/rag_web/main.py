"""FastAPI app for RAG Web Scraping service."""
import os
import sys
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.security import InternalAuthMiddleware
from shared.bootstrap import (
    LazyModel,
    backfill_is_active,
    create_qdrant_client,
    ensure_payload_index,
)

from services.rag_web import search as search_module
from services.rag_web import sync as sync_module
from services.rag_web.models import (
    SearchRequest, UnifiedSearchRequest, TriggerRequest, UpdateRequest, SyncRequest,
    DeleteRequest, GetContentRequest
)

logger = setup_logging("rag_web")

qdrant: AsyncQdrantClient = None


def _wire_model(model):
    """Wire loaded local model ke search module."""
    search_module.set_instances(model, qdrant)


model_holder = LazyModel(
    config.EMBEDDING_MODEL_PATH,
    on_load=_wire_model,
    name="embedding",
)


async def get_model():
    """Return local model (None saat shared embedding aktif)."""
    return await model_holder.get()


async def init_qdrant():
    """Initialize Qdrant connection."""
    global qdrant
    qdrant = create_qdrant_client()

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

        if config.COLLECTION_WEB_STATE not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_WEB_STATE}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_WEB_STATE,
                vectors_config=VectorParams(
                    size=1,
                    distance=Distance.COSINE
                )
            )

        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "web_bank_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "link_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "opd_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "is_deleted",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "is_active",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "chunk_level",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB,
            "parent_chunk_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB_STATE,
            "web_bank_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB_STATE,
            "is_active",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_WEB_STATE,
            "last_scrape_status",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await backfill_is_active(qdrant, config.COLLECTION_WEB)
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")

    # Wire Qdrant-dependent modules immediately so non-embedding endpoints
    # such as delete/content/update-state can work before lazy model loading.
    search_module.set_instances(model_holder.model, qdrant)
    sync_module.set_instances(qdrant)

    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - model loaded lazily on first request."""
    await init_qdrant()

    model_holder.start_idle_unload()

    logger.info(
        f"RAG Web Service Started on port {config.WEB_SERVICE_PORT} "
        f"(model: shared={config.USE_SHARED_EMBEDDING}, lazy local fallback)"
    )

    yield

    await model_holder.stop_idle_unload()
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

app.add_middleware(InternalAuthMiddleware)


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
        "model_loaded": model_holder.loaded,
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
    """
    await get_model()
    logger.info(
        f"[TRIGGER] web_bank_id={request.web_bank_id}, opd_id={request.opd_id}, "
        f"url={request.url}, css_selector={request.css_selector}, "
        f"scrape_interval={request.scrape_interval}, is_active={request.is_active}"
    )

    if not request.is_active:
        return await sync_module.register_inactive_web_bank(
            web_bank_id=request.web_bank_id,
            name=request.name,
            opd_id=request.opd_id,
            url=request.url,
            css_selector=request.css_selector,
            scrape_interval=request.scrape_interval,
            metadata=request.metadata,
        )

    if not sync_module.reserve_job(request.web_bank_id):
        logger.info(f"[TRIGGER] Duplicate in-flight scrape skipped: {request.web_bank_id}")
        return {
            "status": "skipped",
            "message": "Scraping untuk website ini masih berjalan",
            "web_bank_id": request.web_bank_id,
        }

    job_id = str(uuid.uuid4())

    background_tasks.add_task(
        sync_module.process_url,
        web_bank_id=request.web_bank_id,
        name=request.name,
        opd_id=request.opd_id,
        url=request.url,
        css_selector=request.css_selector,
        scrape_interval=request.scrape_interval,
        is_active=request.is_active,
        metadata=request.metadata,
        job_id=job_id,
    )

    return {
        "status": "processing",
        "message": "Scraping job started",
        "web_bank_id": request.web_bank_id,
        "job_id": job_id,
        "options": {
            "css_selector": request.css_selector,
            "scrape_interval": request.scrape_interval,
        }
    }


@app.put("/internal/update")
async def internal_update(request: UpdateRequest, background_tasks: BackgroundTasks):
    """Update web bank metadata and rescrape only if source settings changed."""
    await get_model()
    logger.info(
        f"[UPDATE] web_bank_id={request.web_bank_id}, opd_id={request.opd_id}, "
        f"url={request.url}, css_selector={request.css_selector}, "
        f"scrape_interval={request.scrape_interval}, is_active={request.is_active}"
    )

    return await sync_module.update_web_bank(
        web_bank_id=request.web_bank_id,
        name=request.name,
        opd_id=request.opd_id,
        url=request.url,
        css_selector=request.css_selector,
        scrape_interval=request.scrape_interval,
        is_active=request.is_active,
        metadata=request.metadata,
        background_tasks=background_tasks,
    )


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """Internal sync endpoint - untuk edited content."""
    await get_model()
    logger.info(f"[SYNC] web_bank_id={request.web_bank_id}")
    
    result = await sync_module.sync_edited_content(
        web_bank_id=request.web_bank_id,
        edited_content=request.edited_content
    )
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Content not found")
    
    return result


@app.delete("/internal/delete")
async def internal_delete(request: DeleteRequest):
    """Internal delete endpoint - soft delete indexed chunks and deactivate state."""
    logger.info(f"[DELETE] web_bank_id={request.web_bank_id}")
    
    return await sync_module.soft_delete_web_bank(request.web_bank_id)


@app.post("/internal/content")
async def internal_get_content(request: GetContentRequest):
    """Internal get content endpoint."""
    logger.info(f"[GET-CONTENT] web_bank_id={request.web_bank_id}")
    
    result = await sync_module.get_content(request.web_bank_id)
    
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
