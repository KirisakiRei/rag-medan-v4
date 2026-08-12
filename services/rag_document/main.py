import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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

from services.rag_document import search as search_module
from services.rag_document import sync as sync_module
from services.rag_document import delete as delete_module
from services.rag_document.models import SearchRequest, UnifiedSearchRequest, SyncRequest, DeleteRequest

logger = setup_logging("rag_document")

qdrant: AsyncQdrantClient = None


def _wire_model(model):
    """Wire loaded local model into search module."""
    search_module.set_instances(model, qdrant)


model_holder = LazyModel(
    config.EMBEDDING_MODEL_PATH_LARGE,
    on_load=_wire_model,
    name="large embedding",
)


async def get_model():
    """Return local large model (None saat shared embedding aktif)."""
    return await model_holder.get()


async def init_qdrant():
    """Initialize Qdrant connection."""
    global qdrant
    qdrant = create_qdrant_client()

    try:
        collections = await qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if config.COLLECTION_DOCUMENT not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_DOCUMENT}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_DOCUMENT,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION_LARGE,
                    distance=Distance.COSINE
                )
            )

        await ensure_payload_index(
            qdrant,
            config.COLLECTION_DOCUMENT,
            "mysql_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_DOCUMENT,
            "is_deleted",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_DOCUMENT,
            "is_active",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_DOCUMENT,
            "chunk_level",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await ensure_payload_index(
            qdrant,
            config.COLLECTION_DOCUMENT,
            "parent_chunk_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await backfill_is_active(qdrant, config.COLLECTION_DOCUMENT)
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")

    # Wire Qdrant-dependent modules immediately so delete/search paths are ready
    # even before the lazy embedding model is first loaded.
    search_module.set_instances(model_holder.model, qdrant)
    delete_module.set_instances(qdrant)

    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — model is lazy loaded on first request."""
    await init_qdrant()

    model_holder.start_idle_unload()

    logger.info(f"RAG Document Service Started on port {config.DOCUMENT_SERVICE_PORT}")
    logger.info(f"  Model loading: shared={config.USE_SHARED_EMBEDDING} (lazy local fallback)")
    logger.info(f"  Idle timeout: {config.MODEL_IDLE_TIMEOUT}s")
    logger.info(f"  Shared embedding: {config.USE_SHARED_EMBEDDING}")

    yield

    await model_holder.stop_idle_unload()
    logger.info("RAG Document Service Shutting down...")


app = FastAPI(
    title="RAG Document Service",
    description="Service untuk RAG Document - Document Bank (OCR)",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(InternalAuthMiddleware)


# ============== ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model_holder.loaded
    model_ok = False

    if model_loaded:
        try:
            _ = model_holder.model.encode("test").tolist()
            model_ok = True
        except:
            pass
    
    try:
        await qdrant.get_collections()
        qdrant_ok = True
    except:
        qdrant_ok = False
    
    status = "healthy" if qdrant_ok else "unhealthy"
    
    return {
        "status": status,
        "service": "rag_document",
        "components": {
            "embedding_model": model_ok,
            "model_loaded": model_loaded,
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """Internal search endpoint - dipanggil oleh orchestrator (direct mode)."""
    logger.info(f"[SEARCH] Query: {request.query[:50]}...")

    await get_model()

    result = await search_module.search_document_bank(
        query=request.query,
        limit=request.limit
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/search-unified")
async def internal_search_unified(request: UnifiedSearchRequest):
    logger.info(f"[SEARCH-UNIFIED] Question: {request.question[:50]}...")

    await get_model()

    result = await search_module.search_document_unified(
        question=request.question,
        original_question=request.original_question,
        wa_number=request.wa_number,
        top_k=request.top_k
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """Internal sync endpoint - trigger OCR worker."""
    logger.info(
        f"[SYNC] doc_id={request.doc_id} | org={request.organization_id} | "
        f"filename={request.filename} | is_active={request.is_active}"
    )

    result = await sync_module.sync_document(
        doc_id=request.doc_id,
        file_url=request.file_url,
        organization_id=request.organization_id,
        filename=request.filename,
        is_active=request.is_active,
    )

    return JSONResponse(status_code=200, content=result)


@app.get("/internal/sync/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    result = sync_module.get_task_status(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return result


@app.get("/internal/sync/tasks")
async def list_tasks():
    """List all tasks."""
    return sync_module.get_all_tasks()


@app.delete("/internal/delete")
async def internal_delete(request: DeleteRequest):
    """Internal delete endpoint - soft delete."""
    logger.info(f"[DELETE] doc_id={request.doc_id}")
    
    result = await delete_module.soft_delete_document(request.doc_id)
    
    return JSONResponse(status_code=200, content=result)


def start_service():
    """Start the service."""
    uvicorn.run(
        "services.rag_document.main:app",
        host="0.0.0.0",
        port=config.DOCUMENT_SERVICE_PORT,
        reload=False,
        log_config=None
    )


if __name__ == "__main__":
    start_service()
