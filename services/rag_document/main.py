import os
import sys
import gc
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging

from services.rag_document import search as search_module
from services.rag_document import sync as sync_module
from services.rag_document import delete as delete_module
from services.rag_document.models import SearchRequest, UnifiedSearchRequest, SyncRequest, DeleteRequest

logger = setup_logging("rag_document")

_model: SentenceTransformer = None
_model_lock = asyncio.Lock()
_last_model_used: float = 0.0
qdrant: AsyncQdrantClient = None


async def _ensure_payload_index(collection_name: str, field_name: str, field_schema) -> None:
    try:
        await qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema
        )
    except Exception:
        pass


async def _backfill_is_active(collection_name: str) -> None:
    """Backfill missing is_active payload for legacy document points."""
    try:
        offset = None
        updated_active = 0
        updated_inactive = 0
        while True:
            points, next_offset = await qdrant.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            active_ids = []
            inactive_ids = []
            for point in points:
                payload = dict(point.payload or {})
                if "is_active" in payload:
                    continue
                if payload.get("is_deleted", False):
                    inactive_ids.append(point.id)
                else:
                    active_ids.append(point.id)

            if active_ids:
                await qdrant.set_payload(
                    collection_name=collection_name,
                    payload={"is_active": True},
                    points=active_ids,
                )
                updated_active += len(active_ids)
            if inactive_ids:
                await qdrant.set_payload(
                    collection_name=collection_name,
                    payload={"is_active": False},
                    points=inactive_ids,
                )
                updated_inactive += len(inactive_ids)

            if next_offset is None:
                break
            offset = next_offset

        if updated_active or updated_inactive:
            logger.info(
                f"Backfilled is_active on {collection_name}: "
                f"active={updated_active}, inactive={updated_inactive}"
            )
    except Exception as exc:
        logger.warning(f"Backfill is_active skipped for {collection_name}: {exc}")


async def get_model() -> SentenceTransformer:
    """
    Lazy load large embedding model with thundering herd protection.
    """
    global _model, _last_model_used
    
    if _model is not None:
        _last_model_used = time.time()
        return _model
    
    async with _model_lock:
        if _model is not None:
            _last_model_used = time.time()
            return _model
        
        logger.info("Loading large embedding model (lazy)...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)
        _last_model_used = time.time()
        logger.info(f"Model loaded: {config.EMBEDDING_MODEL_PATH_LARGE}")
        
        # Update module references
        search_module.set_instances(_model, qdrant)
        
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
        
        if config.COLLECTION_DOCUMENT not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_DOCUMENT}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_DOCUMENT,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION_LARGE,
                    distance=Distance.COSINE
                )
            )

        await _ensure_payload_index(
            config.COLLECTION_DOCUMENT,
            "mysql_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await _ensure_payload_index(
            config.COLLECTION_DOCUMENT,
            "is_deleted",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await _ensure_payload_index(
            config.COLLECTION_DOCUMENT,
            "is_active",
            qdrant_models.PayloadSchemaType.BOOL
        )
        await _ensure_payload_index(
            config.COLLECTION_DOCUMENT,
            "chunk_level",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await _ensure_payload_index(
            config.COLLECTION_DOCUMENT,
            "parent_chunk_id",
            qdrant_models.PayloadSchemaType.KEYWORD
        )
        await _backfill_is_active(config.COLLECTION_DOCUMENT)
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")
    
    # Wire Qdrant-dependent modules immediately so delete/search paths are ready
    # even before the lazy embedding model is first loaded.
    search_module.set_instances(_model, qdrant)
    delete_module.set_instances(qdrant)

    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — model is lazy loaded on first request."""
    await init_qdrant()

    idle_task = asyncio.create_task(_idle_unload_loop())
    
    logger.info(f"RAG Document Service Started on port {config.DOCUMENT_SERVICE_PORT}")
    logger.info(f"  Model loading: LAZY (will load on first request)")
    logger.info(f"  Idle timeout: {config.MODEL_IDLE_TIMEOUT}s")
    logger.info(f"  Shared embedding: {config.USE_SHARED_EMBEDDING}")
    
    yield
    
    idle_task.cancel()
    logger.info("RAG Document Service Shutting down...")


app = FastAPI(
    title="RAG Document Service",
    description="Service untuk RAG Document - Document Bank (OCR)",
    version="3.0.0",
    lifespan=lifespan
)


# ============== ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = _model is not None
    model_ok = False
    
    if model_loaded:
        try:
            _ = _model.encode("test").tolist()
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
    
    model = await get_model()
    search_module.set_instances(model, qdrant)
    
    result = await search_module.search_document_bank(
        query=request.query,
        limit=request.limit
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/search-unified")
async def internal_search_unified(request: UnifiedSearchRequest):
    logger.info(f"[SEARCH-UNIFIED] Question: {request.question[:50]}...")
    
    model = await get_model()
    search_module.set_instances(model, qdrant)
    
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
