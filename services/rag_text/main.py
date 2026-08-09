import os
import sys
import gc
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.security import InternalAuthMiddleware

from services.rag_text import search as search_module
from services.rag_text import sync as sync_module
from services.rag_text.models import SearchRequest, UnifiedSearchRequest, SyncRequest

# Setup logging
logger = setup_logging("rag_text")

_model: SentenceTransformer = None
_model_lock = asyncio.Lock()
_last_model_used: float = 0.0
qdrant: AsyncQdrantClient = None


async def get_model() -> SentenceTransformer:
    """
    Lazy load embedding model with thundering herd protection.
    If USE_SHARED_EMBEDDING is True, this should not be called (use encode_texts instead).
    """
    global _model, _last_model_used
    
    if _model is not None:
        _last_model_used = time.time()
        return _model
    
    async with _model_lock:
        # Double-check after acquiring lock (another request may have loaded it)
        if _model is not None:
            _last_model_used = time.time()
            return _model
        
        logger.info("Loading embedding model (lazy)...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)
        _last_model_used = time.time()
        logger.info(f"Model loaded: {config.EMBEDDING_MODEL_PATH}")
        
        # Update module references
        search_module.set_instances(_model, qdrant)
        sync_module.set_instances(_model, qdrant)
        
        return _model


async def _idle_unload_loop():
    """Background task: unload model after IDLE_TIMEOUT seconds of inactivity."""
    global _model, _last_model_used
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
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
        
        if config.COLLECTION_TEXT not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_TEXT}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_TEXT,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")
    
    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — model is lazy loaded on first request."""
    await init_qdrant()
    
    # Start idle unload background task
    idle_task = asyncio.create_task(_idle_unload_loop())
    
    logger.info(f"RAG Text Service Started on port {config.TEXT_SERVICE_PORT}")
    logger.info(f"  Model loading: LAZY (will load on first request)")
    logger.info(f"  Idle timeout: {config.MODEL_IDLE_TIMEOUT}s")
    logger.info(f"  Shared embedding: {config.USE_SHARED_EMBEDDING}")
    
    yield
    
    # Shutdown
    idle_task.cancel()
    logger.info("RAG Text Service Shutting down...")


app = FastAPI(
    title="RAG Text Service",
    description="Service untuk RAG Text - Knowledge Bank",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(InternalAuthMiddleware)


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
    
    # Service is healthy if qdrant is connected (model will lazy load)
    status = "healthy" if qdrant_ok else "unhealthy"
    
    return {
        "status": status,
        "service": "rag_text",
        "components": {
            "embedding_model": model_ok,
            "model_loaded": model_loaded,
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    logger.info(f"[SEARCH] Question: {request.question[:50]}... | skip_prefilter: {request.skip_prefilter}")
    
    model = await get_model()
    search_module.set_instances(model, qdrant)
    
    result = await search_module.search_knowledge_bank(
        question=request.question,
        wa_number=request.wa_number,
        original_question=request.original_question,
        skip_prefilter=request.skip_prefilter
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/search-unified")
async def internal_search_unified(request: UnifiedSearchRequest):
    """
    Internal search endpoint for unified/parallel mode.
    Called by orchestrator for /api/search.
    """
    logger.info(f"[SEARCH-UNIFIED] Question: {request.question[:50]}...")
    
    model = await get_model()
    search_module.set_instances(model, qdrant)
    
    result = await search_module.search_knowledge_bank(
        question=request.question,
        wa_number=request.wa_number,
        original_question=request.original_question,
        skip_prefilter=True,
        top_k=request.top_k
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """
    Internal sync endpoint.
    """
    logger.info(f"[SYNC] Action: {request.action}")
    
    model = await get_model()
    sync_module.set_instances(model, qdrant)
    
    result = await sync_module.sync_data(
        action=request.action,
        content=request.content
    )
    
    return JSONResponse(status_code=200, content=result)


def start_service():
    """Start the service."""
    uvicorn.run(
        "services.rag_text.main:app",
        host="0.0.0.0",
        port=config.TEXT_SERVICE_PORT,
        reload=False,
        log_config=None
    )


if __name__ == "__main__":
    start_service()
