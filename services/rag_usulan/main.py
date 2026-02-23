"""FastAPI app for RAG Usulan service."""
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

from services.rag_usulan import search as search_module
from services.rag_usulan import sync as sync_module
from services.rag_usulan.models import SearchRequest, SyncRequest

logger = setup_logging("rag_usulan")

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
        
        if config.COLLECTION_USULAN not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_USULAN}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_USULAN,
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
    """Application lifespan - model loaded lazily on first request."""
    await init_qdrant()
    
    asyncio.create_task(_idle_unload_loop())
    
    logger.info(f"RAG Usulan Service Started on port {config.USULAN_SERVICE_PORT} (model: lazy load)")
    
    yield
    
    logger.info("RAG Usulan Service Shutting down...")


app = FastAPI(
    title="RAG Usulan Service",
    description="Service untuk RAG Usulan - Usulan Bank",
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
        "service": "rag_usulan",
        "model_loaded": _model is not None,
        "components": {
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """Internal search endpoint - dipanggil oleh orchestrator."""
    await get_model()
    logger.info(f"[SEARCH] Question: {request.question[:50]}...")
    
    result = await search_module.search_usulan_bank(
        question=request.question,
        wa_number=request.wa_number
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """Internal sync endpoint - dipanggil oleh orchestrator."""
    await get_model()
    logger.info(f"[SYNC] Action: {request.action}")
    
    result = await sync_module.sync_usulan(
        action=request.action,
        content=request.content
    )
    
    return JSONResponse(status_code=200, content=result)


def start_service():
    """Start the service."""
    uvicorn.run(
        "services.rag_usulan.main:app",
        host="0.0.0.0",
        port=config.USULAN_SERVICE_PORT,
        reload=False,
        log_config=None
    )


if __name__ == "__main__":
    start_service()
