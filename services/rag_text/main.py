import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.security import InternalAuthMiddleware
from shared.bootstrap import LazyModel, create_qdrant_client

from services.rag_text import search as search_module
from services.rag_text import sync as sync_module
from services.rag_text.models import SearchRequest, UnifiedSearchRequest, SyncRequest

# Setup logging
logger = setup_logging("rag_text")

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

    # Wire Qdrant-dependent modules immediately (model di-wire saat lazy load)
    search_module.set_instances(model_holder.model, qdrant)

    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — model is lazy loaded on first request."""
    await init_qdrant()

    model_holder.start_idle_unload()

    logger.info(f"RAG Text Service Started on port {config.TEXT_SERVICE_PORT}")
    logger.info(f"  Model loading: shared={config.USE_SHARED_EMBEDDING} (lazy local fallback)")
    logger.info(f"  Idle timeout: {config.MODEL_IDLE_TIMEOUT}s")
    logger.info(f"  Shared embedding: {config.USE_SHARED_EMBEDDING}")

    yield

    await model_holder.stop_idle_unload()
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

    await get_model()

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

    await get_model()

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

    await get_model()

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
