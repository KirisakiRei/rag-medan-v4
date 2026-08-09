"""Shared embedding microservice with ThreadPoolExecutor for CPU-bound encoding."""
import os
import sys
import gc
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import uvicorn
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.security import InternalAuthMiddleware
from services.embedding_service.models import EmbedRequest, EmbedResponse

logger = setup_logging("embedding_service")

_small_model: SentenceTransformer = None
_large_model: SentenceTransformer = None
_large_model_lock = asyncio.Lock()
_last_large_model_used: float = 0.0

_thread_pool: ThreadPoolExecutor = None


def _encode_sync(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Synchronous encoding - runs inside thread pool."""
    embeddings = model.encode(texts)
    return embeddings.tolist()


async def get_small_model() -> SentenceTransformer:
    """Return pre-warmed small model."""
    if _small_model is None:
        raise HTTPException(status_code=503, detail="Small model not loaded yet")
    return _small_model


async def get_large_model() -> SentenceTransformer:
    """Lazy load large model with thundering herd protection."""
    global _large_model, _last_large_model_used

    if _large_model is not None:
        _last_large_model_used = time.time()
        return _large_model

    async with _large_model_lock:
        if _large_model is not None:
            _last_large_model_used = time.time()
            return _large_model

        logger.info("Loading LARGE embedding model (lazy)...")
        loop = asyncio.get_event_loop()
        _large_model = await loop.run_in_executor(
            _thread_pool,
            lambda: SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)
        )
        _last_large_model_used = time.time()
        logger.info(f"Large model loaded: {config.EMBEDDING_MODEL_PATH_LARGE}")
        return _large_model


async def _idle_unload_large_loop():
    """Background task: unload large model after idle timeout."""
    global _large_model, _last_large_model_used
    timeout = config.LARGE_MODEL_IDLE_TIMEOUT

    while True:
        await asyncio.sleep(300)
        if _large_model is not None and _last_large_model_used > 0:
            idle_seconds = time.time() - _last_large_model_used
            if idle_seconds > timeout:
                async with _large_model_lock:
                    if _large_model is not None and (time.time() - _last_large_model_used) > timeout:
                        logger.info(f"Large model idle for {idle_seconds:.0f}s > {timeout}s, unloading...")
                        del _large_model
                        _large_model = None
                        gc.collect()
                        logger.info("Large model unloaded, RAM freed")


# ============== LIFESPAN ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm small model, start idle unload loop for large model."""
    global _small_model, _thread_pool

    # Create thread pool
    _thread_pool = ThreadPoolExecutor(
        max_workers=config.EMBEDDING_THREAD_POOL_SIZE,
        thread_name_prefix="emb"
    )

    logger.info("Pre-warming SMALL embedding model...")
    _small_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)
    logger.info(f"Small model ready: {config.EMBEDDING_MODEL_PATH} (dim={config.EMBEDDING_DIMENSION})")

    asyncio.create_task(_idle_unload_large_loop())

    logger.info(f"Embedding Service Started on port {config.EMBEDDING_SERVICE_PORT}")

    yield
    
    logger.info("Embedding Service Shutting down...")
    _thread_pool.shutdown(wait=False)


app = FastAPI(
    title="Embedding Service",
    description="Shared embedding model service for all RAG services",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(InternalAuthMiddleware)


# ============== ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "embedding_service",
        "small_model_loaded": _small_model is not None,
        "large_model_loaded": _large_model is not None,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Encode texts into embeddings using small or large model."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    if request.model_size == "large":
        model = await get_large_model()
        dimension = config.EMBEDDING_DIMENSION_LARGE
    else:
        model = await get_small_model()
        dimension = config.EMBEDDING_DIMENSION

    prefixed = [f"{request.prefix}{t}" for t in request.texts]

    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        _thread_pool,
        _encode_sync,
        model,
        prefixed
    )

    return EmbedResponse(
        embeddings=embeddings,
        dimension=dimension,
        model_size=request.model_size,
        count=len(embeddings)
    )


def start_service():
    """Start the service."""
    uvicorn.run(
        "services.embedding_service.main:app",
        host="0.0.0.0",
        port=config.EMBEDDING_SERVICE_PORT,
        reload=False,
        log_config=None
    )


if __name__ == "__main__":
    start_service()
