"""
RAG Usulan Service - Main Application
FastAPI app untuk RAG Usulan (usulan_bank)
"""
import os
import sys
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

# Import modules
from services.rag_usulan import search as search_module
from services.rag_usulan import sync as sync_module
from services.rag_usulan.models import SearchRequest, SyncRequest

# Setup logging
logger = setup_logging("rag_usulan")

# Global instances
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


async def init_qdrant():
    """Initialize Qdrant connection."""
    global qdrant
    
    if config.QDRANT_API_KEY:
        qdrant = AsyncQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
        )
    else:
        qdrant = AsyncQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
        )
    
    # Ensure collection exists
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


def init_model():
    """Initialize embedding model."""
    global model
    logger.info("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_PATH)
    logger.info(f"Model loaded: {config.EMBEDDING_MODEL_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Startup
    init_model()
    await init_qdrant()
    
    # Set instances ke modules
    search_module.set_instances(model, qdrant)
    sync_module.set_instances(model, qdrant)
    
    logger.info(f"RAG Usulan Service Started on port {config.USULAN_SERVICE_PORT}")
    
    yield
    
    # Shutdown
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
        _ = model.encode("test").tolist()
        model_ok = True
    except:
        model_ok = False
    
    try:
        await qdrant.get_collections()
        qdrant_ok = True
    except:
        qdrant_ok = False
    
    status = "healthy" if model_ok and qdrant_ok else "unhealthy"
    
    return {
        "status": status,
        "service": "rag_usulan",
        "components": {
            "embedding_model": model_ok,
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """Internal search endpoint - dipanggil oleh orchestrator."""
    logger.info(f"[SEARCH] Question: {request.question[:50]}...")
    
    result = await search_module.search_usulan_bank(
        question=request.question,
        wa_number=request.wa_number
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """Internal sync endpoint - dipanggil oleh orchestrator."""
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
