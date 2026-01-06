"""
RAG Web Service - Main Application
FastAPI app untuk RAG Web Scraping (web_scraping_bank)
"""
import os
import sys
import uuid
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

# Import modules
from services.rag_web import search as search_module
from services.rag_web import sync as sync_module
from services.rag_web.models import (
    SearchRequest, UnifiedSearchRequest, TriggerRequest, SyncRequest, 
    DeleteRequest, GetContentRequest
)

# Setup logging
logger = setup_logging("rag_web")

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
    
    logger.info(f"RAG Web Service Started on port {config.WEB_SERVICE_PORT}")
    
    yield
    
    # Shutdown
    logger.info("RAG Web Service Shutting down...")


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
        "service": "rag_web",
        "components": {
            "embedding_model": model_ok,
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """Internal search endpoint - dipanggil oleh orchestrator (direct mode)."""
    logger.info(f"[SEARCH] Query: {request.query[:50]}...")
    
    result = await search_module.search_web_bank(
        question=request.query,  # search_web_bank expects 'question' parameter
        limit=request.limit
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/search-unified")
async def internal_search_unified(request: UnifiedSearchRequest):
    """Internal search endpoint untuk unified/parallel mode dari orchestrator.\"\"\"
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
    Trigger web scraping - jalankan di background.
    """
    logger.info(f"[TRIGGER] link_id={request.link_id}, url={request.url}")
    
    job_id = str(uuid.uuid4())
    
    # Run di background
    background_tasks.add_task(
        sync_module.process_url,
        link_id=request.link_id,
        url=request.url,
        metadata=request.metadata
    )
    
    return {
        "status": "processing",
        "message": "Scraping job started",
        "link_id": request.link_id,
        "job_id": job_id
    }


@app.post("/internal/sync")
async def internal_sync(request: SyncRequest):
    """Internal sync endpoint - untuk edited content."""
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
