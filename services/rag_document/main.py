"""
RAG Document Service - Main Application
FastAPI app untuk RAG Document (document_bank)
"""
import os
import sys
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

# Import modules
from services.rag_document import search as search_module
from services.rag_document import sync as sync_module
from services.rag_document import delete as delete_module
from services.rag_document.models import SearchRequest, UnifiedSearchRequest, SyncRequest, DeleteRequest

# Setup logging
logger = setup_logging("rag_document")

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
        
        if config.COLLECTION_DOCUMENT not in collection_names:
            logger.info(f"Creating collection: {config.COLLECTION_DOCUMENT}")
            await qdrant.create_collection(
                collection_name=config.COLLECTION_DOCUMENT,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION_LARGE,
                    distance=Distance.COSINE
                )
            )
            
            # Create indexes
            await qdrant.create_payload_index(
                collection_name=config.COLLECTION_DOCUMENT,
                field_name="doc_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD
            )
            await qdrant.create_payload_index(
                collection_name=config.COLLECTION_DOCUMENT,
                field_name="is_deleted",
                field_schema=qdrant_models.PayloadSchemaType.BOOL
            )
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")
    
    logger.info(f"Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


def init_model():
    """Initialize embedding model (large)."""
    global model
    logger.info("Loading large embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)
    logger.info(f"Model loaded: {config.EMBEDDING_MODEL_PATH_LARGE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Startup
    init_model()
    await init_qdrant()
    
    # Set instances ke modules
    search_module.set_instances(model, qdrant)
    delete_module.set_instances(qdrant)
    
    logger.info(f"RAG Document Service Started on port {config.DOCUMENT_SERVICE_PORT}")
    
    yield
    
    # Shutdown
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
        "service": "rag_document",
        "components": {
            "embedding_model": model_ok,
            "qdrant": qdrant_ok
        }
    }


@app.post("/internal/search")
async def internal_search(request: SearchRequest):
    """Internal search endpoint - dipanggil oleh orchestrator (direct mode)."""
    logger.info(f"[SEARCH] Query: {request.query[:50]}...")
    
    result = await search_module.search_document_bank(
        query=request.query,
        limit=request.limit
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/internal/search-unified")
async def internal_search_unified(request: UnifiedSearchRequest):
    """
    Internal search endpoint untuk unified/parallel mode.
    Dipanggil oleh orchestrator saat /api/search.
    Return format sama dengan text service untuk selection.
    """
    logger.info(f"[SEARCH-UNIFIED] Question: {request.question[:50]}...")
    
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
    logger.info(f"[SYNC] doc_id={request.doc_id}")
    
    result = await sync_module.sync_document(
        doc_id=request.doc_id,
        file_url=request.file_url,
        opd_name=request.opd_name
    )
    
    return JSONResponse(status_code=200, content=result)


@app.get("/internal/sync/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    result = sync_module.get_task_status(task_id)
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    
    return result


@app.get("/internal/sync/tasks")
async def list_tasks():
    """List all tasks."""
    return sync_module.list_all_tasks()


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
