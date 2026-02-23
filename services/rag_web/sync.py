"""Sync module for web_scraping_bank."""
import os
import sys
import time
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from services.rag_web.scraper import scraper
from services.rag_web.cleaner import cleaner
from services.rag_web.chunker import chunker, Chunk
from shared.utils import encode_texts

logger = logging.getLogger("rag_web.sync")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def store_chunks(
    link_id: str,
    url: str,
    title: Optional[str],
    chunks: List[Chunk],
    embeddings: List[List[float]],
    metadata: Dict[str, Any] = None
) -> List[str]:
    """Store chunks to Qdrant."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")
    
    points = []
    point_ids = []
    now = datetime.utcnow().isoformat()
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)
        
        payload = {
            "link_id": link_id,
            "url": url,
            "title": title or "",
            "content": chunk.content,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "token_count": chunk.token_count,
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {}
        }
        
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
    
    await qdrant.upsert(collection_name=config.COLLECTION_WEB, points=points)
    logger.info(f"[SYNC] Stored {len(points)} points for link_id: {link_id}")
    
    return point_ids


async def get_chunks_by_link_id(
    link_id: str,
    include_deleted: bool = False
) -> List[Dict[str, Any]]:
    """Get chunks by link_id."""
    filter_conditions = [
        qdrant_models.FieldCondition(
            key="link_id",
            match=qdrant_models.MatchValue(value=link_id)
        )
    ]
    
    if not include_deleted:
        filter_conditions.append(
            qdrant_models.FieldCondition(
                key="is_deleted",
                match=qdrant_models.MatchValue(value=False)
            )
        )
    
    results = await qdrant.scroll(
        collection_name=config.COLLECTION_WEB,
        scroll_filter=qdrant_models.Filter(must=filter_conditions),
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = [{"id": point.id, **point.payload} for point in results[0]]
    chunks.sort(key=lambda x: x.get("chunk_index", 0))
    
    return chunks


async def soft_delete_by_link_id(link_id: str) -> int:
    """Soft delete chunks by link_id."""
    chunks = await get_chunks_by_link_id(link_id)
    
    if not chunks:
        return 0
    
    point_ids = [chunk["id"] for chunk in chunks]
    now = datetime.utcnow().isoformat()
    
    await qdrant.set_payload(
        collection_name=config.COLLECTION_WEB,
        payload={"is_deleted": True, "deleted_at": now, "updated_at": now},
        points=point_ids
    )
    
    logger.info(f"[SYNC] Soft deleted {len(point_ids)} chunks for link_id: {link_id}")
    return len(point_ids)


async def hard_delete_by_link_id(link_id: str) -> int:
    """Hard delete chunks by link_id."""
    results = await qdrant.scroll(
        collection_name=config.COLLECTION_WEB,
        scroll_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="link_id",
                    match=qdrant_models.MatchValue(value=link_id)
                )
            ]
        ),
        limit=1000,
        with_payload=False,
        with_vectors=False
    )
    
    points = results[0]
    
    if not points:
        return 0
    
    point_ids = [point.id for point in points]
    
    await qdrant.delete(
        collection_name=config.COLLECTION_WEB,
        points_selector=qdrant_models.PointIdsList(points=point_ids)
    )
    
    logger.info(f"[SYNC] Hard deleted {len(point_ids)} chunks for link_id: {link_id}")
    return len(point_ids)


async def process_url(
    link_id: str,
    url: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process URL: scrape → clean → chunk → embed → store."""
    try:
        logger.info(f"[SYNC] Processing URL: {url}")
        
        # 1. Scrape
        scraped = await scraper.scrape(url)
        raw_html = scraped.get("raw_html", "")
        
        # 2. Clean
        clean_content = cleaner.clean(raw_html, url)
        title = cleaner.extract_title(raw_html)
        
        if not clean_content or len(clean_content.strip()) < 50:
            raise Exception("Content too short or empty")
        
        # 3. Chunk
        chunks = chunker.chunk(clean_content, url)
        if not chunks:
            raise Exception("No chunks created")
        
        # 4. Embed
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await encode_texts(chunk_texts, model=model, prefix="passage: ")
        
        # 5. Delete existing dan store baru
        await hard_delete_by_link_id(link_id)
        await store_chunks(
            link_id=link_id,
            url=url,
            title=title,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )
        
        logger.info(f"[SYNC] Completed: {len(chunks)} chunks stored for link_id={link_id}")
        
        return {
            "status": "success",
            "link_id": link_id,
            "url": url,
            "title": title,
            "chunks_count": len(chunks),
            "content_length": len(clean_content)
        }
        
    except Exception as e:
        logger.exception(f"[SYNC] Error processing URL: {e}")
        return {
            "status": "error",
            "link_id": link_id,
            "error": str(e)
        }


async def sync_edited_content(
    link_id: str,
    edited_content: str
) -> Dict[str, Any]:
    """Sync edited content (user-edited)."""
    try:
        logger.info(f"[SYNC] Syncing edited content for link_id={link_id}")
        
        # Get existing data untuk URL dan title
        existing_chunks = await get_chunks_by_link_id(link_id, include_deleted=True)
        
        if not existing_chunks:
            return {
                "status": "not_found",
                "link_id": link_id,
                "error": "Content not found"
            }
        
        url = existing_chunks[0].get("url", "")
        title = existing_chunks[0].get("title", "")
        
        # Chunk new content
        chunks = chunker.chunk(edited_content, url)
        if not chunks:
            return {
                "status": "error",
                "link_id": link_id,
                "error": "No chunks created from edited content"
            }
        
        # Embed
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await encode_texts(chunk_texts, model=model, prefix="passage: ")
        
        # Delete old dan store new
        await hard_delete_by_link_id(link_id)
        await store_chunks(
            link_id=link_id,
            url=url,
            title=title,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"is_edited": True, "edited_at": datetime.utcnow().isoformat()}
        )
        
        logger.info(f"[SYNC] Updated: {len(chunks)} chunks for link_id={link_id}")
        
        return {
            "status": "success",
            "link_id": link_id,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        logger.exception(f"[SYNC] Error syncing edited content: {e}")
        return {
            "status": "error",
            "link_id": link_id,
            "error": str(e)
        }


async def get_content(link_id: str) -> Dict[str, Any]:
    """Get combined content by link_id."""
    chunks = await get_chunks_by_link_id(link_id)
    
    if not chunks:
        return {
            "status": "not_found",
            "link_id": link_id
        }
    
    # Combine chunks
    clean_content = "\n\n".join([c.get("content", "") for c in chunks])
    
    return {
        "status": "success",
        "link_id": link_id,
        "url": chunks[0].get("url", ""),
        "title": chunks[0].get("title", ""),
        "clean_content": clean_content,
        "chunks_count": len(chunks),
        "created_at": chunks[0].get("created_at", ""),
        "updated_at": chunks[0].get("updated_at", "")
    }
