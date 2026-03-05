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
            # Tambahan: content_type dan faq_question dari chunk metadata
            "content_type": chunk.metadata.get("content_type", "general"),
            "faq_question": chunk.metadata.get("faq_question", None),
            "metadata": metadata or {}
        }

        # Hapus key None agar payload bersih
        payload = {k: v for k, v in payload.items() if v is not None}

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
    css_selector: Optional[str] = None,
    use_js_renderer: Optional[bool] = None,
    wait_selector: Optional[str] = None,
    content_type: str = "general",
    faq_question_selector: Optional[str] = None,
    faq_answer_selector: Optional[str] = None,
    force_rescrape: bool = False,
    callback_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pipeline: scrape → clean/extract → chunk → embed → store."""
    from services.rag_web.webhook import send_callback

    try:
        logger.info(
            f"[SYNC] process_url start: link_id={link_id}, url={url}, "
            f"content_type={content_type}, css_selector={css_selector}, "
            f"use_js_renderer={use_js_renderer}, force_rescrape={force_rescrape}"
        )


        if not force_rescrape:
            existing = await get_chunks_by_link_id(link_id)
            if existing:
                logger.info(
                    f"[SYNC] link_id={link_id} sudah ada ({len(existing)} chunks). "
                    "Gunakan force_rescrape=True untuk memproses ulang."
                )
                result = {
                    "status": "skipped",
                    "reason": "already_exists",
                    "link_id": link_id,
                    "url": url,
                    "chunks_count": len(existing),
                }
                await send_callback(link_id, url, "skipped", result, callback_url)
                return result

        await send_callback(link_id, url, "scraping", {})
        scraped = await scraper.scrape(
            url=url,
            use_js_renderer=use_js_renderer,
            wait_selector=wait_selector,
        )
        raw_html = scraped.get("raw_html", "")


        title = cleaner.extract_title(raw_html)


        chunks: List[Chunk] = []

        if content_type == "faq":
            # Mode FAQ: ekstrak pasangan Q&A
            from services.rag_web.faq_extractor import extract_faq_pairs

            pairs = extract_faq_pairs(
                raw_html=raw_html,
                question_selector=faq_question_selector,
                answer_selector=faq_answer_selector,
            )

            if pairs:
                logger.info(f"[SYNC] FAQ mode: {len(pairs)} pasangan Q&A ditemukan")
                chunks = chunker.chunk_faq(pairs, url)
                base_metadata = {
                    **(metadata or {}),
                    "content_type": "faq",
                    "faq_pairs_count": len(pairs),
                }
            else:
                logger.warning(
                    "[SYNC] FAQ mode: tidak ada pasangan Q&A terdeteksi, "
                    "fallback ke general extraction"
                )
                # Fallback ke general
                content_type = "general"

        if content_type in ("general", "article") or (content_type == "faq" and not chunks):
            # Mode general/article: clean → chunk
            if css_selector:
                clean_content = cleaner.clean_with_selector(raw_html, css_selector, url)
            else:
                clean_content = cleaner.clean(raw_html, url)

            if not clean_content or len(clean_content.strip()) < 50:
                raise Exception(
                    f"Konten terlalu pendek atau kosong setelah cleaning ({len(clean_content or '')} chars)"
                )

            chunks = chunker.chunk(clean_content, url)
            base_metadata = {
                **(metadata or {}),
                "content_type": content_type,
            }

        if not chunks:
            raise Exception("Tidak ada chunk yang berhasil dibuat")


        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await encode_texts(chunk_texts, model=model, prefix="passage: ")


        await hard_delete_by_link_id(link_id)
        await store_chunks(
            link_id=link_id,
            url=url,
            title=title,
            chunks=chunks,
            embeddings=embeddings,
            metadata=base_metadata
        )


        result = {
            "status": "success",
            "link_id": link_id,
            "url": url,
            "title": title,
            "chunks_count": len(chunks),
            "content_length": sum(len(c.content) for c in chunks),
            "content_type": content_type,
        }
        if content_type == "faq" and "faq_pairs_count" in base_metadata:
            result["faq_pairs_count"] = base_metadata["faq_pairs_count"]

        logger.info(
            f"[SYNC] Selesai: {len(chunks)} chunks untuk link_id={link_id} "
            f"(content_type={content_type})"
        )

        await send_callback(link_id, url, "completed", result, callback_url)
        return result

    except Exception as e:
        logger.exception(f"[SYNC] Error processing URL link_id={link_id}: {e}")
        error_result = {
            "status": "failed",
            "link_id": link_id,
            "url": url,
            "error": str(e),
        }
        await send_callback(link_id, url, "failed", error_result, callback_url)
        return error_result


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
