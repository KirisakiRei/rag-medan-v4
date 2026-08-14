"""Sync module for usulan_bank."""
import os
import sys
import logging
import traceback
import hashlib
import asyncio
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.utils import encode_texts
from shared.lightrag_sync import sync_lightrag_usulan, delete_lightrag_source

logger = logging.getLogger("rag_usulan.sync")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def _sync_usulan_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Index one usulan ke LightRAG dan tunggu hingga PROCESSED."""
    source_id = str(item["request_rag_id"])
    request_rag_name = str(item.get("request_rag_name") or "").strip()
    if not request_rag_name:
        raise ValueError(f"request_rag_name kosong untuk request_rag_id={source_id}")
    organization_id = item.get("organization_id")
    content_hash = hashlib.sha256(
        f"{request_rag_name}\n{organization_id or ''}".encode("utf-8")
    ).hexdigest()
    return await sync_lightrag_usulan(
        source_id=source_id,
        title=request_rag_name,
        content="",
        content_hash=content_hash,
        is_active=True,
        organization_id=organization_id,
        request_id=item.get("request_id"),
        request_name=item.get("request_name"),
        question=request_rag_name,
    )


async def sync_usulan(action: str, content: Any) -> Dict[str, Any]:
    """Sync data to usulan_bank."""
    collection = config.COLLECTION_USULAN
    
    try:
        if action == "bulk_sync":
            if not isinstance(content, list):
                return {
                    "status": "error",
                    "error": {"type": "ValidationError", "message": "Content harus berupa list"}
                }
            
            vectors = await encode_texts([item["request_rag_name"] for item in content], model=model, prefix="passage: ")
            points = []
            for item, vector in zip(content, vectors):
                point_id = str(item["request_rag_id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "request_id": item["request_id"],
                        "organization_id": item["organization_id"],
                        "request_name": item["request_name"],
                        "request_rag_name": item["request_rag_name"]
                    }
                })
            
            await qdrant.upsert(collection_name=collection, points=points)
            
            await qdrant.create_payload_index(
                collection_name=collection,
                field_name="request_rag_name",
                field_schema=qdrant_models.TextIndexParams(
                    type="text",
                    tokenizer=qdrant_models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )
            
            logger.info(f"[SYNC-USULAN] Sinkronisasi {len(points)} data ke {collection}")

            semaphore = asyncio.Semaphore(5)

            async def sync_bounded(item):
                async with semaphore:
                    return await _sync_usulan_item(item)

            await asyncio.gather(*(sync_bounded(item) for item in content))

            return {
                "status": "success",
                "message": f"{len(points)} data berhasil disinkronkan ke {collection}"
            }
        
        elif action in ["add", "update"]:
            point_id = str(content["request_rag_id"])
            [vector] = await encode_texts([content["request_rag_name"]], model=model, prefix="passage: ")
            await qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "request_id": content["request_id"],
                        "organization_id": content["organization_id"],
                        "request_name": content["request_name"],
                        "request_rag_name": content["request_rag_name"]
                    }
                }]
            )
            await _sync_usulan_item(content)
            logger.info(f"[SYNC-USULAN] Data {action} berhasil (ID={point_id})")
            
            return {"status": "success", "message": f"Data {action} berhasil"}
        
        elif action == "delete":
            point_id = str(content["request_rag_id"])
            await qdrant.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
                wait=True
            )
            await delete_lightrag_source(source_type="usulan", source_id=point_id)
            logger.info(f"[SYNC-USULAN] Data dihapus (ID={point_id})")
            
            return {"status": "success", "message": "Data berhasil dihapus"}
        
        else:
            return {
                "status": "error",
                "error": {"type": "ValidationError", "message": f"Action '{action}' tidak dikenali"}
            }

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"[ERROR][sync_usulan] {str(e)}\n{error_traceback}")
        return {
            "status": "error",
            "error": {"type": "ServerError", "message": "Kesalahan internal saat sinkronisasi usulan", "detail": str(e)}
        }
