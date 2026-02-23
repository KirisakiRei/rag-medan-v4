"""Sync module for usulan_bank."""
import os
import sys
import logging
import traceback
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.utils import encode_texts

logger = logging.getLogger("rag_usulan.sync")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


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
            logger.info(f"[SYNC-USULAN] Data {action} berhasil (ID={point_id})")
            
            return {"status": "success", "message": f"Data {action} berhasil"}
        
        elif action == "delete":
            point_id = str(content["request_rag_id"])
            await qdrant.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
                wait=True
            )
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
