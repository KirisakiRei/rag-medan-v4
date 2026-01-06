"""
RAG Usulan Service - Sync Module
Logic sinkronisasi usulan_bank.
"""
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

logger = logging.getLogger("rag_usulan.sync")

# Global instances (diinisialisasi dari main.py)
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances dari main.py"""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def sync_usulan(action: str, content: Any) -> Dict[str, Any]:
    """
    Sinkronisasi data usulan_bank.
    PAYLOAD DAN RESPONSE PERSIS SEPERTI V2!
    
    Args:
        action: "bulk_sync", "add", "update", "delete"
        content: data sesuai action
        
    Returns:
        Dict response sesuai V2
    """
    collection = config.COLLECTION_USULAN
    
    try:
        # =====================================================
        # BULK SYNC (PERSIS V2)
        # =====================================================
        if action == "bulk_sync":
            if not isinstance(content, list):
                return {
                    "status": "error",
                    "error": {"type": "ValidationError", "message": "Content harus berupa list"}
                }
            
            points = []
            for item in content:
                # PERSIS V2: encoding dengan prefix "passage:"
                vector = model.encode("passage: " + item["request_rag_name"]).tolist()
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
            
            # Create text index (PERSIS V2)
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
            
            # RESPONSE PERSIS V2:
            return {
                "status": "success",
                "message": f"{len(points)} data berhasil disinkronkan ke {collection}"
            }
        
        # =====================================================
        # ADD / UPDATE (PERSIS V2)
        # =====================================================
        elif action in ["add", "update"]:
            point_id = str(content["request_rag_id"])
            vector = model.encode("passage: " + content["request_rag_name"]).tolist()
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
            
            # RESPONSE PERSIS V2:
            return {"status": "success", "message": f"Data {action} berhasil"}
        
        # =====================================================
        # DELETE (PERSIS V2)
        # =====================================================
        elif action == "delete":
            point_id = str(content["request_rag_id"])
            await qdrant.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"[SYNC-USULAN] Data dihapus (ID={point_id})")
            
            # RESPONSE PERSIS V2:
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
