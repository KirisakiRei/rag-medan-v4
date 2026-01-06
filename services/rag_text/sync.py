"""
RAG Text Service - Sync Module
Logic sinkronisasi knowledge_bank.
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

logger = logging.getLogger("rag_text.sync")

# Global instances (diinisialisasi dari main.py)
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances dari main.py"""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def sync_data(action: str, content: Any) -> Dict[str, Any]:
    """
    Sinkronisasi data knowledge_bank.
    PAYLOAD DAN RESPONSE PERSIS SEPERTI V2!
    
    Args:
        action: "bulk_sync", "add", "update", "delete"
        content: data sesuai action
        
    Returns:
        Dict response sesuai V2
    """
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
                vector = model.encode("passage: " + item["question_rag_name"]).tolist()
                point_id = str(item["question_rag_id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": item["question_id"],
                        "answer_id": item["answer_id"],
                        "category_id": item["category_id"],
                        "question": item["question"],
                        "question_rag_name": item["question_rag_name"]
                    }
                })
            
            await qdrant.upsert(collection_name=config.COLLECTION_TEXT, points=points)
            
            # Create text index (PERSIS V2)
            await qdrant.create_payload_index(
                collection_name=config.COLLECTION_TEXT,
                field_name="question_rag_name",
                field_schema=qdrant_models.TextIndexParams(
                    type="text",
                    tokenizer=qdrant_models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )
            
            logger.info(f"[SYNC-DATA] Sinkronisasi {len(points)} data ke Knowledge Bank berhasil")
            
            # RESPONSE PERSIS V2:
            return {
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil",
                "total_synced": len(points)
            }
        
        # =====================================================
        # ADD (PERSIS V2)
        # =====================================================
        elif action == "add":
            point_id = str(content["question_rag_id"])
            vector = model.encode("passage: " + content["question_rag_name"]).tolist()
            await qdrant.upsert(
                collection_name=config.COLLECTION_TEXT,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": content["question_id"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id"),
                        "question": content["question"],
                        "question_rag_name": content["question_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-DATA] Data berhasil ditambahkan ke Knowledge Bank: ID={point_id}")
            
            # RESPONSE PERSIS V2:
            return {"status": "success", "message": "Data berhasil ditambahkan", "id": point_id}
        
        # =====================================================
        # UPDATE (PERSIS V2)
        # =====================================================
        elif action == "update":
            point_id = str(content["question_rag_id"])
            vector = model.encode("passage: " + content["question_rag_name"]).tolist()
            await qdrant.upsert(
                collection_name=config.COLLECTION_TEXT,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": content["question_id"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id"),
                        "question": content["question"],
                        "question_rag_name": content["question_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-DATA] Data berhasil Diperbarui di Knowledge Bank: ID={point_id}")
            
            # RESPONSE PERSIS V2:
            return {"status": "success", "message": "Data berhasil diperbarui"}
        
        # =====================================================
        # DELETE (PERSIS V2)
        # =====================================================
        elif action == "delete":
            point_id = str(content["question_rag_id"])
            await qdrant.delete(
                collection_name=config.COLLECTION_TEXT,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"[SYNC-DATA] Data dihapus : ID={point_id}")
            
            # RESPONSE PERSIS V2:
            return {"status": "success", "message": "Data berhasil dihapus"}
        
        else:
            return {
                "status": "error",
                "error": {"type": "ValidationError", "message": f"Action '{action}' tidak dikenali"}
            }

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"[ERROR][sync_data] {str(e)}\n{error_traceback}")
        return {
            "status": "error",
            "error": {"type": "ServerError", "message": "Kesalahan internal saat sinkronisasi", "detail": str(e)}
        }
