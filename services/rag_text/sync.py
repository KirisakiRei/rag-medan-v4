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
from shared.lightrag_sync import fire_lightrag_sync_text, fire_lightrag_delete

logger = logging.getLogger("rag_text.sync")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def sync_data(action: str, content: Any) -> Dict[str, Any]:
    try:
        if action == "bulk_sync":
            if not isinstance(content, list):
                return {
                    "status": "error",
                    "error": {"type": "ValidationError", "message": "Content harus berupa list"}
                }
            
            vectors = await encode_texts([item["question_rag_name"] for item in content], model=model, prefix="passage: ")
            points = []
            for item, vector in zip(content, vectors):
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

            # Fire-and-forget: sync ke LightRAG
            for item in content:
                fire_lightrag_sync_text(
                    source_id=str(item["question_rag_id"]),
                    title=item.get("question", ""),
                    content="",
                    content_hash="",
                    is_active=True,
                    category=item.get("category_id"),
                    question=item.get("question"),
                    answer=None,
                )

            return {
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil",
                "total_synced": len(points)
            }
        
        elif action == "add":
            point_id = str(content["question_rag_id"])
            [vector] = await encode_texts([content["question_rag_name"]], model=model, prefix="passage: ")
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

            # Fire-and-forget: sync ke LightRAG
            fire_lightrag_sync_text(
                source_id=point_id,
                title=content.get("question", ""),
                content="",
                content_hash="",
                is_active=True,
                category=content.get("category_id"),
                question=content.get("question"),
                answer=None,
            )

            return {"status": "success", "message": "Data berhasil ditambahkan", "id": point_id}
        
        elif action == "update":
            point_id = str(content["question_rag_id"])
            [vector] = await encode_texts([content["question_rag_name"]], model=model, prefix="passage: ")
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

            # Fire-and-forget: sync ke LightRAG
            fire_lightrag_sync_text(
                source_id=point_id,
                title=content.get("question", ""),
                content="",
                content_hash="",
                is_active=True,
                category=content.get("category_id"),
                question=content.get("question"),
                answer=None,
            )

            return {"status": "success", "message": "Data berhasil diperbarui"}
        
        elif action == "delete":
            point_id = str(content["question_rag_id"])
            await qdrant.delete(
                collection_name=config.COLLECTION_TEXT,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"[SYNC-DATA] Data dihapus : ID={point_id}")

            # Fire-and-forget: hapus dari LightRAG
            fire_lightrag_delete(source_type="text", source_id=point_id)

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
