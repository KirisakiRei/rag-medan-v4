import os
import sys
import logging
import traceback
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.lightrag_sync import fire_lightrag_sync_text, fire_lightrag_delete

logger = logging.getLogger("rag_text.sync")


async def sync_data(action: str, content: Any) -> Dict[str, Any]:
    try:
        if action == "bulk_sync":
            if not isinstance(content, list):
                return {
                    "status": "error",
                    "error": {"type": "ValidationError", "message": "Content harus berupa list"}
                }
            
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
                "message": f"Sinkronisasi {len(content)} data berhasil",
                "total_synced": len(content)
            }
        
        elif action == "add":
            point_id = str(content["question_rag_id"])
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
