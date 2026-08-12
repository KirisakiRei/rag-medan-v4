import os
import sys
import logging
import traceback
import hashlib
import asyncio
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.lightrag_sync import sync_lightrag_text, delete_lightrag_source

logger = logging.getLogger("rag_text.sync")


async def _sync_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Index one FAQ question and wait until LightRAG confirms PROCESSED."""
    source_id = str(item["question_rag_id"])
    question = str(item.get("question") or "").strip()
    if not question:
        raise ValueError(f"question kosong untuk question_rag_id={source_id}")
    category = item.get("category_id")
    content_hash = hashlib.sha256(
        f"{question}\n{category or ''}".encode("utf-8")
    ).hexdigest()
    return await sync_lightrag_text(
        source_id=source_id,
        title=question,
        content="",
        content_hash=content_hash,
        is_active=True,
        category=category,
        question=question,
        answer=None,
    )


async def sync_data(action: str, content: Any) -> Dict[str, Any]:
    try:
        if action == "bulk_sync":
            if not isinstance(content, list):
                return {
                    "status": "error",
                    "error": {"type": "ValidationError", "message": "Content harus berupa list"}
                }
            
            semaphore = asyncio.Semaphore(5)

            async def sync_bounded(item):
                async with semaphore:
                    return await _sync_item(item)

            results = await asyncio.gather(*(sync_bounded(item) for item in content))

            return {
                "status": "success",
                "message": f"Sinkronisasi {len(content)} data berhasil",
                "total_synced": len(content)
            }
        
        elif action == "add":
            point_id = str(content["question_rag_id"])
            await _sync_item(content)
            logger.info(f"[SYNC-DATA] Data berhasil ditambahkan ke LightRAG: ID={point_id}")

            return {"status": "success", "message": "Data berhasil ditambahkan", "id": point_id}
        
        elif action == "update":
            point_id = str(content["question_rag_id"])
            await _sync_item(content)
            logger.info(f"[SYNC-DATA] Data berhasil diperbarui di LightRAG: ID={point_id}")

            return {"status": "success", "message": "Data berhasil diperbarui"}
        
        elif action == "delete":
            point_id = str(content["question_rag_id"])
            await delete_lightrag_source(source_type="text", source_id=point_id)
            logger.info(f"[SYNC-DATA] Data berhasil dihapus dari LightRAG: ID={point_id}")

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
