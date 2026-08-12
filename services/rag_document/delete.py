import os
import sys
import logging
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.lightrag_sync import fire_lightrag_delete

logger = logging.getLogger("rag_document.delete")


async def soft_delete_document(doc_id: str) -> Dict[str, Any]:
    """
    Soft-delete dokumen dari index LightRAG.
    """
    try:
        logger.info(f"[SOFT-DELETE] Soft deleted (LightRAG only) untuk doc_id={doc_id}")

        # Fire-and-forget: hapus dari LightRAG
        fire_lightrag_delete(source_type="document", source_id=doc_id)

        return {"status": "deleted", "deleted": 1}

    except Exception as e:
        logger.exception(f"[SOFT-DELETE] Error: {e}")
        return {"status": "error", "error": str(e)}


async def hard_delete_document(doc_id: str) -> Dict[str, Any]:
    """
    Hard-delete (permanen) dokumen dari index LightRAG.
    """
    try:
        logger.info(f"[HARD-DELETE] Deleted (LightRAG only) untuk doc_id={doc_id}")

        # Fire-and-forget: hapus dari LightRAG
        fire_lightrag_delete(source_type="document", source_id=doc_id)

        return {"status": "deleted", "deleted": 1}

    except Exception as e:
        logger.exception(f"[HARD-DELETE] Error: {e}")
        return {"status": "error", "error": str(e)}
