import os
import sys
import logging
from typing import Dict, Any
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.lightrag_sync import fire_lightrag_delete

logger = logging.getLogger("rag_document.delete")

qdrant: AsyncQdrantClient = None


def set_instances(qdrant_client: AsyncQdrantClient):
    """Set global instance."""
    global qdrant
    qdrant = qdrant_client


async def soft_delete_document(doc_id: str) -> Dict[str, Any]:
    """
    Soft-delete semua chunk milik doc_id.
    Menggunakan pagination loop — menangani dokumen dengan >100 chunk.
    """
    try:
        all_point_ids = []
        offset = None

        while True:
            results = await qdrant.scroll(
                collection_name=config.COLLECTION_DOCUMENT,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="mysql_id",
                            match=qdrant_models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True
            )
            batch = results[0] if results else []
            next_offset = results[1] if results and len(results) > 1 else None
            all_point_ids.extend([p.id for p in batch])
            if next_offset is None:
                break
            offset = next_offset

        if not all_point_ids:
            return {"status": "not_found", "deleted": 0}

        await qdrant.set_payload(
            collection_name=config.COLLECTION_DOCUMENT,
            payload={
                "is_active": False,
                "is_deleted": True,
                "deleted_at": datetime.utcnow().isoformat()
            },
            points=all_point_ids
        )

        logger.info(f"[SOFT-DELETE] Soft deleted {len(all_point_ids)} chunks untuk doc_id={doc_id}")

        # Fire-and-forget: hapus dari LightRAG
        fire_lightrag_delete(source_type="document", source_id=doc_id)

        return {"status": "deleted", "deleted": len(all_point_ids)}

    except Exception as e:
        logger.exception(f"[SOFT-DELETE] Error: {e}")
        return {"status": "error", "error": str(e)}


async def hard_delete_document(doc_id: str) -> Dict[str, Any]:
    """
    Hard-delete (permanen) semua chunk milik doc_id.
    Menggunakan pagination loop — menangani dokumen dengan >100 chunk.
    """
    try:
        all_point_ids = []
        offset = None

        while True:
            results = await qdrant.scroll(
                collection_name=config.COLLECTION_DOCUMENT,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="mysql_id",
                            match=qdrant_models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            batch = results[0] if results else []
            next_offset = results[1] if results and len(results) > 1 else None
            all_point_ids.extend([p.id for p in batch])
            if next_offset is None:
                break
            offset = next_offset

        if not all_point_ids:
            return {"status": "not_found", "deleted": 0}

        await qdrant.delete(
            collection_name=config.COLLECTION_DOCUMENT,
            points_selector=qdrant_models.PointIdsList(points=all_point_ids)
        )

        logger.info(f"[HARD-DELETE] Deleted {len(all_point_ids)} chunks untuk doc_id={doc_id}")

        # Fire-and-forget: hapus dari LightRAG
        fire_lightrag_delete(source_type="document", source_id=doc_id)

        return {"status": "deleted", "deleted": len(all_point_ids)}

    except Exception as e:
        logger.exception(f"[HARD-DELETE] Error: {e}")
        return {"status": "error", "error": str(e)}


async def reactivate_document(doc_id: str) -> Dict[str, Any]:
    """
    Pulihkan semua chunk soft-deleted milik doc_id (is_deleted=False).
    Menggunakan pagination loop — menangani dokumen dengan >100 chunk.
    """
    try:
        all_point_ids = []
        offset = None

        while True:
            results = await qdrant.scroll(
                collection_name=config.COLLECTION_DOCUMENT,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="mysql_id",
                            match=qdrant_models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False
            )
            batch = results[0] if results else []
            next_offset = results[1] if results and len(results) > 1 else None
            all_point_ids.extend([p.id for p in batch])
            if next_offset is None:
                break
            offset = next_offset

        if not all_point_ids:
            return {"status": "not_found", "reactivated": 0}

        await qdrant.set_payload(
            collection_name=config.COLLECTION_DOCUMENT,
            payload={
                "is_active": True,
                "is_deleted": False,
                "deleted_at": None,
                "reactivated_at": datetime.utcnow().isoformat()
            },
            points=all_point_ids
        )

        logger.info(f"[REACTIVATE] Dipulihkan {len(all_point_ids)} chunks untuk doc_id={doc_id}")
        return {"status": "reactivated", "reactivated": len(all_point_ids)}

    except Exception as e:
        logger.exception(f"[REACTIVATE] Error: {e}")
        return {"status": "error", "error": str(e)}
