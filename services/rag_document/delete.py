import os
import sys
import logging
from typing import Dict, Any
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

logger = logging.getLogger("rag_document.delete")

qdrant: AsyncQdrantClient = None


def set_instances(qdrant_client: AsyncQdrantClient):
    """Set global instance."""
    global qdrant
    qdrant = qdrant_client


async def soft_delete_document(doc_id: str) -> Dict[str, Any]:

    try:
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
            with_payload=True
        )
        
        points = results[0] if results else []
        
        if not points:
            return {"status": "not_found", "deleted": 0}
        
        point_ids = [p.id for p in points]
        
        await qdrant.set_payload(
            collection_name=config.COLLECTION_DOCUMENT,
            payload={
                "is_deleted": True,
                "deleted_at": datetime.utcnow().isoformat()
            },
            points=point_ids
        )
        
        logger.info(f"[SOFT-DELETE] Soft deleted {len(point_ids)} chunks for doc_id={doc_id}")
        
        return {"status": "deleted", "deleted": len(point_ids)}
        
    except Exception as e:
        logger.exception(f"[SOFT-DELETE] Error: {e}")
        return {"status": "error", "error": str(e)}


async def hard_delete_document(doc_id: str) -> Dict[str, Any]:
    try:
        # Get all chunks with mysql_id
        results = await qdrant.scroll(
            collection_name=config.COLLECTION_DOCUMENT,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="mysql_id",  # PERSIS V2: mysql_id bukan doc_id
                        match=qdrant_models.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=100,
            with_payload=False,
            with_vectors=False
        )
        
        points = results[0] if results else []
        
        if not points:
            return {"status": "not_found", "deleted": 0}
        
        point_ids = [p.id for p in points]
        
        # Delete points
        await qdrant.delete(
            collection_name=config.COLLECTION_DOCUMENT,
            points_selector=qdrant_models.PointIdsList(points=point_ids)
        )
        
        logger.info(f"[HARD-DELETE] Deleted {len(point_ids)} chunks for doc_id={doc_id}")
        
        return {"status": "deleted", "deleted": len(point_ids)}
        
    except Exception as e:
        logger.exception(f"[HARD-DELETE] Error: {e}")
        return {"status": "error", "error": str(e)}
