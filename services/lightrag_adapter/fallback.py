"""
RAG Medan v4 - LightRAG Adapter — Legacy Fallback Routing.

Thin wrapper yang mendelegasikan ke LegacySearchProvider ketika
LightRAG tidak tersedia atau mengembalikan error.

Digunakan oleh LightRAGSearchProvider di orchestrator/search_provider.py
saat LIGHTRAG_FALLBACK_TO_LEGACY=true.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("lightrag_adapter.fallback")


async def fallback_search(
    normalized_question: str,
    user_question: str,
    wa_number: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Route search ke legacy parallel fan-out ketika LightRAG gagal.

    Mendelegasikan ke LegacySearchProvider untuk menghindari
    duplikasi logic dengan orchestrator/search_provider.py.

    Args:
        normalized_question: Cleaned/normalized question text.
        user_question: Original user question.
        wa_number: WhatsApp number (untuk response metadata).
        top_k: Number of results per service.

    Returns:
        Dict dengan format:
        {
            "status": "fallback",
            "engine": "legacy",
            "candidates": [...],
            "services_queried": int,
            "timing": { ... }
        }
    """
    logger.info("[FALLBACK] Routing to legacy RAG via LegacySearchProvider")

    try:
        # Delegate ke LegacySearchProvider — single source of truth untuk
        # legacy search logic. Menghindari duplikasi dengan search_provider.py.
        from orchestrator.search_provider import LegacySearchProvider
        provider = LegacySearchProvider()
        result = await provider.search(
            normalized_question=normalized_question,
            user_question=user_question,
            wa_number=wa_number,
            top_k=top_k,
        )
        # Override status agar caller tahu ini adalah fallback result
        result["status"] = "fallback"
        result["fallback_reason"] = "lightrag_unavailable"
        return result

    except Exception as e:
        logger.error(f"[FALLBACK] Legacy search also failed: {e}", exc_info=True)
        return {
            "status": "error",
            "engine": "legacy_fallback",
            "error": str(e),
            "candidates": [],
            "services_queried": 0,
            "fallback_reason": "lightrag_unavailable",
            "timing": {},
        }
