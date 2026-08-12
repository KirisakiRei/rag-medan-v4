"""
RAG Medan v4 - LightRAG Adapter — Search Logic.

Bertanggung jawab:
- Memanggil LightRAG query API
- Mentransformasi raw response ke canonical format
- Membangun timing metrics
- Menangani empty/error results

Alur:
  Query text
      ↓
  LightRAG Server (POST /query)
      ↓
  Raw contexts + references
      ↓
  map_lightrag_context_to_canonical()
      ↓
  build_references()
      ↓
  Canonical SearchResponse
"""
import time
import logging
from typing import Dict, Any

from services.lightrag_adapter.client import lightrag_client
from services.lightrag_adapter.config import adapter_config
from services.lightrag_adapter.references import (
    map_lightrag_context_to_canonical,
    build_references,
)
from services.lightrag_adapter.errors import (
    LightRAGConnectionError,
    LightRAGSearchError,
)

logger = logging.getLogger("lightrag_adapter.search")


async def search(
    query: str,
    mode: str = None,
    top_k: int = None,
) -> Dict[str, Any]:
    """
    Execute unified search via LightRAG.

    Args:
        query: Normalized question text.
        mode: Query mode override (naive|local|global|hybrid|mix).
              None = gunakan adapter_config.QUERY_MODE.
        top_k: Number of results. None = gunakan adapter_config.TOP_K.

    Returns:
        Canonical response dict:
        {
            "status": "success" | "no_results" | "error",
            "engine": "lightrag",
            "query": "...",
            "contexts": [...],
            "references": [...],
            "timing": { "retrieval_sec": ..., "rerank_sec": ..., "total_sec": ... }
        }
    """
    search_start = time.time()
    # Gunakan None-check eksplisit agar top_k=0 tidak salah fallback ke default
    mode = mode if mode is not None else adapter_config.QUERY_MODE
    top_k = top_k if top_k is not None else adapter_config.TOP_K

    logger.info(f"[LR-SEARCH] query='{query[:80]}' mode={mode} top_k={top_k}")

    try:
        # Call LightRAG Server
        retrieval_start = time.time()
        lightrag_result = await lightrag_client.query(
            query_text=query,
            mode=mode,
            top_k=top_k,
            include_references=True,
        )
        retrieval_sec = time.time() - retrieval_start

        # Parse raw contexts dari LightRAG response.
        # LightRAG bisa return context dalam berbagai format tergantung versi:
        # - "contexts" list
        # - "retrieved_contexts" list
        # - "response" string (single generated answer)
        raw_contexts = (
            lightrag_result.get("contexts")
            or lightrag_result.get("retrieved_contexts")
            or []
        )

        # Fallback: jika tidak ada structured contexts, coba extract dari response
        if not raw_contexts and isinstance(lightrag_result.get("response"), str):
            raw_contexts = [{
                "content": lightrag_result["response"],
                "doc_id": "unknown",
            }]

        # Map ke canonical format
        canonical_contexts = map_lightrag_context_to_canonical(raw_contexts)
        references = build_references(canonical_contexts) if canonical_contexts else []

        total_sec = time.time() - search_start

        if not canonical_contexts:
            logger.info("[LR-SEARCH] No results found")
            return {
                "status": "no_results",
                "engine": "lightrag",
                "query": query,
                "contexts": [],
                "references": [],
                "timing": {
                    "retrieval_sec": round(retrieval_sec, 3),
                    "rerank_sec": 0.0,
                    "total_sec": round(total_sec, 3),
                },
            }

        logger.info(
            f"[LR-SEARCH] Found {len(canonical_contexts)} contexts, "
            f"{len(references)} unique sources in {total_sec:.2f}s"
        )

        return {
            "status": "success",
            "engine": "lightrag",
            "query": query,
            "contexts": canonical_contexts,
            "references": references,
            "timing": {
                "retrieval_sec": round(retrieval_sec, 3),
                "rerank_sec": 0.0,    # Diisi saat reranker diaktifkan
                "total_sec": round(total_sec, 3),
            },
        }

    except LightRAGConnectionError as e:
        total_sec = time.time() - search_start
        logger.error(f"[LR-SEARCH] Connection error: {e}")
        return {
            "status": "error",
            "engine": "lightrag",
            "query": query,
            "error": str(e),
            "contexts": [],
            "references": [],
            "timing": {"total_sec": round(total_sec, 3)},
        }

    except LightRAGSearchError as e:
        total_sec = time.time() - search_start
        logger.error(f"[LR-SEARCH] Search error: {e}")
        return {
            "status": "error",
            "engine": "lightrag",
            "query": query,
            "error": str(e),
            "contexts": [],
            "references": [],
            "timing": {"total_sec": round(total_sec, 3)},
        }

    except Exception as e:
        total_sec = time.time() - search_start
        logger.error(f"[LR-SEARCH] Unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "engine": "lightrag",
            "query": query,
            "error": str(e),
            "contexts": [],
            "references": [],
            "timing": {"total_sec": round(total_sec, 3)},
        }
