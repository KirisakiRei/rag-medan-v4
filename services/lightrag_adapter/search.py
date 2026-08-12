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
from services.lightrag_adapter.stats import stats

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
        # - "references" list (saat response_type=default / answer generation)
        # - "response" string (single generated answer)
        raw_contexts = (
            lightrag_result.get("contexts")
            or lightrag_result.get("retrieved_contexts")
            or []
        )

        # Fallback 1: extract contexts dari references (berisi file_source
        # yang kita set saat ingest: "web:<id>", "document:<id>", "text:<id>").
        if not raw_contexts:
            references = (
                lightrag_result.get("references")
                or lightrag_result.get("retrieved_contexts")
                or []
            )
            for ref in references:
                doc_descriptor = (
                    ref.get("file_source")
                    or ref.get("file_path")
                    or ref.get("source")
                    or ref.get("document_id")
                    or ""
                )
                raw_contexts.append({
                    "content": ref.get("content") or ref.get("text") or "",
                    "doc_id": doc_descriptor,
                    "title": ref.get("source") or ref.get("title") or "",
                    "score": ref.get("score"),
                })

        # Fallback 2: jika tetap tidak ada structured contexts,
        # gunakan generated response sebagai single context.
        if not raw_contexts and isinstance(lightrag_result.get("response"), str):
            raw_contexts = [{
                "content": lightrag_result["response"],
                "doc_id": "",
                "title": "",
                "score": None,
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
        stats.record_query(total_sec)

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
        stats.record_query_error()
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
        stats.record_query_error()
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
        stats.record_query_error()
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
