"""
RAG Document Service - Search Module
Logic pencarian di document_bank
PAYLOAD HARUS PERSIS SEPERTI V2!
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.summarizer_utils import summarize_text

logger = logging.getLogger("rag_document.search")

# Global instances (diinisialisasi dari main.py)
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances dari main.py"""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


def embed_query(model_doc: SentenceTransformer, text: str) -> List[float]:
    """Generate embedding untuk query."""
    return model_doc.encode(text, convert_to_numpy=True).tolist()


async def search_document_bank(
    query: str,
    limit: int = 5,
    request_source: str = "unknown"
) -> Dict[str, Any]:
    """
    Search di document_bank.
    PAYLOAD PERSIS SEPERTI V2!
    
    Returns:
        Dict dengan format v2-compatible response
    """
    logger.info(f"[API] doc-search query='{query}' limit={limit} | source={request_source}")

    try:
        # Embed query langsung
        query_vector = embed_query(model, query)

        # Query ke Qdrant
        qdrant_hits = await qdrant.query_points(
            collection_name=config.COLLECTION_DOCUMENT,
            query=query_vector,
            limit=limit
        )

        result_points = getattr(qdrant_hits, "points", None) or getattr(qdrant_hits, "result", None) or qdrant_hits
        search_results = []

        for hit in result_points:
            result_item = hit[0] if isinstance(hit, tuple) else hit
            result_payload = getattr(result_item, "payload", {}) or result_item.get("payload", {})
            result_score = getattr(result_item, "score", 0.0)

            # PAYLOAD RESULT PERSIS V2:
            search_results.append({
                "doc_id": result_payload.get("mysql_id"),
                "opd": result_payload.get("opd"),
                "filename": result_payload.get("filename"),
                "page_number": result_payload.get("page_number"),
                "chunk_index": result_payload.get("chunk_index"),
                "section": result_payload.get("section"),
                "summary": result_payload.get("summary"),
                "text": result_payload.get("text"),
                "score": float(result_score)
            })

        # RESPONSE PERSIS V2:
        if not search_results:
            logger.info(f"[API] doc-search no results for query='{query}'")
            return {"status": "empty", "results": []}

        logger.info(f"[API] doc-search results={len(search_results)} hits | top_score={search_results[0]['score']:.3f}")

        # Cek config untuk post-summary
        use_post_summary = config.RAG_USE_POST_SUMMARY
        post_summary_top_k = config.RAG_POST_SUMMARY_TOP_K

        if use_post_summary:
            logger.info(f"[POST-SUM] Aktif → meringkas top {post_summary_top_k} hasil ...")
            top_ranked_results = sorted(search_results, key=lambda x: -x["score"])[:post_summary_top_k]
            combined_document_text = "\n\n".join(
                [result["text"] or "" for result in top_ranked_results if result.get("text")]
            )

            try:
                generated_summary = summarize_text(
                    f"Berdasarkan potongan dokumen berikut, jawab pertanyaan pengguna dengan ringkas dan informatif:\n\n{combined_document_text}",
                    max_sentences=5
                )
            except Exception as e:
                logger.warning(f"[POST-SUM] Gagal meringkas hasil: {e}")
                generated_summary = "Tidak dapat membuat ringkasan hasil."

            # RESPONSE DENGAN POST-SUMMARY PERSIS V2:
            return {
                "status": "success",
                "mode": "post-summary",
                "query": query,
                "summary": generated_summary,
                "results": top_ranked_results
            }

        # RESPONSE DIRECT MODE PERSIS V2:
        return {
            "status": "success",
            "mode": "direct",
            "query": query,
            "results": search_results
        }

    except Exception as e:
        logger.exception("doc-search error")
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }
