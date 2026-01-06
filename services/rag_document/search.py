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
from shared.utils import format_for_display
# REMOVED: ai_check_relevance - post-filter sekarang di orchestrator

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


async def search_document_unified(
    question: str,
    original_question: str,
    wa_number: str = "unknown",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Search di document_bank untuk unified/parallel mode.
    
    V3 CHANGES:
    - TIDAK ada AI relevance check di sini (pindah ke orchestrator)
    - Return TOP 3 scored results untuk orchestrator aggregate
    - Threshold lebih rendah (0.4) karena AI check di orchestrator
    
    Args:
        question: Clean question dari orchestrator
        original_question: Pertanyaan asli user
        wa_number: Nomor WhatsApp
        top_k: Jumlah top results untuk return
    """
    start_time = time.time()
    
    logger.info(f"[DOC-SEARCH] Question: {question[:50]}...")

    try:
        # 1. Embed query
        embedding_start = time.time()
        query_vector = embed_query(model, question)
        embedding_duration = time.time() - embedding_start

        # 2. Query ke Qdrant dengan filter is_deleted=False
        qdrant_start = time.time()
        qdrant_hits = await qdrant.query_points(
            collection_name=config.COLLECTION_DOCUMENT,
            query=query_vector,
            query_filter=qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(key="is_deleted", match=qdrant_models.MatchValue(value=False))]
            ),
            limit=top_k * 2  # Fetch lebih banyak untuk filtering
        )
        qdrant_duration = time.time() - qdrant_start

        result_points = getattr(qdrant_hits, "points", None) or getattr(qdrant_hits, "result", None) or qdrant_hits

        # Tidak ada hasil
        if not result_points:
            total_duration = time.time() - start_time
            logger.info("[DOC-SEARCH] No document results")
            return {
                "status": "empty",
                "message": "No results from document_bank",
                "source": "document",
                "data": {
                    "results": [],
                    "total_found": 0,
                    "metadata": {
                        "wa_number": wa_number,
                        "original_question": original_question,
                        "final_question": question,
                        "category": "Dokumen"
                    }
                },
                "timing": {
                    "embedding_sec": round(embedding_duration, 3),
                    "qdrant_sec": round(qdrant_duration, 3),
                    "total_sec": round(total_duration, 3)
                }
            }

        # 3. Process results - NO AI CHECK, just scoring
        scored_results = []
        
        for hit in result_points:
            result_item = hit[0] if isinstance(hit, tuple) else hit
            payload = getattr(result_item, "payload", {}) or result_item.get("payload", {})
            score = float(getattr(result_item, "score", 0.0))
            document_text = payload.get("text", "")
            
            # Threshold check (lebih rendah, orchestrator yang final decide)
            if score >= 0.4 and document_text:
                # Determine acceptance note
                if score >= 0.85:
                    note = "high_score"
                elif score >= 0.70:
                    note = "good_score"
                else:
                    note = "marginal"
                
                scored_results.append({
                    "source": "document",
                    "question": "-",
                    "question_rag_name": "-",
                    "answer_id": None,
                    "answer_doc": format_for_display(document_text),
                    "category_id": None,
                    "dense_score": round(score, 3),
                    "overlap_score": 0.0,
                    "final_score": round(score, 3),
                    "note": note,
                    # Content untuk AI relevance check di orchestrator
                    "content_for_check": document_text[:2000],  # Limit untuk AI check
                    # Document metadata
                    "document_info": {
                        "filename": payload.get("filename", "-"),
                        "page_number": payload.get("page_number", "-"),
                        "opd": payload.get("opd", "-"),
                        "doc_id": payload.get("mysql_id", "-")
                    }
                })
        
        # Sort by score descending
        scored_results = sorted(scored_results, key=lambda x: x["final_score"], reverse=True)
        
        # Take top K
        top_results = scored_results[:top_k]
        
        # Log results
        logger.info(f"[DOC-SEARCH] Found {len(scored_results)} results, returning top {len(top_results)}")
        for i, r in enumerate(top_results):
            logger.info(f"  [{i+1}] {r['document_info']['filename']} p.{r['document_info']['page_number']} | score={r['final_score']:.3f}")

        total_duration = time.time() - start_time

        if top_results:
            return {
                "status": "has_candidates",
                "message": f"Found {len(top_results)} document candidates",
                "source": "document",
                "data": {
                    "results": top_results,
                    "count": len(top_results),
                    "metadata": {
                        "wa_number": wa_number,
                        "original_question": original_question,
                        "final_question": question,
                        "category": "Dokumen"
                    }
                },
                "timing": {
                    "embedding_sec": round(embedding_duration, 3),
                    "qdrant_sec": round(qdrant_duration, 3),
                    "total_sec": round(total_duration, 3)
                }
            }
        else:
            return {
                "status": "no_results",
                "message": "No document results found above threshold",
                "source": "document",
                "data": {
                    "results": [],
                    "count": 0,
                    "metadata": {
                        "wa_number": wa_number,
                        "original_question": original_question,
                        "final_question": question,
                        "category": "Dokumen"
                    }
                },
                "timing": {
                    "embedding_sec": round(embedding_duration, 3),
                    "qdrant_sec": round(qdrant_duration, 3),
                    "total_sec": round(total_duration, 3)
                }
            }

    except Exception as e:
        logger.exception(f"[DOC-SEARCH] Error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "source": "document",
            "data": {
                "results": [],
                "count": 0,
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": original_question,
                    "final_question": question,
                    "category": "Dokumen"
                }
            },
            "timing": {"total_sec": round(time.time() - start_time, 3)}
        }
