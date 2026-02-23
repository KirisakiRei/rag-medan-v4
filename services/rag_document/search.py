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
from shared.utils import format_for_display, encode_texts

logger = logging.getLogger("rag_document.search")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def search_document_bank(
    query: str,
    limit: int = 5,
    request_source: str = "unknown"
) -> Dict[str, Any]:
    """Search document_bank (direct mode)."""
    logger.info(f"[API] doc-search query='{query}' limit={limit} | source={request_source}")

    try:
        query_vector, = await encode_texts([query], model=model, prefix="query: ", model_size="large")

        qdrant_hits = await qdrant.query_points(
            collection_name=config.COLLECTION_DOCUMENT,
            query=query_vector,
            limit=limit
        )

        result_points = getattr(qdrant_hits, "points", None) or getattr(qdrant_hits, "result", None) or qdrant_hits
        search_results = []

        for hit in result_points:
            point = hit[0] if isinstance(hit, tuple) else hit
            payload = getattr(point, "payload", {}) or point.get("payload", {})
            score = getattr(point, "score", 0.0)

            search_results.append({
                "doc_id": payload.get("mysql_id"),
                "opd": payload.get("opd"),
                "filename": payload.get("filename"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "section": payload.get("section"),
                "summary": payload.get("summary"),
                "text": payload.get("text"),
                "score": float(score)
            })

        if not search_results:
            logger.info(f"[API] doc-search no results for query='{query}'")
            return {"status": "empty", "results": []}

        logger.info(f"[API] doc-search results={len(search_results)} hits | top_score={search_results[0]['score']:.3f}")

        use_post_summary = config.USE_POST_SUMMARY
        post_summary_top_k = config.POST_SUMMARY_TOP_K

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

            return {
                "status": "success",
                "mode": "post-summary",
                "query": query,
                "summary": generated_summary,
                "results": top_ranked_results
            }

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
    """Search document_bank for unified mode, return top-K candidates."""
    start_time = time.time()
    
    logger.info(f"[DOC-SEARCH] Question: {question[:50]}...")

    try:
        embedding_start = time.time()
        query_vector, = await encode_texts([question], model=model, prefix="query: ", model_size="large")
        embedding_duration = time.time() - embedding_start

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

        scored_results = []
        
        for hit in result_points:
            result_item = hit[0] if isinstance(hit, tuple) else hit
            payload = getattr(result_item, "payload", {}) or result_item.get("payload", {})
            score = float(getattr(result_item, "score", 0.0))
            document_text = payload.get("text", "")
            
            if score >= 0.4 and document_text:
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
                    "content_for_check": document_text[:2000],
                    "document_info": {
                        "filename": payload.get("filename", "-"),
                        "page_number": payload.get("page_number", "-"),
                        "opd": payload.get("opd", "-"),
                        "doc_id": payload.get("mysql_id", "-")
                    }
                })
        
        scored_results = sorted(scored_results, key=lambda x: x["final_score"], reverse=True)
        
        top_results = scored_results[:top_k]
        
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
