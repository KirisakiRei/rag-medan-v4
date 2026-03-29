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


def _extract_query_points(qdrant_hits: Any) -> List[Any]:
    """Normalize Qdrant query_points response across client versions."""
    if qdrant_hits is None:
        return []

    points = getattr(qdrant_hits, "points", None)
    if points is not None:
        return list(points)

    result = getattr(qdrant_hits, "result", None)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        nested_points = result.get("points")
        if isinstance(nested_points, list):
            return nested_points
        return []

    if isinstance(qdrant_hits, dict):
        direct_points = qdrant_hits.get("points")
        if isinstance(direct_points, list):
            return direct_points
        nested_result = qdrant_hits.get("result")
        if isinstance(nested_result, list):
            return nested_result
        if isinstance(nested_result, dict):
            nested_points = nested_result.get("points")
            if isinstance(nested_points, list):
                return nested_points
        return []

    if isinstance(qdrant_hits, list):
        return qdrant_hits

    return []


def _extract_payload_and_score(point: Any) -> tuple[Dict[str, Any], float]:
    """Normalize point payload/score across object and dict responses."""
    if isinstance(point, tuple) and point:
        point = point[0]

    if isinstance(point, dict):
        payload = point.get("payload") or {}
        score = point.get("score", 0.0)
        return dict(payload), float(score or 0.0)

    payload = getattr(point, "payload", {}) or {}
    score = getattr(point, "score", 0.0)
    return dict(payload), float(score or 0.0)


async def _retrieve_payloads_by_ids(point_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    valid_ids = [point_id for point_id in point_ids if point_id]
    if not valid_ids:
        return {}
    points = await qdrant.retrieve(
        collection_name=config.COLLECTION_DOCUMENT,
        ids=valid_ids,
        with_payload=True,
        with_vectors=False,
    )
    return {str(point.id): dict(point.payload or {}) for point in points}


async def _expand_document_context(payload: Dict[str, Any]) -> str:
    """Expand context around a child chunk using parent or adjacent siblings."""
    text = payload.get("text", "") or ""
    if not config.RETRIEVAL_CONTEXT_EXPANSION:
        return format_for_display(text)

    point_ids = [
        payload.get("parent_chunk_id"),
        payload.get("window_prev_id"),
        payload.get("window_next_id"),
    ]
    related_payloads = await _retrieve_payloads_by_ids(point_ids)
    parent_payload = related_payloads.get(str(payload.get("parent_chunk_id")))

    if parent_payload and len(text) < 450:
        return format_for_display(parent_payload.get("text", text))

    parts = []
    prev_payload = related_payloads.get(str(payload.get("window_prev_id")))
    next_payload = related_payloads.get(str(payload.get("window_next_id")))
    if prev_payload:
        parts.append(prev_payload.get("text", ""))
    parts.append(text)
    if next_payload:
        parts.append(next_payload.get("text", ""))

    expanded = "\n\n".join(part for part in parts if part and part.strip())
    return format_for_display(expanded or text)


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
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(key="is_active", match=qdrant_models.MatchValue(value=True)),
                    qdrant_models.FieldCondition(key="is_deleted", match=qdrant_models.MatchValue(value=False)),
                    qdrant_models.FieldCondition(key="chunk_level", match=qdrant_models.MatchValue(value="child")),
                ]
            ),
            limit=limit
        )

        result_points = _extract_query_points(qdrant_hits)
        search_results = []

        for hit in result_points:
            payload, score = _extract_payload_and_score(hit)
            if not payload:
                continue

            expanded_text = await _expand_document_context(payload)
            search_results.append({
                "doc_id": payload.get("mysql_id"),
                "opd": payload.get("opd") or payload.get("organization_id"),
                "filename": payload.get("filename"),
                "page_number": payload.get("page_start") or payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "section": payload.get("section_title") or payload.get("section"),
                "heading_path": payload.get("heading_path"),
                "chunk_level": payload.get("chunk_level"),
                "summary": payload.get("summary"),
                "text": expanded_text,
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
                must=[
                    qdrant_models.FieldCondition(key="is_active", match=qdrant_models.MatchValue(value=True)),
                    qdrant_models.FieldCondition(key="is_deleted", match=qdrant_models.MatchValue(value=False)),
                    qdrant_models.FieldCondition(key="chunk_level", match=qdrant_models.MatchValue(value="child")),
                ]
            ),
            limit=top_k * 2  # Fetch lebih banyak untuk filtering
        )
        qdrant_duration = time.time() - qdrant_start

        result_points = _extract_query_points(qdrant_hits)

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
            payload, score = _extract_payload_and_score(hit)
            if not payload:
                continue
            document_text = await _expand_document_context(payload)
            
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
                        "page_number": payload.get("page_start", payload.get("page_number", "-")),
                        "opd": payload.get("opd", payload.get("organization_id", "-")),
                        "doc_id": payload.get("mysql_id", "-"),
                        "section_title": payload.get("section_title", payload.get("section", "-")),
                        "heading_path": payload.get("heading_path", "-"),
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
