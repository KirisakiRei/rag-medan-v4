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
from shared.utils import (
    normalize_text,
    clean_location_terms,
    keyword_overlap,
    detect_category,
    safe_parse_answer_id,
    encode_texts,
)
from shared.filtering import ai_pre_filter

logger = logging.getLogger("rag_text.search")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def search_knowledge_bank(
    question: str,
    wa_number: str = "unknown",
    limit: int = 5,
    original_question: str = None,
    skip_prefilter: bool = False,
    top_k: int = 3
) -> Dict[str, Any]:
    """Search knowledge_bank, return top-K scored results."""
    start_time = time.time()
    user_question = (question or "").strip()
    display_question = original_question if (skip_prefilter and original_question) else user_question

    if not user_question:
        return {
            "status": "error",
            "message": "Field 'question' wajib diisi",
            "source": "text"
        }

    logger.info(f"[TEXT-SEARCH] Question: {display_question[:50]}...")

    pre_filter_duration = 0.0
    
    if skip_prefilter:
        logger.info("[PRE-FILTER] Skipped (handled by orchestrator)")
        pre_filter_result = {"valid": True, "clean_question": user_question}
    else:
        pre_filter_start = time.time()
        pre_filter_result = await ai_pre_filter(user_question)
        pre_filter_duration = time.time() - pre_filter_start

        if not pre_filter_result.get("valid", True):
            total_duration = time.time() - start_time
            return {
                "status": "low_confidence",
                "message": pre_filter_result.get("reason", "Pertanyaan tidak relevan"),
                "source": "text",
                "data": {
                    "results": [],
                    "metadata": {
                        "wa_number": wa_number,
                        "original_question": display_question,
                        "final_question": "-",
                        "category": "-"
                    }
                },
                "timing": {
                    "ai_domain_sec": round(pre_filter_duration, 3),
                    "embedding_sec": 0.0,
                    "qdrant_sec": 0.0,
                    "total_sec": round(total_duration, 3)
                }
            }

    normalized_question = normalize_text(clean_location_terms(pre_filter_result.get("clean_question", user_question)))
    detected_category = detect_category(normalized_question)
    category_id = detected_category["id"] if detected_category else None

    embedding_start = time.time()
    [query_vector] = await encode_texts([normalized_question], model=model, prefix="query: ")
    embedding_duration = time.time() - embedding_start

    qdrant_start = time.time()
    category_filter = qdrant_models.Filter(must=[
        qdrant_models.FieldCondition(
            key="category_id",
            match=qdrant_models.MatchValue(value=category_id)
        )
    ]) if category_id else None

    qdrant_results = await qdrant.search(
        collection_name=config.COLLECTION_TEXT,
        query_vector=query_vector,
        limit=limit,
        query_filter=category_filter
    )
    qdrant_duration = time.time() - qdrant_start

    scored_results = []
    
    for hit in qdrant_results:
        dense_score = float(hit.score)
        rag_question = hit.payload.get("question_rag_name", "")
        overlap_score = keyword_overlap(normalized_question, rag_question)
        
        final_score = round((0.65 * dense_score) + (0.35 * overlap_score), 3)
        
        acceptance_note = "-"
        passes_threshold = False
        
        if dense_score >= 0.90:
            passes_threshold = True
            acceptance_note = "high_dense"
        elif dense_score >= 0.86 and overlap_score >= 0.25:
            passes_threshold = True
            acceptance_note = "good_overlap"
        elif dense_score >= 0.83 and overlap_score >= 0.15:
            passes_threshold = True
            acceptance_note = "needs_ai_check"
        elif dense_score >= 0.80:
            passes_threshold = True
            acceptance_note = "marginal"
        
        if passes_threshold:
            scored_results.append({
                "source": "text",
                "question": hit.payload.get("question", ""),
                "question_rag_name": rag_question,
                "answer_id": safe_parse_answer_id(hit.payload.get("answer_id")),
                "answer_doc": "",
                "category_id": hit.payload.get("category_id"),
                "dense_score": dense_score,
                "overlap_score": overlap_score,
                "final_score": final_score,
                "note": acceptance_note,
                "content_for_check": rag_question
            })

    scored_results = sorted(scored_results, key=lambda x: x["final_score"], reverse=True)
    
    top_results = scored_results[:top_k]
    
    logger.info(f"[TEXT-SEARCH] Found {len(scored_results)} results, returning top {len(top_results)}")
    for i, r in enumerate(top_results):
        logger.info(f"  [{i+1}] {r['question_rag_name'][:50]}... | dense={r['dense_score']:.3f} | overlap={r['overlap_score']:.3f} | final={r['final_score']:.3f}")

    total_duration = time.time() - start_time

    if top_results:
        return {
            "status": "has_candidates",
            "message": f"Found {len(top_results)} text candidates",
            "source": "text",
            "data": {
                "results": top_results,
                "count": len(top_results),
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": display_question,
                    "final_question": normalized_question,
                    "category": detected_category["name"] if detected_category else "Global"
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "embedding_sec": round(embedding_duration, 3),
                "qdrant_sec": round(qdrant_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }
    else:
        return {
            "status": "no_results",
            "message": "No text results found above threshold",
            "source": "text",
            "data": {
                "results": [],
                "count": 0,
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": display_question,
                    "final_question": normalized_question,
                    "category": detected_category["name"] if detected_category else "Global"
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "embedding_sec": round(embedding_duration, 3),
                "qdrant_sec": round(qdrant_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }
