"""
RAG Text Service - Search Module
Logic pencarian di knowledge_bank
TANPA fallback - hanya search di knowledge_bank
PAYLOAD DAN SCORING HARUS PERSIS SEPERTI V2!
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
from shared.utils import (
    normalize_text,
    clean_location_terms,
    keyword_overlap,
    detect_category,
    safe_parse_answer_id,
)
from shared.filtering import ai_pre_filter, ai_check_relevance

logger = logging.getLogger("rag_text.search")

# Global instances (diinisialisasi dari main.py)
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances dari main.py"""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def search_knowledge_bank(
    question: str,
    wa_number: str = "unknown",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search di knowledge_bank.
    TIDAK ada fallback - hanya search di collection ini.
    PAYLOAD DAN SCORING PERSIS SEPERTI V2!
    
    Returns:
        Dict dengan format v2-compatible response
    """
    start_time = time.time()
    user_question = (question or "").strip()
    whatsapp_number = wa_number

    if not user_question:
        return {
            "status": "error",
            "message": "Field 'question' wajib diisi"
        }

    logger.info(f"[USER-QUESTION] Pertanyaan User: {user_question}")

    # =====================================================
    # 1. AI Pre-Filter (PERSIS V2)
    # =====================================================
    pre_filter_start = time.time()
    pre_filter_result = ai_pre_filter(user_question)
    pre_filter_duration = time.time() - pre_filter_start

    # Jika tidak valid dari pre-filter
    if not pre_filter_result.get("valid", True):
        total_duration = time.time() - start_time
        return {
            "status": "low_confidence",
            "message": pre_filter_result.get("reason", "Pertanyaan tidak relevan"),
            "data": {
                "similar_questions": [],
                "metadata": {
                    "wa_number": whatsapp_number,
                    "original_question": user_question,
                    "final_question": "-",
                    "category": "-",
                    "ai_reason": pre_filter_result.get("reason", "-"),
                    "ai_reformulated": "-",
                    "final_score_top": "-"
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "ai_relevance_sec": 0.0,
                "embedding_sec": 0.0,
                "qdrant_sec": 0.0,
                "total_sec": round(total_duration, 3)
            }
        }

    # =====================================================
    # 2. Normalize question dan detect category (PERSIS V2)
    # =====================================================
    normalized_question = normalize_text(clean_location_terms(pre_filter_result.get("clean_question", user_question)))
    detected_category = detect_category(normalized_question)
    category_id = detected_category["id"] if detected_category else None

    # =====================================================
    # 3. Embedding & Query Qdrant (PERSIS V2)
    # =====================================================
    embedding_start = time.time()
    query_vector = model.encode("query: " + normalized_question).tolist()
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

    # Log kandidat hasil
    try:
        if qdrant_results:
            logger.info("[RAG-SEARCH] Kandidat Hasil Pencarian Awal")
            for index, hit in enumerate(qdrant_results[:3], start=1):
                rag_question = (hit.payload.get("question_rag_name") or "-").strip()
                dense_score = float(getattr(hit, "score", 0.0))
                answer_id = safe_parse_answer_id(hit.payload.get("answer_id"))
                category_id_hit = hit.payload.get("category_id", "-")

                overlap_score = keyword_overlap(normalized_question, rag_question)
                # PERSIS V2: Final score = 0.65 * dense + 0.35 * overlap
                final_score = round((0.65 * dense_score) + (0.35 * overlap_score), 3)

                logger.info(
                    f"[{index}] Question: {rag_question} | Dense: {dense_score:.3f} | Overlap: {overlap_score:.3f} | Final: {final_score:.3f}"
                )
        else:
            logger.warning("[RAG-SEARCH] Tidak ada hasil dari Qdrant.")
    except Exception as e:
        logger.error(f"[RAG-SEARCH] Gagal mencetak hasil pencarian awal: {e}")

    # =====================================================
    # 4. AI Relevance Check (PERSIS V2)
    # =====================================================
    relevance_check_start = time.time()
    relevance_result = {}
    if qdrant_results:
        relevance_result = ai_check_relevance(user_question, qdrant_results[0].payload["question_rag_name"])
    relevance_check_duration = time.time() - relevance_check_start

    # =====================================================
    # 5. Scoring Logic (PERSIS V2!)
    # =====================================================
    accepted_results, rejected_results = [], []
    for hit in qdrant_results:
        dense_score = float(hit.score)
        overlap_score = keyword_overlap(normalized_question, hit.payload["question_rag_name"])
        # PERSIS V2: Final score = 0.65 * dense + 0.35 * overlap
        final_score = round((0.65 * dense_score) + (0.35 * overlap_score), 3)

        acceptance_note, is_accepted = "-", False
        
        # SCORING LOGIC PERSIS V2:
        # 1. Auto accept jika dense >= 0.90
        if dense_score >= 0.90:
            is_accepted, acceptance_note = True, "auto_accepted_by_dense"
        # 2. Accept jika dense 0.86-0.89 DAN overlap >= 0.25
        elif 0.86 <= dense_score <= 0.89 and overlap_score >= 0.25:
            is_accepted, acceptance_note = True, "accepted_by_overlap"

        # 3. Accept by AI relevance jika dense >= 0.83 DAN overlap >= 0.15 DAN AI bilang relevant
        try:
            if not is_accepted and dense_score >= 0.83 and overlap_score >= 0.15 and relevance_result.get("relevant", False):
                is_accepted, acceptance_note = True, "accepted_by_ai_relevance"
        except Exception:
            pass

        # PAYLOAD RESULT PERSIS V2:
        result_item = {
            "question": hit.payload["question"],
            "question_rag_name": hit.payload["question_rag_name"],
            "answer_id": safe_parse_answer_id(hit.payload.get("answer_id")),
            "answer_doc": "",
            "category_id": hit.payload.get("category_id"),
            "dense_score": dense_score,
            "overlap_score": overlap_score,
            "final_score": final_score,
            "note": acceptance_note
        }
        (accepted_results if is_accepted else rejected_results).append(result_item)

    # Sort results
    accepted_results = sorted(accepted_results, key=lambda x: x["final_score"], reverse=True)
    rejected_results = sorted(rejected_results, key=lambda x: x["final_score"], reverse=True)

    # Cek AI relevance
    is_question_relevant = relevance_result.get("relevant", True)
    if not is_question_relevant:
        logger.info("[AI-POST] Pertanyaan dinilai TIDAK relevan oleh model relevance-check.")
        accepted_results = []

    if accepted_results:
        final_rag_output = accepted_results[0]["question_rag_name"]
    else:
        final_rag_output = "-"

    logger.info(f"[AI-POST] Output akan dikirim ke WABOT: '{final_rag_output}'")

    total_duration = time.time() - start_time

    # =====================================================
    # 6. RESPONSE PAYLOAD PERSIS V2!
    # =====================================================
    response_payload = {
        "status": "success" if accepted_results else "low_confidence",
        "message": "Hasil ditemukan" if accepted_results else "Tidak ada hasil cukup relevan",
        "source": "text",
        "data": {
            "similar_questions": accepted_results if accepted_results else rejected_results,
            "metadata": {
                "wa_number": whatsapp_number,
                "original_question": user_question,
                "final_question": normalized_question,
                "category": (detected_category["name"] if detected_category else "Global"),
                "ai_reason": relevance_result.get("reason", "-") if relevance_result else "-",
                "ai_reformulated": relevance_result.get("reformulated_question", "-") if relevance_result else "-",
                "final_score_top": (accepted_results[0]["final_score"] if accepted_results else "-")
            }
        },
        "timing": {
            "ai_domain_sec": round(pre_filter_duration, 3),
            "ai_relevance_sec": round(relevance_check_duration, 3),
            "embedding_sec": round(embedding_duration, 3),
            "qdrant_sec": round(qdrant_duration, 3),
            "total_sec": round(total_duration, 3)
        }
    }

    logger.info(f"[REQUEST] Total waktu: {total_duration:.3f} detik")

    return response_payload
