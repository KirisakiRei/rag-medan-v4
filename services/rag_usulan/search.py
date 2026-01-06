"""
RAG Usulan Service - Search Module
Logic pencarian di usulan_bank
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
from shared.filtering import ai_pre_filter_usulan, ai_relevance_usulan

logger = logging.getLogger("rag_usulan.search")

# Global instances (diinisialisasi dari main.py)
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None
rag_summary_logger = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient, summary_logger=None):
    """Set global instances dari main.py"""
    global model, qdrant, rag_summary_logger
    model = embedding_model
    qdrant = qdrant_client
    rag_summary_logger = summary_logger


async def search_usulan_bank(
    question: str,
    wa_number: str = "unknown",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search di usulan_bank.
    PAYLOAD DAN SCORING PERSIS SEPERTI V2!
    
    Returns:
        Dict dengan format v2-compatible response
    """
    start_time = time.time()
    user_request = (question or "").strip()
    whatsapp_number = wa_number

    if not user_request:
        return {
            "status": "error",
            "message": "Field 'question' wajib diisi"
        }

    logger.info(f"[USER-REQUEST] Request User: {user_request}")

    # =====================================================
    # 1. AI Pre-filter / Reformulation (PERSIS V2)
    # =====================================================
    reformulation_start = time.time()
    reformulation_result = ai_pre_filter_usulan(user_request)
    reformulation_duration = time.time() - reformulation_start
    clean_request = reformulation_result.get("clean_request", user_request)

    # =====================================================
    # 2. Embedding & Query Qdrant (PERSIS V2)
    # =====================================================
    embedding_start = time.time()
    query_vector = model.encode("query: " + clean_request).tolist()
    embedding_duration = time.time() - embedding_start

    qdrant_start = time.time()
    qdrant_results = await qdrant.search(
        collection_name=config.COLLECTION_USULAN,
        query_vector=query_vector,
        limit=limit
    )
    qdrant_duration = time.time() - qdrant_start

    # Log kandidat hasil
    try:
        if qdrant_results:
            logger.info("[USULAN-SEARCH] Kandidat Hasil Pencarian Usulan")
            for index, hit in enumerate(qdrant_results[:3], start=1):
                request_rag_name = (hit.payload.get("request_rag_name") or "-").strip()
                dense_score = float(getattr(hit, "score", 0.0))
                logger.info(f"[{index}] {request_rag_name} | Dense: {dense_score:.3f}")
        else:
            logger.warning("[USULAN-SEARCH] Tidak ada hasil dari Qdrant.")
    except Exception as e:
        logger.error(f"[USULAN-SEARCH] Gagal mencetak hasil pencarian: {e}")

    # =====================================================
    # 3. Scoring Logic (PERSIS V2!)
    # =====================================================
    accepted_results, rejected_results = [], []
    for hit in qdrant_results:
        dense_score = float(hit.score)
        # PERSIS V2: final_score = dense_score (tidak ada overlap untuk usulan)
        final_score = round(dense_score, 3)
        acceptance_note, is_accepted = "-", False
        
        # SCORING LOGIC PERSIS V2:
        # Accept jika dense >= 0.85
        if dense_score >= 0.85:
            is_accepted, acceptance_note = True, "Data yang Relevan Ditemukan"

        # PAYLOAD RESULT PERSIS V2:
        result_item = {
            "request_id": hit.payload.get("request_id"),
            "organization_id": hit.payload.get("organization_id"),
            "request_name": hit.payload.get("request_name"),
            "request_rag_name": hit.payload.get("request_rag_name"),
            "dense_score": dense_score,
            "final_score": final_score,
            "note": acceptance_note
        }
        (accepted_results if is_accepted else rejected_results).append(result_item)

    # Sort results
    accepted_results = sorted(accepted_results, key=lambda x: x["final_score"], reverse=True)
    rejected_results = sorted(rejected_results, key=lambda x: x["final_score"], reverse=True)

    # =====================================================
    # 4. AI Topic Relevance Check (PERSIS V2)
    # =====================================================
    if qdrant_results:
        top_rag_name = qdrant_results[0].payload.get("request_rag_name", "-")
        topic_check_result = ai_relevance_usulan(user_request, top_rag_name)
    else:
        topic_check_result = {"relevant": True, "reason": "Tidak ada hasil RAG"}

    # Jika topik tidak relevan
    if not topic_check_result.get("relevant", True):
        total_duration = time.time() - start_time
        logger.info(f"[AI-TOPIC-USULAN] Topik tidak relevan | Reason: {topic_check_result.get('reason')}")
        logger.info(f"[REQUEST] Total waktu: {total_duration:.3f} detik")

        if rag_summary_logger:
            rag_summary_logger.info(
                f"\n{'='*60}\n[USULAN TOPIC CHECK]\nUser: {user_request}\nTopik RAG: {top_rag_name}\n"
                f"Relevan: {topic_check_result.get('relevant')} | Reason: {topic_check_result.get('reason')}\n{'='*60}\n"
            )
        
        # RESPONSE PERSIS V2 untuk topik tidak relevan:
        return {
            "status": "low_confidence",
            "message": "Topik tidak relevan dengan pertanyaan pengguna",
            "reason": topic_check_result.get("reason", "-"),
            "data": {"similar_questions": []},
            "timing": {"total_sec": round(total_duration, 3)}
        }

    if accepted_results:
        final_usulan_output = accepted_results[0]["request_rag_name"]
    else:
        final_usulan_output = "-"

    logger.info(f"[USULAN-POST] Output akan dikirim ke WABOT: '{final_usulan_output}'")
    total_duration = time.time() - start_time

    # =====================================================
    # 5. RESPONSE PAYLOAD PERSIS V2!
    # =====================================================
    response_payload = {
        "status": "success" if accepted_results else "low_confidence",
        "message": "Hasil ditemukan" if accepted_results else "Tidak ada hasil cukup relevan",
        "data": {
            "similar_questions": accepted_results if accepted_results else rejected_results,
            "metadata": {
                "wa_number": whatsapp_number,
                "user_question": user_request,
                "final_score_top": (accepted_results[0]["final_score"] if accepted_results else "-")
            }
        },
        "timing": {
            "reform_sec": round(reformulation_duration, 3),
            "embedding_sec": round(embedding_duration, 3),
            "qdrant_sec": round(qdrant_duration, 3),
            "total_sec": round(total_duration, 3)
        }
    }

    logger.info(f"[REQUEST] Total waktu: {total_duration:.3f} detik")

    # Log summary
    try:
        if rag_summary_logger:
            summary_lines = [
                f"[USULAN] User: {user_request}",
                f"Results: {len(qdrant_results)} | Relevant: {topic_check_result.get('relevant')}"
            ]
            for index, result in enumerate(accepted_results[:3], start=1):
                summary_lines.append(f"{index}. {result['request_rag_name']} | Dense={result['dense_score']:.3f}")
            rag_summary_logger.info(" | ".join(summary_lines))
    except Exception as e:
        logger.warning(f"[LOGGING ERROR] Gagal mencetak ringkasan: {e}")

    return response_payload
