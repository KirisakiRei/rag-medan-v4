"""Search module for usulan_bank."""
import os
import sys
import time
import logging
from typing import Dict, Any, List

import httpx
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.filtering import ai_pre_filter_usulan, ai_relevance_usulan
from shared.utils import encode_texts

logger = logging.getLogger("rag_usulan.search")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None
rag_summary_logger = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient, summary_logger=None):
    """Set global instances."""
    global model, qdrant, rag_summary_logger
    model = embedding_model
    qdrant = qdrant_client
    rag_summary_logger = summary_logger


async def _search_lightrag_usulan(clean_request: str, limit: int) -> List[Dict[str, Any]]:
    """
    Query LightRAG Adapter dan kembalikan HANYA kandidat source_type='usulan'.

    Pipeline usulan terpisah dari unified search: index LightRAG sama,
    tetapi pemisahan dilakukan di level retrieval — konteks non-usulan
    dibuang di sini, dan unified search (/api/search) otomatis membuang
    konteks usulan (bukan bagian dari _SOURCE_PRIORITY).

    Returns:
        List of usulan candidate dicts (request_id, organization_id,
        request_name, request_rag_name, dense_score, final_score, note).
    """
    async with httpx.AsyncClient(
        base_url=config.LIGHTRAG_ADAPTER_URL,
        headers={"X-API-Key": config.INTERNAL_API_KEY},
        timeout=60.0,
    ) as client:
        response = await client.post(
            "/internal/search",
            json={
                "query": clean_request,
                "knowledge_base_id": "usulan-main",
                "mode": config.LIGHTRAG_QUERY_MODE,
                "top_k": limit * 3,
                "include_references": True,
            },
        )
        response.raise_for_status()
        result = response.json()

    if result.get("status") not in ("success", "no_results"):
        raise RuntimeError(f"LightRAG adapter gagal: {result.get('message') or result}")

    candidates = []
    for ctx in result.get("contexts") or []:
        if str(ctx.get("source_type") or "") != "usulan":
            continue
        content = str(ctx.get("content") or "").strip()
        if not content:
            continue
        candidates.append({
            "request_id": ctx.get("request_id"),
            "organization_id": ctx.get("organization_id"),
            "request_name": ctx.get("request_name"),
            "request_rag_name": str(ctx.get("title") or "").strip() or content,
            "dense_score": 0.0,
            "final_score": 0.0,
            "note": "lightrag_engine",
        })

    return candidates[:limit]


async def search_usulan_bank(
    question: str,
    wa_number: str = "unknown",
    limit: int = 5
) -> Dict[str, Any]:
    """Search usulan_bank, return scored results."""
    start_time = time.time()
    user_request = (question or "").strip()

    if not user_request:
        return {
            "status": "error",
            "message": "Field 'question' wajib diisi"
        }

    logger.info(f"[USER-REQUEST] Request User: {user_request}")

    # 1. AI Pre-filter / Reformulation
    reformulation_start = time.time()
    reformulation_result = await ai_pre_filter_usulan(user_request)
    reformulation_duration = time.time() - reformulation_start
    clean_request = reformulation_result.get("clean_request", user_request)

    # 2. Retrieval — LightRAG (filter source usulan), fallback Qdrant
    lightrag_candidates = []
    try:
        lightrag_candidates = await _search_lightrag_usulan(clean_request, limit)
        if lightrag_candidates:
            logger.info(f"[USULAN-SEARCH] LightRAG: {len(lightrag_candidates)} kandidat usulan")
    except Exception as e:
        logger.warning(f"[USULAN-SEARCH] LightRAG retrieval gagal, fallback Qdrant: {e}")

    if lightrag_candidates:
        accepted_results = lightrag_candidates
        rejected_results = []
        qdrant_results = []
        embedding_duration = 0.0
        qdrant_duration = 0.0
    else:
        # 2b. Embedding & Query Qdrant (fallback / legacy)
        embedding_start = time.time()
        [query_vector] = await encode_texts([clean_request], model=model, prefix="query: ")
        embedding_duration = time.time() - embedding_start

        qdrant_start = time.time()
        _qp_response = await qdrant.query_points(
            collection_name=config.COLLECTION_USULAN,
            query=query_vector,
            limit=limit
        )
        qdrant_results = _qp_response.points if hasattr(_qp_response, "points") else _qp_response
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

        # 3. Scoring Logic (legacy)
        accepted_results, rejected_results = [], []
        for hit in qdrant_results:
            dense_score = float(hit.score)
            final_score = round(dense_score, 3)
            acceptance_note, is_accepted = "-", False

            if dense_score >= 0.85:
                is_accepted, acceptance_note = True, "Data yang Relevan Ditemukan"

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

    # 4. AI Topic Relevance Check
    if accepted_results:
        top_rag_name = accepted_results[0]["request_rag_name"]
        topic_check_result = await ai_relevance_usulan(user_request, top_rag_name)
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

    # 5. Build Response
    response_payload = {
        "status": "success" if accepted_results else "low_confidence",
        "message": "Hasil ditemukan" if accepted_results else "Tidak ada hasil cukup relevan",
        "data": {
            "similar_questions": accepted_results if accepted_results else rejected_results,
            "metadata": {
                "wa_number": wa_number,
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
                f"Results: {len(accepted_results) + len(rejected_results)} | Relevant: {topic_check_result.get('relevant')}"
            ]
            for index, result in enumerate(accepted_results[:3], start=1):
                summary_lines.append(f"{index}. {result['request_rag_name']} | Dense={result['dense_score']:.3f}")
            rag_summary_logger.info(" | ".join(summary_lines))
    except Exception as e:
        logger.warning(f"[LOGGING ERROR] Gagal mencetak ringkasan: {e}")

    return response_payload
