"""Search handler — unified search orchestration with parallel fan-out."""

import asyncio
import time
from typing import Dict, List, Any, Tuple
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from shared.filtering import (
    ai_pre_filter,
    ai_check_relevance,
    ai_check_batch_relevance,
    ai_extract_answer,
    get_relevance_mode,
)
from shared.utils import normalize_text, clean_location_terms, detect_category
from orchestrator.answer_validation import validate_extracted_answer
from orchestrator.service_client import call_service, call_service_safe
from orchestrator.aggregation import aggregate_and_sort_candidates

logger = logging.getLogger(__name__)


# ============== LIGHTRAG MAIN MODE ==============

async def _run_lightrag_search(
    normalized_question: str,
    user_question: str,
    wa_number: str,
    top_k: int = 3
) -> tuple[List[Dict[str, Any]], float, int]:
    """
    Eksekusi pencarian utama via LightRAG Adapter (menggantikan legacy parallel search).
    Map hasil dari LightRAG format (contexts) ke Legacy format (candidates)
    agar sisa pipeline (AI Relevance, AI Extraction, format Response) tetap berjalan normal.
    """
    logger.info(f"[LIGHTRAG-SEARCH] Querying LightRAG Adapter: {normalized_question[:60]}...")
    search_start = time.time()
    
    try:
        result = await call_service(
            config.LIGHTRAG_ADAPTER_URL,
            "/internal/search",
            "POST",
            {
                "query": normalized_question,
                "mode": config.LIGHTRAG_QUERY_MODE,
                "top_k": config.LIGHTRAG_TOP_K,
            },
            timeout=float(config.LIGHTRAG_TIMEOUT_SEC),
        )
    except Exception as e:
        logger.error(f"[LIGHTRAG-SEARCH] Adapter error: {e}")
        return [], time.time() - search_start, 1

    search_duration = time.time() - search_start
    contexts = result.get("contexts", [])
    
    all_candidates = []
    for ctx in contexts:
        source_type = ctx.get("source_type", "unknown")
        # Default fallback score jika LightRAG tidak mengeluarkan skor spesifik
        score = ctx.get("score")
        if score is None:
            score = 0.85
            
        candidate = {
            "source": source_type,
            "final_score": float(score),
            "content_for_check": ctx.get("content", ""),
            "answer_doc": ctx.get("content", ""),
            "question": ctx.get("title", ""),
            "note": "lightrag_engine"
        }
        
        # Mapping spesifik metadata untuk legacy response compatibility
        if source_type == "text":
            candidate["answer_id"] = ctx.get("source_id")
        elif source_type == "web":
            candidate["web_info"] = {
                "web_bank_id": ctx.get("source_id"),
                "url": ctx.get("source_uri", ""),
                "title": ctx.get("title", ""),
            }
        elif source_type == "document":
            candidate["document_info"] = {
                "doc_id": ctx.get("source_id"),
                "filename": ctx.get("title", ""),
            }
            
        all_candidates.append(candidate)
        
    # Sort berdasarkan skor
    all_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    # Ambil Top K
    all_candidates = all_candidates[:top_k]
    
    logger.info(f"[LIGHTRAG-SEARCH] Found {len(all_candidates)} mapped candidates in {search_duration:.2f}s")
    return all_candidates, search_duration, 1

def _run_shadow_lightrag_comparison(
    normalized_question: str,
    legacy_candidates: List[Dict[str, Any]],
) -> asyncio.Task:
    """
    Jalankan query LightRAG di background untuk evaluasi (shadow mode).

    Hasil perbandingan Legacy vs LightRAG di-log tanpa mempengaruhi
    response ke user. Legacy tetap menjadi sumber jawaban utama.
    """
    async def _shadow_comparison_task():
        try:
            lightrag_result = await call_service(
                config.LIGHTRAG_ADAPTER_URL,
                "/internal/search",
                "POST",
                {
                    "query": normalized_question,
                    "mode": config.LIGHTRAG_QUERY_MODE,
                    "top_k": config.LIGHTRAG_TOP_K,
                },
                timeout=float(config.LIGHTRAG_TIMEOUT_SEC),
            )

            # ── Legacy metrics ──
            legacy_candidate_count = len(legacy_candidates)
            legacy_top_score = (
                f"{float(legacy_candidates[0].get('final_score') or 0):.4f}"
                if legacy_candidates
                else "-"
            )

            # ── LightRAG metrics ──
            lightrag_contexts = lightrag_result.get("contexts", [])
            lightrag_result_count = len(lightrag_contexts)
            lightrag_status = lightrag_result.get("status", "unknown")
            lightrag_top_score = (
                f"{float(lightrag_contexts[0].get('score') or 0):.4f}"
                if lightrag_contexts
                else "-"
            )

            logger.info(
                f"[SHADOW] query='{normalized_question[:60]}' | "
                f"legacy={legacy_candidate_count} (top_score={legacy_top_score}) | "
                f"lightrag={lightrag_result_count} (top_score={lightrag_top_score}) | "
                f"status={lightrag_status}"
            )

        except Exception as exc:
            logger.warning(
                f"[SHADOW] LightRAG comparison failed: "
                f"{type(exc).__name__}: {exc}"
            )

    return asyncio.create_task(_shadow_comparison_task())


async def parallel_search_services(
    normalized_question: str,
    user_question: str,
    wa_number: str,
    top_k: int = 3
) -> tuple[Dict[str, Any], float, int]:
    """
    Fan-out to 3 services in parallel with adaptive early exit.
    Returns (service_results, parallel_duration, services_queried).
    """
    logger.info("[PARALLEL-SEARCH] Calling 3 services (adaptive fan-out)")
    logger.info("-" * 80)
    logger.info(f"[QUERY] {normalized_question}")
    parallel_start = time.time()

    # Create named tasks for identification
    task_configs = [
        ("text", config.TEXT_SERVICE_URL),
        ("document", config.DOCUMENT_SERVICE_URL),
        ("web", config.WEB_SERVICE_URL),
    ]
    
    tasks = {}
    for service_name, service_url in task_configs:
        task = asyncio.create_task(
            call_service_safe(
                service_url,
                "/internal/search-unified",
                "POST",
                {
                    "question": normalized_question,
                    "original_question": user_question,
                    "wa_number": wa_number,
                    "top_k": top_k
                },
                timeout=60.0,
                service_name=service_name
            ),
            name=f"search_{service_name}"
        )
        tasks[task] = service_name

    service_results = {}
    services_queried = 0
    early_exit = False
    pending = set(tasks.keys())

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        
        for completed_task in done:
            services_queried += 1
            try:
                result = completed_task.result()
                if isinstance(result, Exception):
                    logger.error(f"[PARALLEL] Exception: {result}")
                    continue
                    
                service_name, response = result
                service_results[service_name] = response
                status = response.get("status", "error")
                count = response.get("data", {}).get("count", 0)
                elapsed = time.time() - parallel_start
                logger.info(f"  ✓ {service_name.upper()}: status={status} | candidates={count} | elapsed={elapsed:.2f}s")
                
                # Check for early exit condition
                if not early_exit and status == "has_candidates":
                    candidates = response.get("data", {}).get("results", [])
                    for candidate in candidates:
                        if candidate.get("dense_score", 0) >= config.EARLY_EXIT_THRESHOLD:
                            logger.info(f"  ⚡ EARLY EXIT: {service_name.upper()} has dense_score={candidate['dense_score']:.4f} >= {config.EARLY_EXIT_THRESHOLD}")
                            early_exit = True
                            break
                
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[PARALLEL] Task error: {e}")
        
    # Cancel remaining tasks on early exit
        if early_exit and pending:
            cancelled_names = [tasks[t] for t in pending]
            logger.info(f"  ⚡ Cancelling pending services: {', '.join(cancelled_names)}")
            for task in pending:
                task.cancel()
            # Collect results from cancelled tasks that may have completed
            if pending:
                done_cancelled, _ = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
                for t in done_cancelled:
                    try:
                        result = t.result()
                        if not isinstance(result, Exception):
                            sn, resp = result
                            if resp.get("status") != "cancelled":
                                service_results[sn] = resp
                                services_queried += 1
                    except (asyncio.CancelledError, Exception):
                        pass
            break

    parallel_duration = time.time() - parallel_start
    logger.info(f"  Services queried: {services_queried}/3 | Early exit: {early_exit}")
    logger.info("=" * 80)
    return service_results, parallel_duration, services_queried


async def check_relevance_with_ai(
    all_candidates: List[Dict[str, Any]],
    user_question: str,
    max_check: int = 3
) -> tuple[Dict[str, Any] | None, str, int, float]:
    """
    Check top candidates for relevance using AI.
    Returns (selected_candidate, ai_reason, candidates_checked, duration).
    """
    logger.info("[AI-RELEVANCE] Checking top candidates")
    logger.info("-" * 80)
    
    relevance_start = time.time()
    selected_candidate = None
    ai_reason = "-"
    candidates_checked = 0
    
    # Check top N candidates
    max_check = min(max_check, len(all_candidates))
    
    for idx, candidate in enumerate(all_candidates[:max_check]):
        candidates_checked += 1
        source = candidate.get("source", "unknown")
        score = candidate.get("final_score", 0)
        content_for_check = candidate.get("content_for_check", "")
        
        if candidate.get("final_score", 0) >= 0.90:
            logger.info(f"  [{idx+1}] {source.upper()}: score={score:.4f} → HIGH SCORE, SKIP AI CHECK ✓")
            selected_candidate = candidate
            ai_reason = f"High confidence score (final >= 0.90)"
            break
        
        logger.info(f"  [{idx+1}] {source.upper()}: score={score:.4f} → Checking relevance...")
        relevance_result = await ai_check_relevance(user_question, content_for_check)
        
        is_relevant = relevance_result.get("relevant", False)
        ai_reason = relevance_result.get("reason", "-")
        
        if is_relevant:
            logger.info(f"       → RELEVANT ✓")
            selected_candidate = candidate
            break
        else:
            logger.info(f"       → NOT RELEVANT ✗ (trying next...)")
    
    logger.info("=" * 80)
    relevance_duration = time.time() - relevance_start
    
    return selected_candidate, ai_reason, candidates_checked, relevance_duration


async def check_relevance_batch_with_ai(
    all_candidates: List[Dict[str, Any]],
    user_question: str,
    max_check: int = 5,
) -> tuple[Dict[str, Any] | None, str, int, float]:
    """Check ordered candidates in one AI request, with single-mode fallback."""
    relevance_start = time.time()
    batch_candidates = all_candidates[:min(max_check, len(all_candidates))]

    if not batch_candidates:
        return None, "Tidak ada kandidat untuk diperiksa", 0, 0.0

    top_candidate = batch_candidates[0]
    top_score = top_candidate.get("final_score", 0)
    if top_score >= 0.90:
        logger.info(
            f"[AI-RELEVANCE] BATCH: top score={top_score:.4f} -> HIGH SCORE, SKIP AI CHECK ✓"
        )
        return (
            top_candidate,
            "High confidence score (final >= 0.90)",
            1,
            time.time() - relevance_start,
        )

    logger.info(
        f"[AI-RELEVANCE] BATCH: checking {len(batch_candidates)} candidates in one request"
    )
    batch_result = await ai_check_batch_relevance(user_question, batch_candidates)

    if batch_result is None:
        batch_duration = time.time() - relevance_start
        logger.warning(
            "[AI-RELEVANCE] Batch response invalid, falling back to single mode"
        )
        selected, reason, checked, single_duration = await check_relevance_with_ai(
            all_candidates,
            user_question,
            max_check=max_check,
        )
        return selected, reason, checked, batch_duration + single_duration

    ai_reason = batch_result.get("reason", "-")
    if not batch_result.get("relevant", False):
        relevance_duration = time.time() - relevance_start
        logger.info(
            f"[AI-RELEVANCE] BATCH: no relevant candidate | reason={ai_reason[:100]}"
        )
        return None, ai_reason, len(batch_candidates), relevance_duration

    selected_rank = batch_result["selected_rank"]
    selected_candidate = batch_candidates[selected_rank - 1]
    relevance_duration = time.time() - relevance_start
    logger.info(
        f"[AI-RELEVANCE] BATCH: selected rank={selected_rank} "
        f"source={selected_candidate.get('source', 'unknown').upper()} "
        f"score={selected_candidate.get('final_score', 0):.4f}"
    )
    return selected_candidate, ai_reason, len(batch_candidates), relevance_duration


async def check_relevance_by_mode(
    all_candidates: List[Dict[str, Any]],
    user_question: str,
    max_check: int = 5,
) -> tuple[Dict[str, Any] | None, str, int, float]:
    """Dispatch relevance checking using the configured single or batch mode."""
    relevance_mode = get_relevance_mode()
    logger.info(f"[AI-RELEVANCE] Mode: {relevance_mode.upper()}")

    if relevance_mode == "batch":
        return await check_relevance_batch_with_ai(
            all_candidates,
            user_question,
            max_check=max_check,
        )

    return await check_relevance_with_ai(
        all_candidates,
        user_question,
        max_check=max_check,
    )


def build_success_response(
    selected_candidate: Dict[str, Any],
    user_question: str,
    normalized_question: str,
    wa_number: str,
    detected_category: Dict[str, Any] | None,
    ai_reason: str,
    candidates_checked: int,
    total_candidates: int,
    pre_filter_duration: float,
    relevance_duration: float,
    parallel_duration: float,
    total_duration: float,
    services_queried: int = 3
) -> Dict[str, Any]:
    """Build success response payload."""
    source = selected_candidate.get("source", "unknown")
    
    similar_question = {
        "question": selected_candidate.get("question", "-"),
        "question_rag_name": selected_candidate.get("question", "-"),
        "answer_id": selected_candidate.get("answer_id"),
        "answer_doc": selected_candidate.get("answer_doc", ""),
        "category_id": selected_candidate.get("category_id"),
        "dense_score": selected_candidate.get("dense_score", 0.0),
        "overlap_score": selected_candidate.get("overlap_score", 0.0),
        "final_score": selected_candidate.get("final_score", 0.0),
        "note": selected_candidate.get("note", "-")
    }
    
    web_info = selected_candidate.get("web_info")
    document_info = selected_candidate.get("document_info")
    
    response_payload = {
        "status": "success",
        "message": f"Hasil ditemukan dari {source}",
        "source": source,
        "data": {
            "similar_questions": [similar_question],
            "metadata": {
                "wa_number": wa_number,
                "original_question": user_question,
                "final_question": normalized_question,
                "category": detected_category.get("name", "Global") if detected_category else "Global",
                "ai_reason": ai_reason,
                "ai_reformulated": "-",
                "final_score_top": selected_candidate.get("final_score", 0.0),
                "candidates_checked": candidates_checked,
                "total_candidates": total_candidates,
                "services_queried": services_queried
            }
        },
        "timing": {
            "ai_domain_sec": round(pre_filter_duration, 3),
            "ai_relevance_sec": round(relevance_duration, 3),
            "parallel_search_sec": round(parallel_duration, 3),
            "total_sec": round(total_duration, 3)
        }
    }
    
    if web_info:
        response_payload["data"]["metadata"]["web_info"] = web_info
    if document_info:
        response_payload["data"]["metadata"]["document_info"] = document_info
    
    logger.info("[RESULT] SUCCESS ✓")
    answer_preview = selected_candidate.get("answer_doc", "")[:120].replace("\n", " ")
    logger.info(f"  Answer ID   : {selected_candidate.get('answer_id', '-')}")
    logger.info(f"  Answer Preview: {answer_preview}...")
    logger.info(f"  Source: {source.upper()}")
    logger.info(f"  Score: {selected_candidate.get('final_score', 0):.4f}")
    logger.info(f"  Candidates Checked: {candidates_checked}/{total_candidates}")
    logger.info(f"  Services Queried: {services_queried}/3")
    logger.info(f"  Total Time: {total_duration:.3f}s")
    logger.info("=" * 80)
    
    return response_payload


def build_low_confidence_response(
    all_candidates: List[Dict[str, Any]],
    user_question: str,
    normalized_question: str,
    wa_number: str,
    detected_category: Dict[str, Any] | None,
    ai_reason: str,
    candidates_checked: int,
    pre_filter_duration: float,
    relevance_duration: float,
    parallel_duration: float,
    total_duration: float,
    services_queried: int = 3,
    selected_candidate_override: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Build low confidence response payload."""
    top_candidate = selected_candidate_override or (all_candidates[0] if all_candidates else {})
    source = top_candidate.get("source", "none")
    
    similar_question = {
        "question": top_candidate.get("question", "-"),
        "question_rag_name": top_candidate.get("question", "-"),
        "answer_id": top_candidate.get("answer_id"),
        "answer_doc": top_candidate.get("answer_doc", "")[:500] + "..." if len(top_candidate.get("answer_doc", "")) > 500 else top_candidate.get("answer_doc", ""),
        "category_id": top_candidate.get("category_id"),
        "dense_score": top_candidate.get("dense_score", 0.0),
        "overlap_score": top_candidate.get("overlap_score", 0.0),
        "final_score": top_candidate.get("final_score", 0.0),
        "note": f"not_relevant_checked_{candidates_checked}"
    }
    
    response_payload = {
        "status": "low_confidence",
        "message": "Tidak ada hasil yang cukup relevan",
        "source": source,
        "data": {
            "similar_questions": [similar_question] if top_candidate else [],
            "metadata": {
                "wa_number": wa_number,
                "original_question": user_question,
                "final_question": normalized_question,
                "category": detected_category.get("name", "Global") if detected_category else "Global",
                "ai_reason": ai_reason,
                "ai_reformulated": "-",
                "final_score_top": top_candidate.get("final_score", "-") if top_candidate else "-",
                "candidates_checked": candidates_checked,
                "total_candidates": len(all_candidates),
                "services_queried": services_queried
            }
        },
        "timing": {
            "ai_domain_sec": round(pre_filter_duration, 3),
            "ai_relevance_sec": round(relevance_duration, 3),
            "parallel_search_sec": round(parallel_duration, 3),
            "total_sec": round(total_duration, 3)
        }
    }
    
    best_score = top_candidate.get('final_score', 0) if top_candidate else 0
    logger.info("[RESULT] LOW_CONFIDENCE ✗")
    logger.info(f"  Best Source: {source.upper()}")
    logger.info(f"  Best Score: {best_score:.4f}")
    logger.info(f"  Candidates Checked: {candidates_checked}/{len(all_candidates)}")
    logger.info(f"  Services Queried: {services_queried}/3")
    logger.info(f"  Total Time: {total_duration:.3f}s")
    logger.info("=" * 80)
    
    return response_payload


async def unified_search(
    user_question: str,
    wa_number: str = "test-wa",
    use_ai_pre_filter: bool = False
) -> Dict[str, Any]:
    """
    Main unified search orchestration.
    Flow: pre-filter → parallel search → aggregate → AI relevance → response.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"[UNIFIED SEARCH] Question: {user_question}")
    logger.info("=" * 80)
    
    # 1. AI PRE-FILTER
    pre_filter_duration = 0.0
    clean_question = user_question
    
    if use_ai_pre_filter:
        logger.info("[AI-PRE-FILTER] Checking domain relevance...")
        pre_filter_start = time.time()
        pre_filter_result = await ai_pre_filter(user_question)
        pre_filter_duration = time.time() - pre_filter_start
        
        if not pre_filter_result.get("relevant", True):
            # Not relevant to domain
            total_duration = time.time() - start_time
            return {
                "status": "low_confidence",
                "message": "Pertanyaan di luar domain knowledge base",
                "source": "none",
                "data": {
                    "similar_questions": [],
                    "metadata": {
                        "wa_number": wa_number,
                        "original_question": user_question,
                        "final_question": "-",
                        "category": "-",
                        "ai_reason": pre_filter_result.get("reason", "Out of domain"),
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
        
        clean_question = pre_filter_result.get("clean_question", user_question)
        logger.info(f"[AI-PRE-FILTER] clean='{clean_question[:80]}' | duration={pre_filter_duration:.2f}s")

    # Normalize question
    normalized_question = normalize_text(clean_location_terms(clean_question))
    detected_category = detect_category(normalized_question)
    category_name = detected_category.get("name", "Global") if detected_category else "Global"
    logger.info(f"[NORMALIZE] normalized='{normalized_question[:60]}' | category={category_name}")

    engine = config.RAG_SEARCH_ENGINE.lower()
    shadow_lightrag_task = None

    if engine == "lightrag":
        # 2 & 3. MAIN LIGHTRAG SEARCH
        logger.info("[ENGINE] Menggunakan LightRAG sebagai mesin pencarian utama")
        all_candidates, parallel_duration, services_queried = await _run_lightrag_search(
            normalized_question, user_question, wa_number, top_k=5
        )
    else:
        # 2. PARALLEL SEARCH (LEGACY / SHADOW)
        logger.info(f"[ENGINE] Menggunakan Legacy parallel search (Mode: {engine})")
        service_results, parallel_duration, services_queried = await parallel_search_services(
            normalized_question,
            user_question,
            wa_number
        )
        
        # 3. AGGREGATE + BOOST + SORT
        all_candidates = aggregate_and_sort_candidates(
            service_results,
            clean_question
        )

        # 3.5 SHADOW MODE — query LightRAG di background untuk evaluasi
        if engine == "shadow":
            shadow_lightrag_task = _run_shadow_lightrag_comparison(
                normalized_question,
                all_candidates,
            )
    
    # 4. EMPTY CHECK
    total_duration = time.time() - start_time
    
    if not all_candidates:
        logger.info("[RESULT] No candidates from any service")
        return {
            "status": "low_confidence",
            "message": "Tidak ada hasil ditemukan dari semua sumber",
            "source": "none",
            "data": {
                "similar_questions": [],
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": user_question,
                    "final_question": normalized_question,
                    "category": detected_category.get("name", "Global") if detected_category else "Global",
                    "ai_reason": "Tidak ada kandidat dari text, document, dan web",
                    "ai_reformulated": "-",
                    "final_score_top": "-"
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "ai_relevance_sec": 0.0,
                "embedding_sec": 0.0,
                "qdrant_sec": 0.0,
                "parallel_search_sec": round(parallel_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }
    
    # 5. AI RELEVANCE CHECK
    selected_candidate, ai_reason, candidates_checked, relevance_duration = \
        await check_relevance_by_mode(all_candidates, user_question, max_check=5)
    extraction_failure_candidate = None
    
    # 5.5 AI EXTRACTION FOR DOCUMENT & WEB
    if selected_candidate:
        source = selected_candidate.get("source", "unknown")
        if source in ["document", "web"]:
            logger.info(f"[AI-EXTRACT] Extracting answer from {source.upper()}...")
            extract_start = time.time()
            raw_text = selected_candidate.get("answer_doc", "")
            
            # Prepare metadata
            metadata = {}
            if source == "document":
                metadata = selected_candidate.get("document_info", {})
            elif source == "web":
                metadata = selected_candidate.get("web_info", {})
                
            extracted_answer = await ai_extract_answer(
                user_question, 
                raw_text, 
                source, 
                metadata
            )

            is_valid_answer, invalid_reason = validate_extracted_answer(extracted_answer)
            if is_valid_answer:
                selected_candidate["answer_doc"] = extracted_answer.strip()
                logger.info("[AI-EXTRACT] Valid answer accepted")
            else:
                selected_candidate["answer_doc"] = "Tidak ditemukan"
                selected_candidate["note"] = f"invalid_extraction_{invalid_reason}"
                extraction_failure_candidate = selected_candidate
                selected_candidate = None
                ai_reason = f"AI extraction rejected: {invalid_reason}"
                logger.warning(f"[AI-EXTRACT] Invalid answer rejected: {invalid_reason}")
            
            extract_duration = time.time() - extract_start
            logger.info(f"[AI-EXTRACT] Done in {extract_duration:.2f}s")
            
    total_duration = time.time() - start_time
    
    # 6. BUILD RESPONSE
    if selected_candidate:
        return build_success_response(
            selected_candidate,
            user_question,
            normalized_question,
            wa_number,
            detected_category,
            ai_reason,
            candidates_checked,
            len(all_candidates),
            pre_filter_duration,
            relevance_duration,
            parallel_duration,
            total_duration,
            services_queried
        )
    else:
        return build_low_confidence_response(
            all_candidates,
            user_question,
            normalized_question,
            wa_number,
            detected_category,
            ai_reason,
            candidates_checked,
            pre_filter_duration,
            relevance_duration,
            parallel_duration,
            total_duration,
            services_queried,
            extraction_failure_candidate
        )
