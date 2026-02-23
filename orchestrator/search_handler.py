"""Search handler — unified search orchestration with parallel fan-out."""

import asyncio
import time
from typing import Dict, List, Any, Tuple
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from shared.filtering import ai_pre_filter, ai_check_relevance
from shared.utils import normalize_text, clean_location_terms, detect_category
from orchestrator.service_client import call_service_safe
from orchestrator.aggregation import aggregate_and_sort_candidates

logger = logging.getLogger(__name__)


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
        
        # High score bypass: skip AI check if dense >= 0.90
        if candidate.get("dense_score", 0) >= 0.90:
            logger.info(f"  [{idx+1}] {source.upper()}: score={score:.4f} → HIGH SCORE, SKIP AI CHECK ✓")
            selected_candidate = candidate
            ai_reason = f"High confidence score (dense >= 0.90)"
            break
        
        # AI Relevance Check
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
    services_queried: int = 3
) -> Dict[str, Any]:
    """Build low confidence response payload."""
    top_candidate = all_candidates[0] if all_candidates else {}
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

    # 2. PARALLEL SEARCH
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
        await check_relevance_with_ai(all_candidates, user_question, max_check=3)
    
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
            services_queried
        )
