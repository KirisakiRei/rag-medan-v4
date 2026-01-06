"""
RAG Web Service - Search Module
Logic pencarian di web_scraping_bank
PAYLOAD HARUS PERSIS SEPERTI web-scraping V2!
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
# REMOVED: ai_check_relevance - post-filter sekarang di orchestrator

logger = logging.getLogger("rag_web.search")

# Global instances (diinisialisasi dari main.py)
model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances dari main.py"""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def search_web_bank(
    question: str,
    wa_number: str = "unknown",
    limit: int = 5,
    score_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Semantic search di RAG Web Scraping.
    PAYLOAD PERSIS SEPERTI web-scraping V2!
    
    Returns:
        Dict dengan format v2-compatible response
    """
    start_total = time.time()
    
    logger.info(f"Search request: question='{question}', wa_number={wa_number}")
    
    # Preprocess question (PERSIS V2)
    final_question = question.strip().rstrip("?").strip()
    
    # 1. Generate embedding untuk query
    start_embed = time.time()
    try:
        query_embedding = model.encode(final_question, convert_to_numpy=True).tolist()
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return _build_error_response(
            wa_number=wa_number,
            original_question=question,
            final_question=final_question,
            error_message=f"Embedding error: {str(e)}"
        )
    embed_time = time.time() - start_embed
    
    # 2. Search di Qdrant
    start_qdrant = time.time()
    try:
        results = await qdrant.search(
            collection_name=config.COLLECTION_WEB,
            query_vector=query_embedding,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="is_deleted",
                        match=qdrant_models.MatchValue(value=False)
                    )
                ]
            ),
            limit=limit,
            score_threshold=score_threshold
        )
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        return _build_error_response(
            wa_number=wa_number,
            original_question=question,
            final_question=final_question,
            error_message=f"Search error: {str(e)}"
        )
    qdrant_time = time.time() - start_qdrant
    
    total_time = time.time() - start_total
    
    # 3. Build response (PERSIS V2)
    if not results:
        logger.info(f"No results found for: {final_question}")
        return {
            "status": "success",
            "message": "Tidak ditemukan hasil yang relevan dari web scraping",
            "source": "web_scraping",
            "data": {
                "similar_questions": [],
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": question,
                    "final_question": final_question,
                    "category": "Web Scraping",
                    "ai_reason": "No results found",
                    "ai_reformulated": "",
                    "final_score_top": 0.0,
                    "web_info": None
                }
            },
            "timing": {
                "embedding_sec": round(embed_time, 3),
                "qdrant_sec": round(qdrant_time, 3),
                "total_sec": round(total_time, 3)
            }
        }
    
    # Build similar_questions dari hasil (PERSIS V2)
    similar_questions = []
    top_result_payload = results[0].payload
    top_result_score = float(results[0].score)
    
    for hit in results:
        payload = hit.payload
        score = float(hit.score)
        
        # PAYLOAD RESULT PERSIS V2:
        similar_questions.append({
            "question": "-",
            "question_rag_name": "-",
            "answer_id": None,
            "answer_doc": payload.get("content", ""),
            "category_id": None,
            "dense_score": round(score, 3),
            "overlap_score": 0.0,
            "final_score": round(score, 3),
            "note": "from_web_scraping"
        })
    
    # Build web_info dari top result (PERSIS V2)
    web_info = {
        "url": top_result_payload.get("url", ""),
        "title": top_result_payload.get("title", ""),
        "link_id": top_result_payload.get("link_id", "")
    }
    
    logger.info(f"Search completed: {len(results)} results, top_score={top_result_score:.3f}")
    
    # RESPONSE PERSIS V2:
    return {
        "status": "success",
        "message": "Hasil ditemukan dari web scraping",
        "source": "web_scraping",
        "data": {
            "similar_questions": similar_questions,
            "metadata": {
                "wa_number": wa_number,
                "original_question": question,
                "final_question": final_question,
                "category": "Web Scraping",
                "ai_reason": "",
                "ai_reformulated": "",
                "final_score_top": round(top_result_score, 3),
                "web_info": web_info
            }
        },
        "timing": {
            "embedding_sec": round(embed_time, 3),
            "qdrant_sec": round(qdrant_time, 3),
            "total_sec": round(total_time, 3)
        }
    }


def _build_error_response(
    wa_number: str,
    original_question: str,
    final_question: str,
    error_message: str
) -> Dict[str, Any]:
    """Build error response dengan format yang konsisten (PERSIS V2)."""
    return {
        "status": "error",
        "message": error_message,
        "source": "web_scraping",
        "data": {
            "similar_questions": [],
            "metadata": {
                "wa_number": wa_number,
                "original_question": original_question,
                "final_question": final_question,
                "category": "Web Scraping",
                "ai_reason": error_message,
                "ai_reformulated": "",
                "final_score_top": 0.0,
                "web_info": None
            }
        },
        "timing": {
            "embedding_sec": 0.0,
            "qdrant_sec": 0.0,
            "total_sec": 0.0
        }
    }


async def search_web_unified(
    question: str,
    original_question: str,
    wa_number: str = "unknown",
    top_k: int = 3,
    score_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Search di web_scraping_bank untuk unified/parallel mode.
    
    PERUBAHAN ARSITEKTUR (Option B):
    - TIDAK ada AI relevance check di sini
    - Return top_k hasil dengan scoring yang memenuhi threshold
    - AI relevance check dilakukan HANYA di orchestrator
    - Service ini hanya bertanggung jawab: query → score → return candidates
    
    LOGIC:
    1. Embed query
    2. Query ke web_scraping_bank
    3. Filter by score threshold >= 0.5
    4. Return top_k results dengan content_for_check untuk orchestrator
    """
    start_time = time.time()
    
    logger.info(f"[UNIFIED] Search web: {question[:50]}...")

    try:
        # 1. Embed query
        embedding_start = time.time()
        final_question = question.strip().rstrip("?").strip()
        query_embedding = model.encode(final_question, convert_to_numpy=True).tolist()
        embedding_duration = time.time() - embedding_start

        # 2. Query ke Qdrant dengan filter is_deleted=False
        qdrant_start = time.time()
        results = await qdrant.search(
            collection_name=config.COLLECTION_WEB,
            query_vector=query_embedding,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="is_deleted",
                        match=qdrant_models.MatchValue(value=False)
                    )
                ]
            ),
            limit=top_k * 2,  # Fetch lebih banyak untuk filtering
            score_threshold=score_threshold
        )
        qdrant_duration = time.time() - qdrant_start

        # Tidak ada hasil
        if not results:
            total_duration = time.time() - start_time
            logger.info("[UNIFIED] No web results found above threshold")
            return {
                "status": "no_results",
                "message": "Tidak ada hasil dari web_scraping_bank",
                "source": "web_scraping",
                "data": {
                    "results": [],
                    "count": 0,
                    "metadata": {
                        "wa_number": wa_number,
                        "original_question": original_question,
                        "final_question": final_question,
                        "score_threshold": score_threshold
                    }
                },
                "timing": {
                    "embedding_sec": round(embedding_duration, 3),
                    "qdrant_sec": round(qdrant_duration, 3),
                    "total_sec": round(total_duration, 3)
                }
            }

        # 3. Build scored results - TIDAK ada AI check
        scored_results = []
        
        for idx, hit in enumerate(results[:top_k]):  # Limit to top_k
            payload = hit.payload
            score = float(hit.score)
            web_content = payload.get("content", "")
            
            # Web info untuk metadata
            web_info = {
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "link_id": payload.get("link_id", "")
            }
            
            # Determine note based on score
            if score >= 0.7:
                note = "web_high_confidence"
            elif score >= 0.6:
                note = "web_good_confidence"
            else:
                note = "web_moderate_confidence"
            
            result_item = {
                "source": "web_scraping",
                "rank": idx + 1,
                "question": web_info.get("title", "-"),
                "answer_doc": web_content,
                "dense_score": round(score, 4),
                "overlap_score": 0.0,  # Web tidak punya overlap
                "final_score": round(score, 4),
                "note": note,
                # Content untuk AI check di orchestrator (limit 2000 chars)
                "content_for_check": web_content[:2000] if len(web_content) > 2000 else web_content,
                # Web-specific metadata
                "web_info": web_info
            }
            
            scored_results.append(result_item)
            logger.info(f"[UNIFIED] Web #{idx+1}: score={score:.4f} | url={web_info.get('url', '-')[:50]}...")

        total_duration = time.time() - start_time
        
        # 4. Return candidates untuk orchestrator
        logger.info(f"[UNIFIED] Web returning {len(scored_results)} candidates for orchestrator evaluation")
        
        return {
            "status": "has_candidates",
            "message": f"Found {len(scored_results)} web candidates",
            "source": "web_scraping",
            "data": {
                "results": scored_results,
                "count": len(scored_results),
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": original_question,
                    "final_question": final_question,
                    "score_threshold": score_threshold,
                    "top_score": scored_results[0]["final_score"] if scored_results else 0
                }
            },
            "timing": {
                "embedding_sec": round(embedding_duration, 3),
                "qdrant_sec": round(qdrant_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }

    except Exception as e:
        logger.exception(f"[UNIFIED] Web search error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "source": "web_scraping",
            "data": {
                "results": [],
                "count": 0,
                "metadata": {
                    "wa_number": wa_number,
                    "original_question": original_question,
                    "final_question": question,
                    "error": str(e)
                }
            },
            "timing": {"total_sec": round(time.time() - start_time, 3)}
        }
