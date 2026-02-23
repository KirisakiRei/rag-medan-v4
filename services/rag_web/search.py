"""Search module for web_scraping_bank."""
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
from shared.utils import encode_texts

logger = logging.getLogger("rag_web.search")

model: SentenceTransformer = None
qdrant: AsyncQdrantClient = None


def set_instances(embedding_model: SentenceTransformer, qdrant_client: AsyncQdrantClient):
    """Set global instances."""
    global model, qdrant
    model = embedding_model
    qdrant = qdrant_client


async def search_web_bank(
    question: str,
    wa_number: str = "unknown",
    limit: int = 5,
    score_threshold: float = 0.5
) -> Dict[str, Any]:
    """Semantic search in web_scraping_bank."""
    start_total = time.time()
    
    logger.info(f"Search request: question='{question}', wa_number={wa_number}")
    
    final_question = question.strip().rstrip("?").strip()
    
    start_embed = time.time()
    try:
        [query_embedding] = await encode_texts([final_question], model=model, prefix="query: ")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return _build_error_response(
            wa_number=wa_number,
            original_question=question,
            final_question=final_question,
            error_message=f"Embedding error: {str(e)}"
        )
    embed_time = time.time() - start_embed
    
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
    
    similar_questions = []
    top_result_payload = results[0].payload
    top_result_score = float(results[0].score)
    
    for hit in results:
        payload = hit.payload
        score = float(hit.score)
        
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
    
    web_info = {
        "url": top_result_payload.get("url", ""),
        "title": top_result_payload.get("title", ""),
        "link_id": top_result_payload.get("link_id", "")
    }
    
    logger.info(f"Search completed: {len(results)} results, top_score={top_result_score:.3f}")
    
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
    """Build consistent error response."""
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
    """Search web_scraping_bank for unified mode, return top-K candidates."""
    start_time = time.time()
    
    logger.info(f"[UNIFIED] Search web: {question[:50]}...")

    try:
        embedding_start = time.time()
        final_question = question.strip().rstrip("?").strip()
        [query_embedding] = await encode_texts([final_question], model=model, prefix="query: ")
        embedding_duration = time.time() - embedding_start

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

        scored_results = []
        
        for idx, hit in enumerate(results[:top_k]):
            payload = hit.payload
            score = float(hit.score)
            web_content = payload.get("content", "")
            
            web_info = {
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "link_id": payload.get("link_id", "")
            }
            
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
                "overlap_score": 0.0,
                "final_score": round(score, 4),
                "note": note,
                "content_for_check": web_content[:2000] if len(web_content) > 2000 else web_content,
                "web_info": web_info
            }
            
            scored_results.append(result_item)
            logger.info(f"[UNIFIED] Web #{idx+1}: score={score:.4f} | url={web_info.get('url', '-')[:50]}...")

        total_duration = time.time() - start_time
        
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
