"""
RAG Medan v3 - Shared Filtering Module
AI-based filtering and relevance checking
"""
import json
import re
import logging
import requests
from typing import Dict, Any, Optional
from requests.exceptions import ConnectionError, Timeout

from config import config
from shared.db import get_variable
from shared.utils import hard_filter_local
from shared.prompts import (
    PROMPT_PRE_FILTER_RAG,
    PROMPT_PRE_FILTER_USULAN,
    PROMPT_RELEVANCE_RAG,
    PROMPT_RELEVANCE_USULAN,
    PROMPT_RERANK
)

logger = logging.getLogger("filtering")


def _call_gemini_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 256
) -> Optional[str]:
    """
    Helper function untuk memanggil Gemini API.
    """
    try:
        url = f"{config.LLM_BASE_URL}/{config.LLM_MODEL}:generateContent?key={config.LLM_API_KEY}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt.strip()},
                        {"text": user_message.strip()}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": 1,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT)
        
        if response.status_code != 200:
            logger.error(f"[GEMINI] HTTP {response.status_code}: {response.text}")
            return None
            
        response_data = response.json()
        candidates = response_data.get("candidates", [])
        
        if not candidates:
            logger.warning(f"[GEMINI] No candidates in response")
            return None
            
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            return None
            
        return parts[0].get("text", "").strip()
        
    except Exception as e:
        logger.error(f"[GEMINI] Error calling API: {e}")
        return None


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from text response, handling markdown code blocks."""
    if not text:
        logger.warning("[JSON PARSE] Empty text received")
        return None
    
    try:
        cleaned_text = text.strip()
        original_text = cleaned_text  # Keep for logging
        
        # Step 1: Remove markdown code blocks - multiple patterns for robustness
        # Pattern 1: ```json\n{...}\n``` (standard)
        # Pattern 2: ```\n{...}\n``` (no language specified)
        # Pattern 3: ```json{...}``` (no newlines)
        code_block_patterns = [
            r"```json\s*\n?([\s\S]*?)\n?```",  # ```json ... ```
            r"```\s*\n?([\s\S]*?)\n?```",       # ``` ... ```
            r"```json([\s\S]*?)```",            # ```json...``` (no newlines)
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                cleaned_text = match.group(1).strip()
                logger.debug(f"[JSON PARSE] Stripped code block with pattern, content length: {len(cleaned_text)}")
                break
        
        # Step 2: Try direct JSON parse first (cleaner approach)
        try:
            result = json.loads(cleaned_text)
            logger.debug(f"[JSON PARSE] Direct parse SUCCESS")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"[JSON PARSE] Direct parse failed: {e}, trying regex...")
        
        # Step 3: Fallback - extract JSON object using regex (find first complete JSON object)
        # This handles cases where there's extra text before/after JSON
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Nested objects
            r"\{[^{}]+\}",                        # Simple object
            r"\{.*?\}",                           # Lazy match
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, cleaned_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    logger.debug(f"[JSON PARSE] Regex parse SUCCESS with pattern")
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Log full response for debugging (single line for PM2)
        safe_log = original_text.replace('\n', ' | ')[:300]
        logger.warning(f"[JSON PARSE] No valid JSON found. Response: {safe_log}")
        return None
        
    except json.JSONDecodeError as e:
        safe_log = text.replace('\n', ' | ')[:200]
        logger.warning(f"[JSON PARSE] Invalid JSON: {e}. Text: {safe_log}")
        return None
    except Exception as e:
        safe_log = text.replace('\n', ' | ')[:200]
        logger.exception(f"[JSON PARSE] Unexpected error: {e}. Text: {safe_log}")
        return None


def ai_pre_filter(question: str) -> Dict[str, Any]:
    """
    AI Pre-Filter untuk pertanyaan RAG.
    Menjalankan hard filter lokal dulu, kemudian AI filter.
    """
    try:
        logger.info(f"[AI-FILTER] Starting pre-filter for: {question[:50]}...")
        
        # Hard filter first
        hard_filter_result = hard_filter_local(question)
        if not hard_filter_result["valid"]:
            logger.info(f"[HARD FILTER] Rejected: {hard_filter_result['reason']}")
            return hard_filter_result

        # Get prompt from DB or use default
        prompt_from_db = get_variable("prompt_pre_filter_rag")
        system_prompt = prompt_from_db or PROMPT_PRE_FILTER_RAG

        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=question,
            temperature=0.0,
            max_tokens=256
        )

        if not llm_content:
            logger.warning("[AI-FILTER] No LLM response, defaulting to valid=True")
            return {"valid": True, "reason": "LLM tidak merespons (fallback)", "clean_question": question}

        parsed_result = _extract_json(llm_content)
        
        if not parsed_result or not isinstance(parsed_result, dict):
            # Fallback: try extract valid status from text
            logger.warning(f"[AI-FILTER] JSON parse failed, attempting text extraction...")
            lower_content = llm_content.lower()
            
            if '"valid": false' in lower_content or '"valid":false' in lower_content:
                logger.info("[AI-FILTER] Extracted valid=false from text")
                return {"valid": False, "reason": "Extracted from text (JSON parse failed)", "clean_question": question}
            
            # Default to valid=True to not block legitimate questions
            logger.warning("[AI-FILTER] Could not extract valid status, defaulting to True")
            return {"valid": True, "reason": "JSON parse gagal (fallback)", "clean_question": question}

        logger.info(f"[AI-FILTER] Result: valid={parsed_result.get('valid')}, reason={parsed_result.get('reason', '-')[:50]}")
        return parsed_result

    except (ConnectionError, Timeout) as e:
        logger.error(f"[AI-FILTER] Connection error: {e}")
        return {"valid": True, "reason": "LLM connection error (fallback)", "clean_question": question}

    except Exception as e:
        logger.exception(f"[AI-FILTER] Exception: {e}")
        return {"valid": True, "reason": f"Fallback error: {str(e)[:50]}", "clean_question": question}


def ai_check_relevance(user_question: str, rag_result: str) -> Dict[str, Any]:
    """
    AI Post-Filter untuk cek relevansi hasil RAG dengan pertanyaan.
    """
    try:
        logger.info(f"[AI-POST] Checking relevance...")
        
        prompt_from_db = get_variable("prompt_relevance_rag")
        system_prompt = prompt_from_db or PROMPT_RELEVANCE_RAG

        user_prompt = f"User: {user_question}\nRAG Result: {rag_result}"
        
        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.1,
            max_tokens=256
        )

        if not llm_content:
            logger.warning("[AI-POST] No LLM response, defaulting to relevant=True")
            return {"relevant": True, "reason": "LLM tidak merespons", "reformulated_question": ""}

        parsed_result = _extract_json(llm_content)
        if not parsed_result or not isinstance(parsed_result, dict):
            # IMPORTANT: Jika JSON parse gagal tapi LLM memberikan response,
            # coba extract relevant status dari text secara manual
            logger.warning(f"[AI-POST] JSON parse failed, attempting text extraction...")
            
            # Simple text-based extraction as fallback
            lower_content = llm_content.lower()
            if '"relevant": false' in lower_content or '"relevant":false' in lower_content:
                logger.info("[AI-POST] Extracted relevant=false from text")
                return {"relevant": False, "reason": "Extracted from text (JSON parse failed)", "reformulated_question": ""}
            elif '"relevant": true' in lower_content or '"relevant":true' in lower_content:
                logger.info("[AI-POST] Extracted relevant=true from text")
                return {"relevant": True, "reason": "Extracted from text (JSON parse failed)", "reformulated_question": ""}
            
            # Ultimate fallback - default to True untuk tidak memblok user
            logger.warning("[AI-POST] Could not extract relevant status, defaulting to True")
            return {"relevant": True, "reason": "JSON parse gagal", "reformulated_question": ""}

        # Truncate reformulated question
        reformulated_text = (parsed_result.get("reformulated_question") or "").strip()
        if len(reformulated_text.split()) > 12:
            parsed_result["reformulated_question"] = " ".join(reformulated_text.split()[:12]) + "..."

        logger.info(f"[AI-POST] Relevance result: relevant={parsed_result.get('relevant')}, reason={parsed_result.get('reason', '-')[:50]}")
        return parsed_result

    except (ConnectionError, Timeout) as e:
        logger.error(f"[AI-POST] Connection error: {e}")
        return {"relevant": True, "reason": f"LLM connection error", "reformulated_question": ""}

    except Exception as e:
        logger.exception(f"[AI-POST] Exception: {e}")
        return {"relevant": True, "reason": f"Error: {str(e)[:50]}", "reformulated_question": ""}


def ai_pre_filter_usulan(user_input: str) -> Dict[str, str]:
    """
    AI Pre-Filter untuk usulan - reformulasi input user.
    """
    try:
        logger.info(f"[AI-REFORM-USULAN] Starting reformulation...")

        prompt_from_db = get_variable("prompt_pre_filter_usulan")
        system_prompt = prompt_from_db or PROMPT_PRE_FILTER_USULAN

        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=user_input,
            temperature=0.2,
            max_tokens=256
        )

        if not llm_content:
            return {"clean_request": user_input}

        parsed_result = _extract_json(llm_content)
        clean_request = (parsed_result or {}).get("clean_request", user_input)

        logger.info(f"[AI-REFORM-USULAN] Result: {clean_request[:50]}...")
        return {"clean_request": clean_request}

    except Exception as e:
        logger.error(f"[AI-REFORM-USULAN] Error: {e}")
        return {"clean_request": user_input}


def ai_relevance_usulan(user_input: str, top_result: str) -> Dict[str, Any]:
    """
    AI Relevance Check untuk usulan.
    """
    try:
        logger.info(f"[AI-TOPIC-USULAN] Checking relevance...")

        prompt_from_db = get_variable("prompt_relevance_usulan")
        system_prompt = prompt_from_db or PROMPT_RELEVANCE_USULAN

        user_prompt = f'Pertanyaan pengguna: "{user_input}"\nTopik hasil RAG: "{top_result}"'

        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.0,
            max_tokens=256
        )

        if not llm_content:
            return {"relevant": True, "reason": "LLM error, skip relevance check"}

        parsed_result = _extract_json(llm_content)
        if not parsed_result or not isinstance(parsed_result, dict):
            parsed_result = {"relevant": True, "reason": "Fallback: invalid JSON"}

        logger.info(f"[AI-TOPIC-USULAN] Relevant: {parsed_result.get('relevant')}")
        return parsed_result

    except Exception as e:
        logger.error(f"[AI-TOPIC-USULAN] Error: {e}")
        return {"relevant": True, "reason": f"Fallback (error: {e})"}


def ai_rerank_results(
    question: str,
    text_results: list,
    document_results: list,
    web_results: list
) -> Dict[str, Any]:
    """
    AI Rerank - memilih dan menggabungkan hasil terbaik dari berbagai sumber.
    """
    try:
        logger.info(f"[AI-RERANK] Reranking results...")

        # Build context for AI
        context_parts = []
        
        if text_results:
            text_summary = "\n".join([
                f"- {r.get('question_rag_name', '-')} (score: {r.get('final_score', 0):.3f})"
                for r in text_results[:3]
            ])
            context_parts.append(f"TEXT RESULTS:\n{text_summary}")
        
        if document_results:
            doc_summary = "\n".join([
                f"- {r.get('text', '-')[:100]}... (score: {r.get('score', 0):.3f})"
                for r in document_results[:3]
            ])
            context_parts.append(f"DOCUMENT RESULTS:\n{doc_summary}")
        
        if web_results:
            web_summary = "\n".join([
                f"- {r.get('content', '-')[:100]}... (score: {r.get('score', 0):.3f})"
                for r in web_results[:3]
            ])
            context_parts.append(f"WEB RESULTS:\n{web_summary}")

        if not context_parts:
            return {
                "best_source": "none",
                "confidence": 0.0,
                "reason": "No results to rerank",
                "should_combine": False,
                "combined_sources": []
            }

        user_prompt = f"Question: {question}\n\n" + "\n\n".join(context_parts)

        llm_content = _call_gemini_llm(
            system_prompt=PROMPT_RERANK,
            user_message=user_prompt,
            temperature=0.1,
            max_tokens=256
        )

        if not llm_content:
            # Fallback: use highest score
            return _fallback_rerank(text_results, document_results, web_results)

        parsed_result = _extract_json(llm_content)
        if not parsed_result:
            return _fallback_rerank(text_results, document_results, web_results)

        logger.info(f"[AI-RERANK] Best source: {parsed_result.get('best_source')}")
        return parsed_result

    except Exception as e:
        logger.error(f"[AI-RERANK] Error: {e}")
        return _fallback_rerank(text_results, document_results, web_results)


def _fallback_rerank(text_results: list, document_results: list, web_results: list) -> Dict[str, Any]:
    """Fallback reranking berdasarkan score tertinggi."""
    scores = []
    
    if text_results:
        scores.append(("text", text_results[0].get("final_score", 0)))
    if document_results:
        scores.append(("document", document_results[0].get("score", 0)))
    if web_results:
        scores.append(("web", web_results[0].get("score", 0)))
    
    if not scores:
        return {
            "best_source": "none",
            "confidence": 0.0,
            "reason": "No results available",
            "should_combine": False,
            "combined_sources": []
        }
    
    best = max(scores, key=lambda x: x[1])
    return {
        "best_source": best[0],
        "confidence": best[1],
        "reason": "Fallback: highest score",
        "should_combine": False,
        "combined_sources": []
    }
