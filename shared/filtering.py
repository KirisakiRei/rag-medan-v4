"""
RAG Medan v3 - Shared Filtering Module
AI-based filtering and relevance checking (async version)
"""
import asyncio
import json
import re
import time
import logging
import httpx
from typing import Dict, Any, List, Optional, Tuple

from config import config
from shared.db import get_variable
from shared.utils import hard_filter_local
from shared.prompts import (
    PROMPT_PRE_FILTER_RAG,
    PROMPT_PRE_FILTER_USULAN,
    PROMPT_RELEVANCE_RAG,
    PROMPT_AI_BATCH_RELEVANCE,
    PROMPT_RELEVANCE_USULAN,
    PROMPT_RERANK
)

logger = logging.getLogger("filtering")

_gemini_client: httpx.AsyncClient = None
_gemini_semaphore: asyncio.Semaphore = None
_prompt_cache: Dict[str, Tuple[str, float]] = {}


def _get_gemini_client() -> httpx.AsyncClient:
    """Get or create shared async HTTP client for Gemini API."""
    global _gemini_client
    if _gemini_client is None or _gemini_client.is_closed:
        _gemini_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=30.0)
        )
    return _gemini_client


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create semaphore for Gemini concurrency control."""
    global _gemini_semaphore
    if _gemini_semaphore is None:
        _gemini_semaphore = asyncio.Semaphore(config.GEMINI_MAX_CONCURRENT)
    return _gemini_semaphore

def get_cached_variable(key: str) -> Optional[str]:
    """Get variable from cache, falling back to DB if stale or missing."""
    now = time.time()
    if key in _prompt_cache:
        value, cached_at = _prompt_cache[key]
        if now - cached_at < config.PROMPT_CACHE_TTL:
            return value
    
    try:
        value = get_variable(key)
        _prompt_cache[key] = (value, now)
        return value
    except Exception as e:
        logger.warning(f"[CACHE] DB query failed for '{key}': {e}")
        if key in _prompt_cache:
            return _prompt_cache[key][0]
        return None


def get_relevance_mode() -> str:
    """Resolve relevance mode from database variable with config fallback."""
    database_mode = get_cached_variable("relevance_mode")
    configured_mode = (
        database_mode
        if str(database_mode or "").strip()
        else config.RELEVANCE_MODE
    )
    relevance_mode = str(configured_mode or "single").strip().lower()

    if relevance_mode not in {"single", "batch"}:
        logger.warning(
            f"[AI-RELEVANCE] Invalid relevance_mode='{relevance_mode}', falling back to single"
        )
        return "single"

    return relevance_mode


async def _call_gemini_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Optional[str]:
    """Call Gemini API with concurrency control via semaphore."""
    semaphore = _get_semaphore()
    async with semaphore:
        try:
            client = _get_gemini_client()
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
                    "topP": 1
                }
            }
            if max_tokens is not None:
                payload["generationConfig"]["maxOutputTokens"] = max_tokens
            
            headers = {"Content-Type": "application/json"}
            response = await client.post(url, headers=headers, json=payload)
            
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


async def call_filter_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Optional[str]:
    """Call LLM based on configured mode (Gemini or Router)."""
    mode = config.LLM_PROVIDER.lower()
    
    if mode == "gemini":
        return await _call_gemini_llm(system_prompt, user_message, temperature, max_tokens)
        
    # Router mode (OpenAI format)
    semaphore = _get_semaphore()
    async with semaphore:
        try:
            client = _get_gemini_client()
            url = config.ROUTER_API_URL
            api_key = get_cached_variable("router_api_key") or config.ROUTER_API_KEY
            model_name = get_cached_variable("llm_model") or config.LLM_MODEL
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_message.strip()}
                ],
                "temperature": temperature,
                "stream": False
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"[ROUTER] HTTP {response.status_code}: {response.text}")
                logger.warning("[ROUTER] Falling back to Gemini...")
                return await _call_gemini_llm(system_prompt, user_message, temperature, max_tokens)
                
            response_data = response.json()
            choices = response_data.get("choices", [])
            
            if not choices:
                logger.warning(f"[ROUTER] No choices in response")
                return None
                
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
            
        except Exception as e:
            logger.error(f"[ROUTER] Error calling API: {type(e).__name__}: {e}", exc_info=True)
            logger.warning("[ROUTER] Falling back to Gemini...")
            return await _call_gemini_llm(system_prompt, user_message, temperature, max_tokens)


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from text response, handling markdown code blocks."""
    if not text:
        logger.warning("[JSON PARSE] Empty text received")
        return None
    
    try:
        cleaned_text = text.strip()
        original_text = cleaned_text
        
        code_block_patterns = [
            r"```json\s*\n?([\s\S]*?)\n?```",
            r"```\s*\n?([\s\S]*?)\n?```",
            r"```json([\s\S]*?)```",
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                cleaned_text = match.group(1).strip()
                break
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
        
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            r"\{[^{}]+\}",
            r"\{.*?\}",
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, cleaned_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    continue
        
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


async def ai_pre_filter(question: str) -> Dict[str, Any]:
    """
    AI Pre-Filter untuk pertanyaan RAG (async).
    Menjalankan hard filter lokal dulu, kemudian AI filter.
    """
    try:
        logger.info(f"[AI-FILTER] Starting pre-filter for: {question[:50]}...")
        
        hard_filter_result = hard_filter_local(question)
        if not hard_filter_result["valid"]:
            logger.info(f"[HARD FILTER] Rejected: {hard_filter_result['reason']}")
            return hard_filter_result

        prompt = get_cached_variable("prompt_pre_filter_rag") or PROMPT_PRE_FILTER_RAG

        llm_response = await call_filter_llm(
            system_prompt=prompt,
            user_message=question,
            temperature=0.0
        )

        if not llm_response:
            logger.warning("[AI-FILTER] No LLM response, defaulting to valid=True")
            return {"valid": True, "reason": "LLM tidak merespons (fallback)", "clean_question": question}

        parsed = _extract_json(llm_response)
        
        if not parsed or not isinstance(parsed, dict):
            logger.warning("[AI-FILTER] JSON parse failed, attempting text extraction...")
            lower_text = llm_response.lower()
            
            if '"valid": false' in lower_text or '"valid":false' in lower_text:
                return {"valid": False, "reason": "Extracted from text (JSON parse failed)", "clean_question": question}
            
            return {"valid": True, "reason": "JSON parse gagal (fallback)", "clean_question": question}

        logger.info(f"[AI-FILTER] Result: valid={parsed.get('valid')}, reason={parsed.get('reason', '-')[:50]}")
        return parsed

    except httpx.ConnectError as e:
        logger.error(f"[AI-FILTER] Connection error: {e}")
        return {"valid": True, "reason": "LLM connection error (fallback)", "clean_question": question}

    except Exception as e:
        logger.exception(f"[AI-FILTER] Exception: {e}")
        return {"valid": True, "reason": f"Fallback error: {str(e)[:50]}", "clean_question": question}


async def ai_check_relevance(user_question: str, rag_result: str) -> Dict[str, Any]:
    """
    AI Post-Filter untuk cek relevansi hasil RAG dengan pertanyaan (async).
    """
    try:
        logger.info(f"[AI-POST] Checking relevance...")
        
        prompt = get_cached_variable("prompt_relevance_rag") or PROMPT_RELEVANCE_RAG
        user_prompt = f"User: {user_question}\nRAG Result: {rag_result}"
        
        llm_response = await call_filter_llm(
            system_prompt=prompt,
            user_message=user_prompt,
            temperature=0.1
        )

        if not llm_response:
            logger.warning("[AI-POST] No LLM response, defaulting to relevant=True")
            return {"relevant": True, "reason": "LLM tidak merespons", "reformulated_question": ""}

        parsed = _extract_json(llm_response)
        if not parsed or not isinstance(parsed, dict):
            logger.warning("[AI-POST] JSON parse failed, attempting text extraction...")
            lower_text = llm_response.lower()
            
            if '"relevant": false' in lower_text or '"relevant":false' in lower_text:
                return {"relevant": False, "reason": "Extracted from text (JSON parse failed)", "reformulated_question": ""}
            if '"relevant": true' in lower_text or '"relevant":true' in lower_text:
                return {"relevant": True, "reason": "Extracted from text (JSON parse failed)", "reformulated_question": ""}
            
            return {"relevant": True, "reason": "JSON parse gagal", "reformulated_question": ""}

        reformulated = (parsed.get("reformulated_question") or "").strip()
        if len(reformulated.split()) > 12:
            parsed["reformulated_question"] = " ".join(reformulated.split()[:12]) + "..."

        logger.info(f"[AI-POST] Relevance result: relevant={parsed.get('relevant')}, reason={parsed.get('reason', '-')[:50]}")
        return parsed

    except httpx.ConnectError as e:
        logger.error(f"[AI-POST] Connection error: {e}")
        return {"relevant": True, "reason": f"LLM connection error", "reformulated_question": ""}

    except Exception as e:
        logger.exception(f"[AI-POST] Exception: {e}")
        return {"relevant": True, "reason": f"Error: {str(e)[:50]}", "reformulated_question": ""}


async def ai_check_batch_relevance(
    user_question: str,
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Evaluate ordered RAG candidates in one LLM request."""
    batch_candidates = []
    for rank, candidate in enumerate(candidates[:5], start=1):
        content = str(candidate.get("content_for_check", "") or "")[:2000]
        batch_candidates.append({
            "rank": rank,
            "source": candidate.get("source", "unknown"),
            "final_score": candidate.get("final_score", 0.0),
            "content": content,
        })

    if not batch_candidates:
        return {
            "relevant": False,
            "selected_rank": None,
            "reason": "Tidak ada kandidat untuk diperiksa",
            "reformulated_question": "",
        }

    try:
        prompt = (
            get_cached_variable("prompt_ai_batch_relevance")
            or PROMPT_AI_BATCH_RELEVANCE
        )
        user_message = json.dumps(
            {
                "user_question": user_question,
                "candidates": batch_candidates,
            },
            ensure_ascii=False,
        )

        logger.info(
            f"[AI-BATCH] Checking {len(batch_candidates)} ordered candidates in one request"
        )
        llm_response = await call_filter_llm(
            system_prompt=prompt,
            user_message=user_message,
            temperature=0.0,
        )

        if not llm_response:
            logger.warning("[AI-BATCH] Empty LLM response")
            return None

        parsed = _extract_json(llm_response)
        if not parsed or not isinstance(parsed, dict):
            logger.warning("[AI-BATCH] Invalid JSON response")
            return None

        relevant = parsed.get("relevant")
        selected_rank = parsed.get("selected_rank")
        if type(relevant) is not bool:
            logger.warning("[AI-BATCH] Field 'relevant' must be boolean")
            return None

        if relevant:
            if (
                isinstance(selected_rank, bool)
                or not isinstance(selected_rank, int)
                or selected_rank < 1
                or selected_rank > len(batch_candidates)
            ):
                logger.warning(
                    f"[AI-BATCH] Invalid selected_rank={selected_rank} for {len(batch_candidates)} candidates"
                )
                return None
        elif selected_rank is not None:
            logger.warning("[AI-BATCH] selected_rank must be null when relevant=false")
            return None

        reason = parsed.get("reason")
        reformulated_question = parsed.get("reformulated_question")
        if not isinstance(reason, str):
            logger.warning("[AI-BATCH] Field 'reason' must be a string")
            return None
        if not isinstance(reformulated_question, str):
            logger.warning(
                "[AI-BATCH] Field 'reformulated_question' must be a string"
            )
            return None

        reformulated = reformulated_question.strip()
        if len(reformulated.split()) > 12:
            reformulated = " ".join(reformulated.split()[:12]) + "..."

        return {
            "relevant": relevant,
            "selected_rank": selected_rank,
            "reason": reason.strip() or "-",
            "reformulated_question": reformulated,
        }

    except Exception as e:
        logger.exception(f"[AI-BATCH] Error: {e}")
        return None


async def ai_pre_filter_usulan(user_input: str) -> Dict[str, str]:
    """
    AI Pre-Filter untuk usulan - reformulasi input user (async).
    """
    try:
        logger.info(f"[AI-REFORM-USULAN] Starting reformulation...")

        prompt = get_cached_variable("prompt_pre_filter_usulan") or PROMPT_PRE_FILTER_USULAN

        llm_response = await call_filter_llm(
            system_prompt=prompt,
            user_message=user_input,
            temperature=0.2
        )

        if not llm_response:
            return {"clean_request": user_input}

        parsed = _extract_json(llm_response)
        clean_request = (parsed or {}).get("clean_request", user_input)

        logger.info(f"[AI-REFORM-USULAN] Result: {clean_request[:50]}...")
        return {"clean_request": clean_request}

    except Exception as e:
        logger.error(f"[AI-REFORM-USULAN] Error: {e}")
        return {"clean_request": user_input}


async def ai_relevance_usulan(user_input: str, top_result: str) -> Dict[str, Any]:
    """
    AI Relevance Check untuk usulan (async).
    """
    try:
        logger.info(f"[AI-TOPIC-USULAN] Checking relevance...")

        prompt = get_cached_variable("prompt_relevance_usulan") or PROMPT_RELEVANCE_USULAN
        user_prompt = f'Pertanyaan pengguna: "{user_input}"\nTopik hasil RAG: "{top_result}"'

        llm_response = await call_filter_llm(
            system_prompt=prompt,
            user_message=user_prompt,
            temperature=0.0
        )

        if not llm_response:
            return {"relevant": True, "reason": "LLM error, skip relevance check"}

        parsed = _extract_json(llm_response)
        if not parsed or not isinstance(parsed, dict):
            parsed = {"relevant": True, "reason": "Fallback: invalid JSON"}

        logger.info(f"[AI-TOPIC-USULAN] Relevant: {parsed.get('relevant')}")
        return parsed

    except Exception as e:
        logger.error(f"[AI-TOPIC-USULAN] Error: {e}")
        return {"relevant": True, "reason": f"Fallback (error: {e})"}


async def ai_rerank_results(
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

        llm_response = await call_filter_llm(
            system_prompt=PROMPT_RERANK,
            user_message=user_prompt,
            temperature=0.1
        )

        if not llm_response:
            return _fallback_rerank(text_results, document_results, web_results)

        parsed = _extract_json(llm_response)
        if not parsed:
            return _fallback_rerank(text_results, document_results, web_results)

        logger.info(f"[AI-RERANK] Best source: {parsed.get('best_source')}")
        return parsed

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


async def ai_extract_answer(question: str, raw_text: str, source_type: str, metadata: Dict[str, Any]) -> Optional[str]:
    """
    RAG Extraction: Summarize the exact answer from the chunk based on the question.
    Only called for document and web sources.
    """
    try:
        from shared.prompts import PROMPT_EXTRACT_ANSWER
        
        logger.info(f"[AI-EXTRACT] Extracting answer for source_type={source_type}...")
        
        prompt = get_cached_variable("prompt_extract_answer") or PROMPT_EXTRACT_ANSWER
        
        # Check citation toggle
        enable_citation_str = get_cached_variable("enable_citation")
        if enable_citation_str is not None:
            enable_citation = str(enable_citation_str).lower() in ("true", "1", "yes")
        else:
            enable_citation = config.ENABLE_CITATION
            
        # Prepare citation metadata text
        citation_text = ""
        if enable_citation:
            if source_type == "document":
                filename = metadata.get("filename", "-")
                page = metadata.get("page_number", "-")
                citation_text = f"Dokumen: {filename}, Halaman: {page}"
            elif source_type == "web":
                url = metadata.get("url", metadata.get("link", "-"))
                citation_text = f"URL: {url}"
        else:
            # If citation is disabled, override the prompt instruction
            prompt += "\nPERHATIAN: DILARANG MENULISKAN ATAU MENYEBUTKAN SUMBER/REFERENSI APAPUN. BERIKAN JAWABAN SAJA."
            
        user_prompt = f"Pertanyaan: {question}\nReferensi Teks:\n{raw_text}\n\n"
        if citation_text:
            user_prompt += f"Metadata Rujukan:\n{citation_text}"

        llm_response = await call_filter_llm(
            system_prompt=prompt,
            user_message=user_prompt,
            temperature=0.1
        )

        if not llm_response:
            return None

        return llm_response.strip()

    except Exception as e:
        logger.error(f"[AI-EXTRACT] Error: {e}")
        return None
