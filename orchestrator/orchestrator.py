"""
RAG Medan v3 - Orchestrator
Controller utama yang mengatur semua service RAG.

ARSITEKTUR V3:
1. User request masuk ke Orchestrator
2. Orchestrator melakukan PRE-FILTER (AI domain check, normalisasi)
3. Orchestrator meneruskan clean_question ke SEMUA 3 services secara PARALEL
4. Semua services mengembalikan hasil ke Orchestrator
5. Orchestrator MEMILIH hasil terbaik berdasarkan status:
   - Priority: text (success) > document (success) > web (success)
   - Jika semua low_confidence → return low_confidence dengan combined results

ENDPOINT SAMA DENGAN V2:
- POST /api/search         -> Unified RAG (Text + Document + Web PARALEL)
- POST /api/sync           -> Sync knowledge_bank
- POST /api/doc-search     -> Direct RAG Document only
- POST /api/doc-sync       -> Sync document_bank
- GET  /api/doc-sync/status/{task_id}
- DELETE /api/doc-delete   -> Delete document
- POST /api/sync-usulan    -> Sync usulan_bank
- POST /api/search-usulan  -> RAG Usulan

NEW IN V3:
- POST /api/web-trigger    -> Trigger scraping
- POST /api/web-sync       -> Sync edited web content
- DELETE /api/web-delete   -> Delete web content
- POST /api/web-search     -> Direct RAG Web only
"""
import os
import sys
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from shared.logging_config import setup_logging
from shared.filtering import ai_pre_filter, ai_check_relevance
from shared.utils import normalize_text, clean_location_terms, detect_category

# Setup logging
logger = setup_logging("orchestrator")

# HTTP client untuk komunikasi ke services
http_client: httpx.AsyncClient = None


# ============== REQUEST/RESPONSE MODELS (V2 COMPATIBLE) ==============

class SearchRequest(BaseModel):
    """Request untuk /api/search - SAMA DENGAN V2"""
    question: str
    wa_number: str = "unknown"


class SyncRequest(BaseModel):
    """Request untuk /api/sync - SAMA DENGAN V2"""
    action: str
    content: Optional[Any] = None


class DocSearchRequest(BaseModel):
    """Request untuk /api/doc-search - SAMA DENGAN V2"""
    query: str
    limit: int = 5


class DocSyncRequest(BaseModel):
    """Request untuk /api/doc-sync - SAMA DENGAN V2"""
    doc_id: str
    opd_name: Optional[str] = None
    file_url: str


class DocDeleteRequest(BaseModel):
    """Request untuk /api/doc-delete - SAMA DENGAN V2"""
    doc_id: str


class UsulanSyncRequest(BaseModel):
    """Request untuk /api/sync-usulan - SAMA DENGAN V2"""
    action: str
    content: Optional[Any] = None


class UsulanSearchRequest(BaseModel):
    """Request untuk /api/search-usulan - SAMA DENGAN V2"""
    question: str
    wa_number: str = "unknown"


# NEW V3 Models
class WebTriggerRequest(BaseModel):
    """Request untuk /api/web-trigger"""
    link_id: str
    url: str
    callback_url: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


class WebSyncRequest(BaseModel):
    """Request untuk /api/web-sync"""
    link_id: str
    edited_content: str


class WebDeleteRequest(BaseModel):
    """Request untuk /api/web-delete"""
    link_id: str


class WebSearchRequest(BaseModel):
    """Request untuk /api/web-search"""
    query: str
    limit: int = 5


# ============== SERVICE COMMUNICATION ==============

async def call_service(
    service_url: str, 
    endpoint: str, 
    method: str = "POST", 
    data: dict = None,
    timeout: float = 120.0
) -> dict:
    """
    Call internal service endpoint.
    
    Args:
        service_url: Base URL service
        endpoint: Endpoint path
        method: HTTP method
        data: Request data
        timeout: Request timeout
        
    Returns:
        Response dict
    """
    url = f"{service_url}{endpoint}"
    
    try:
        if method == "POST":
            response = await http_client.post(url, json=data, timeout=timeout)
        elif method == "GET":
            response = await http_client.get(url, timeout=timeout)
        elif method == "DELETE":
            response = await http_client.request("DELETE", url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response.json()
        
    except httpx.TimeoutException:
        logger.error(f"[SERVICE] Timeout calling {url}")
        return {"status": "error", "error": "Service timeout"}
    except httpx.ConnectError:
        logger.error(f"[SERVICE] Connection refused to {url}")
        return {"status": "error", "error": "Service unavailable"}
    except Exception as e:
        logger.error(f"[SERVICE] Error calling {url}: {e}")
        return {"status": "error", "error": str(e)}


async def call_service_safe(
    service_url: str,
    endpoint: str,
    method: str = "POST",
    data: dict = None,
    timeout: float = 60.0,
    service_name: str = "unknown"
) -> Tuple[str, dict]:
    """
    Call service dengan safety wrapper untuk parallel execution.
    Returns tuple (service_name, result) untuk identification.
    """
    try:
        result = await call_service(service_url, endpoint, method, data, timeout)
        return (service_name, result)
    except Exception as e:
        logger.error(f"[SERVICE] {service_name} error: {e}")
        return (service_name, {"status": "error", "error": str(e)})


# ============== INITIALIZATION ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global http_client
    
    logger.info("Starting RAG Medan v3 - Orchestrator...")
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient()
    
    logger.info("Orchestrator Started - PARALLEL SEARCH MODE")
    logger.info(f"  - Text Service: {config.TEXT_SERVICE_URL}")
    logger.info(f"  - Document Service: {config.DOCUMENT_SERVICE_URL}")
    logger.info(f"  - Web Service: {config.WEB_SERVICE_URL}")
    logger.info(f"  - Usulan Service: {config.USULAN_SERVICE_URL}")
    
    yield
    
    await http_client.aclose()
    logger.info("Orchestrator Shutting down...")


app = FastAPI(
    title="RAG Medan v3 - Orchestrator",
    description="Unified RAG Service Orchestrator with Parallel Search",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== HEALTH ENDPOINTS ==============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Medan v3 - Orchestrator is running!",
        "version": "3.0.0",
        "services": {
            "text": config.TEXT_SERVICE_URL,
            "document": config.DOCUMENT_SERVICE_URL,
            "web": config.WEB_SERVICE_URL,
            "usulan": config.USULAN_SERVICE_URL
        }
    }


@app.get("/health")
async def health_check():
    """Health check - cek semua services."""
    services_status = {}
    
    # Check text service
    try:
        text_health = await call_service(config.TEXT_SERVICE_URL, "/health", "GET", timeout=10.0)
        services_status["text_service"] = text_health.get("status") == "healthy"
    except:
        services_status["text_service"] = False
    
    # Check document service
    try:
        doc_health = await call_service(config.DOCUMENT_SERVICE_URL, "/health", "GET", timeout=10.0)
        services_status["document_service"] = doc_health.get("status") == "healthy"
    except:
        services_status["document_service"] = False
    
    # Check web service
    try:
        web_health = await call_service(config.WEB_SERVICE_URL, "/health", "GET", timeout=10.0)
        services_status["web_service"] = web_health.get("status") == "healthy"
    except:
        services_status["web_service"] = False
    
    # Check usulan service
    try:
        usulan_health = await call_service(config.USULAN_SERVICE_URL, "/health", "GET", timeout=10.0)
        services_status["usulan_service"] = usulan_health.get("status") == "healthy"
    except:
        services_status["usulan_service"] = False
    
    overall_status = all(services_status.values())
    
    return {
        "status": "healthy" if overall_status else "degraded",
        "service": "orchestrator",
        "mode": "parallel_search",
        "components": services_status
    }


# ============== UNIFIED SEARCH (V3 PARALLEL MODE - OPTION B) ==============

@app.post("/api/search")
async def unified_search(request: SearchRequest):
    """
    UNIFIED SEARCH - V3 PARALLEL MODE (OPTION B - Score-based Selection)
    
    ARSITEKTUR BARU:
    1. Pre-filter di orchestrator (AI domain check)
    2. Call SEMUA 3 services PARALEL (text, document, web)
    3. AGGREGATE semua results dari 3 services
    4. SORT by final_score (descending)
    5. AI RELEVANCE CHECK pada top candidates (di orchestrator saja)
    6. Return hasil RELEVANT dengan score tertinggi
    
    KEUNTUNGAN vs V3 lama:
    - AKURASI: Pemilihan berdasarkan score, bukan priority
    - EFISIEN: 1x AI call (di orchestrator), bukan 3x (di tiap service)
    - FALLBACK: Jika top candidate tidak relevant, coba next candidate
    """
    start_time = time.time()
    user_question = (request.question or "").strip()
    wa_number = request.wa_number

    if not user_question:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Field 'question' wajib diisi"}
        )

    logger.info(f"[SEARCH] Question: {user_question} | wa_number: {wa_number}")

    # =====================================================
    # 1. PRE-FILTER DI ORCHESTRATOR
    # =====================================================
    pre_filter_start = time.time()
    pre_filter_result = ai_pre_filter(user_question)
    pre_filter_duration = time.time() - pre_filter_start

    logger.info(f"[PRE-FILTER] Valid: {pre_filter_result.get('valid')} | Clean: {pre_filter_result.get('clean_question', '-')[:50]}...")

    # Jika tidak valid dari pre-filter, return langsung
    if not pre_filter_result.get("valid", True):
        total_duration = time.time() - start_time
        return JSONResponse(
            status_code=200,
            content={
                "status": "low_confidence",
                "message": pre_filter_result.get("reason", "Pertanyaan tidak relevan dengan layanan publik"),
                "source": "none",
                "data": {
                    "similar_questions": [],
                    "metadata": {
                        "wa_number": wa_number,
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
        )

    # Normalize question
    clean_question = pre_filter_result.get("clean_question", user_question)
    normalized_question = normalize_text(clean_location_terms(clean_question))
    detected_category = detect_category(normalized_question)

    # =====================================================
    # 2. PARALLEL CALL KE SEMUA SERVICES
    # =====================================================
    logger.info(f"[PARALLEL] Calling 3 services with clean_question: {normalized_question[:50]}...")
    parallel_start = time.time()

    # Prepare parallel tasks
    tasks = [
        call_service_safe(
            config.TEXT_SERVICE_URL,
            "/internal/search-unified",
            "POST",
            {
                "question": normalized_question,
                "original_question": user_question,
                "wa_number": wa_number,
                "top_k": 3
            },
            timeout=60.0,
            service_name="text"
        ),
        call_service_safe(
            config.DOCUMENT_SERVICE_URL,
            "/internal/search-unified",
            "POST",
            {
                "question": normalized_question,
                "original_question": user_question,
                "wa_number": wa_number,
                "top_k": 3
            },
            timeout=60.0,
            service_name="document"
        ),
        call_service_safe(
            config.WEB_SERVICE_URL,
            "/internal/search-unified",
            "POST",
            {
                "question": normalized_question,
                "original_question": user_question,
                "wa_number": wa_number,
                "top_k": 3
            },
            timeout=60.0,
            service_name="web"
        )
    ]

    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_duration = time.time() - parallel_start

    # Parse results
    service_results = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"[PARALLEL] Exception: {result}")
            continue
        service_name, response = result
        service_results[service_name] = response
        status = response.get("status", "error")
        count = response.get("data", {}).get("count", 0)
        logger.info(f"[PARALLEL] {service_name}: status={status}, candidates={count}")

    # =====================================================
    # 3. AGGREGATE SEMUA HASIL
    # =====================================================
    logger.info("[AGGREGATE] Mengumpulkan semua candidates dari 3 services...")
    
    all_candidates = []
    
    # Collect dari text service
    text_result = service_results.get("text", {})
    if text_result.get("status") == "has_candidates":
        text_candidates = text_result.get("data", {}).get("results", [])
        all_candidates.extend(text_candidates)
        logger.info(f"[AGGREGATE] Text: {len(text_candidates)} candidates")
    
    # Collect dari document service
    doc_result = service_results.get("document", {})
    if doc_result.get("status") == "has_candidates":
        doc_candidates = doc_result.get("data", {}).get("results", [])
        all_candidates.extend(doc_candidates)
        logger.info(f"[AGGREGATE] Document: {len(doc_candidates)} candidates")
    
    # Collect dari web service
    web_result = service_results.get("web", {})
    if web_result.get("status") == "has_candidates":
        web_candidates = web_result.get("data", {}).get("results", [])
        all_candidates.extend(web_candidates)
        logger.info(f"[AGGREGATE] Web: {len(web_candidates)} candidates")
    
    logger.info(f"[AGGREGATE] Total candidates: {len(all_candidates)}")

    # =====================================================
    # 4. SORT BY FINAL_SCORE (DESCENDING)
    # =====================================================
    if all_candidates:
        all_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        logger.info(f"[SORT] Top 3 scores: {[c.get('final_score', 0) for c in all_candidates[:3]]}")

    # =====================================================
    # 5. AI RELEVANCE CHECK (DI ORCHESTRATOR SAJA)
    # =====================================================
    total_duration = time.time() - start_time
    
    if not all_candidates:
        # Tidak ada candidates sama sekali
        logger.info("[RESULT] No candidates from any service")
        return JSONResponse(
            status_code=200,
            content={
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
        )
    
    # Check top candidates untuk relevance
    relevance_start = time.time()
    selected_candidate = None
    ai_reason = "-"
    candidates_checked = 0
    
    # Check top 3 candidates (atau semua jika kurang dari 3)
    max_check = min(3, len(all_candidates))
    
    for idx, candidate in enumerate(all_candidates[:max_check]):
        candidates_checked += 1
        source = candidate.get("source", "unknown")
        score = candidate.get("final_score", 0)
        content_for_check = candidate.get("content_for_check", "")
        
        # Jika score sangat tinggi (dense >= 0.90), skip AI check (trust score)
        if candidate.get("dense_score", 0) >= 0.90:
            logger.info(f"[RELEVANCE] Candidate #{idx+1} ({source}): HIGH SCORE {score:.4f} - SKIP AI CHECK")
            selected_candidate = candidate
            ai_reason = f"High confidence score (dense >= 0.90)"
            break
        
        # AI Relevance Check
        logger.info(f"[RELEVANCE] Checking candidate #{idx+1} ({source}): score={score:.4f}...")
        relevance_result = ai_check_relevance(user_question, content_for_check)
        
        is_relevant = relevance_result.get("relevant", False)
        ai_reason = relevance_result.get("reason", "-")
        
        if is_relevant:
            logger.info(f"[RELEVANCE] ✓ Candidate #{idx+1} ({source}) RELEVANT")
            selected_candidate = candidate
            break
        else:
            logger.info(f"[RELEVANCE] ✗ Candidate #{idx+1} ({source}) NOT RELEVANT - trying next...")
    
    relevance_duration = time.time() - relevance_start
    total_duration = time.time() - start_time

    # =====================================================
    # 6. BUILD RESPONSE
    # =====================================================
    
    if selected_candidate:
        # SUCCESS - Ada candidate relevant
        source = selected_candidate.get("source", "unknown")
        
        # Build similar_questions format (V2 compatible)
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
        
        # Source-specific info
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
                    "total_candidates": len(all_candidates)
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "ai_relevance_sec": round(relevance_duration, 3),
                "parallel_search_sec": round(parallel_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }
        
        # Add source-specific metadata
        if web_info:
            response_payload["data"]["metadata"]["web_info"] = web_info
        if document_info:
            response_payload["data"]["metadata"]["document_info"] = document_info
        
        logger.info(f"[RESULT] SUCCESS from {source} | score={selected_candidate.get('final_score', 0):.4f} | checked={candidates_checked} | total: {total_duration:.3f}s")
        return JSONResponse(status_code=200, content=response_payload)
    
    else:
        # LOW_CONFIDENCE - Semua candidates tidak relevant
        # Return top candidate untuk informasi
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
                    "total_candidates": len(all_candidates)
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "ai_relevance_sec": round(relevance_duration, 3),
                "parallel_search_sec": round(parallel_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }
        
        logger.info(f"[RESULT] LOW_CONFIDENCE | best_source={source} | score={top_candidate.get('final_score', 0):.4f if top_candidate else 0} | checked={candidates_checked} | total: {total_duration:.3f}s")
        return JSONResponse(status_code=200, content=response_payload)


# ============== TEXT RAG SYNC (V2 COMPATIBLE) ==============

@app.post("/api/sync")
async def sync_data(request: SyncRequest):
    """
    Sync data ke knowledge_bank.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SYNC] Action: {request.action}")
    
    result = await call_service(
        config.TEXT_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"action": request.action, "content": request.content}
    )
    
    return JSONResponse(status_code=200, content=result)


# ============== DOCUMENT RAG ENDPOINTS (V2 COMPATIBLE) ==============

@app.post("/api/doc-search")
async def doc_search(request: DocSearchRequest):
    """
    Search di RAG Document (document_bank) - DIRECT MODE.
    Tidak melalui parallel search, langsung ke document service.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[DOC-SEARCH] Query: {request.query}, limit: {request.limit}")
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/search",
        "POST",
        {"query": request.query, "limit": request.limit}
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Internal error"))
    
    return result


@app.post("/api/doc-sync")
async def doc_sync(request: DocSyncRequest):
    """
    Sync document (trigger OCR).
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[DOC-SYNC] doc_id={request.doc_id}, opd={request.opd_name}")
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/sync",
        "POST",
        {
            "doc_id": request.doc_id,
            "opd_name": request.opd_name,
            "file_url": request.file_url
        }
    )
    
    return result


@app.get("/api/doc-sync/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status task OCR.
    SAMA DENGAN V2.
    """
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        f"/internal/sync/status/{task_id}",
        "GET"
    )
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    
    return result


@app.get("/api/doc-sync/tasks")
async def list_tasks():
    """List semua tasks OCR."""
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/sync/tasks",
        "GET"
    )
    return result


@app.delete("/api/doc-delete")
async def doc_delete(request: DocDeleteRequest):
    """
    Soft delete document.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[DOC-DELETE] doc_id={request.doc_id}")
    
    result = await call_service(
        config.DOCUMENT_SERVICE_URL,
        "/internal/delete",
        "DELETE",
        {"doc_id": request.doc_id}
    )
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Document not found")
    
    return result


# ============== USULAN RAG ENDPOINTS (V2 COMPATIBLE) ==============

@app.post("/api/sync-usulan")
async def sync_usulan(request: UsulanSyncRequest):
    """
    Sync data ke usulan_bank.
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SYNC-USULAN] Action: {request.action}")
    
    result = await call_service(
        config.USULAN_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"action": request.action, "content": request.content}
    )
    
    return JSONResponse(status_code=200, content=result)


@app.post("/api/search-usulan")
async def search_usulan(request: UsulanSearchRequest):
    """
    Search di RAG Usulan (usulan_bank).
    Response format SAMA DENGAN V2.
    """
    logger.info(f"[SEARCH-USULAN] Question: {request.question}, wa_number: {request.wa_number}")
    
    result = await call_service(
        config.USULAN_SERVICE_URL,
        "/internal/search",
        "POST",
        {"question": request.question, "wa_number": request.wa_number}
    )
    
    if result.get("status") == "error":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": result.get("error", "Internal error")}
        )
    
    return JSONResponse(status_code=200, content=result)


# ============== WEB RAG ENDPOINTS (NEW IN V3) ==============

@app.post("/api/web-trigger")
async def trigger_web_scraping(request: WebTriggerRequest):
    """
    Trigger web scraping (NEW in v3).
    """
    logger.info(f"[WEB-TRIGGER] link_id={request.link_id}, url={request.url}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/trigger",
        "POST",
        {
            "link_id": request.link_id,
            "url": request.url,
            "callback_url": request.callback_url,
            "metadata": request.metadata
        }
    )
    
    return result


@app.post("/api/web-sync")
async def sync_web_content(request: WebSyncRequest):
    """
    Sync edited web content (NEW in v3).
    """
    logger.info(f"[WEB-SYNC] link_id={request.link_id}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/sync",
        "POST",
        {"link_id": request.link_id, "edited_content": request.edited_content}
    )
    
    return result


@app.delete("/api/web-delete")
async def delete_web_content(request: WebDeleteRequest):
    """
    Delete web content (NEW in v3).
    """
    logger.info(f"[WEB-DELETE] link_id={request.link_id}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/delete",
        "DELETE",
        {"link_id": request.link_id}
    )
    
    return result


@app.post("/api/web-search")
async def search_web(request: WebSearchRequest):
    """
    Search di RAG Web (web_scraping_bank) (NEW in v3).
    """
    logger.info(f"[WEB-SEARCH] query={request.query}")
    
    result = await call_service(
        config.WEB_SERVICE_URL,
        "/internal/search",
        "POST",
        {"query": request.query, "limit": request.limit}
    )
    
    return result


if __name__ == "__main__":
    uvicorn.run(
        "orchestrator.orchestrator:app",
        host=config.API_HOST,
        port=config.ORCHESTRATOR_PORT,
        reload=False,
        log_config=None
    )
