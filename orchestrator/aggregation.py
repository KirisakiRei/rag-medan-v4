"""Aggregation logic — aggregate, boost, and sort candidates from all services."""
import logging
from typing import Dict, List, Any

logger = logging.getLogger("orchestrator")

# Administrative keywords untuk smart boost
ADMINISTRATIVE_KEYWORDS = [
    "ktp", "kartu tanda penduduk", "kk", "kartu keluarga", "akta", "akte",
    "sim", "stnk", "bpjs", "npwp", "surat", "izin", "perizinan",
    "persyaratan", "syarat", "dokumen", "administrasi", "pendaftaran",
    "pembuatan", "pengurusan", "perpanjang", "ganti", "ubah data"
]


def is_administrative_question(question: str) -> bool:
    """Check if question is about administrative services."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in ADMINISTRATIVE_KEYWORDS)


def apply_administrative_boost(
    text_candidates: List[Dict[str, Any]],
    doc_candidates: List[Dict[str, Any]],
    is_administrative: bool
) -> None:
    """Apply +15% boost to TEXT candidates for administrative questions."""
    if not is_administrative or not text_candidates:
        return
    
    if not doc_candidates:
        for candidate in text_candidates:
            original_score = candidate.get("final_score", 0.0)
            boosted_score = min(original_score * 1.15, 1.0)
            candidate["final_score"] = round(boosted_score, 4)
            candidate["score_boosted"] = True
            candidate["boost_reason"] = "administrative_domain"
        logger.info("[BOOST] TEXT +15% applied (no document competition)")
        return
    
    top_doc_score = max([c.get("final_score", 0.0) for c in doc_candidates])
    top_text_score = max([c.get("final_score", 0.0) for c in text_candidates])
    
    should_boost = (
        top_doc_score < 0.85 or 
        (top_text_score >= 0.70 and (top_doc_score - top_text_score) < 0.15)
    )
    
    if should_boost:
        for candidate in text_candidates:
            original_score = candidate.get("final_score", 0.0)
            boosted_score = min(original_score * 1.15, 1.0)
            candidate["final_score"] = round(boosted_score, 4)
            candidate["score_boosted"] = True
            candidate["boost_reason"] = "administrative_domain"
        logger.info(f"[BOOST] TEXT +15% applied (top_doc={top_doc_score:.3f} < 0.85 or reasonable gap)")
    else:
        logger.info(f"[BOOST] TEXT boost SKIPPED (top_doc={top_doc_score:.3f} is very high, let natural competition)")


def aggregate_and_sort_candidates(
    service_results: Dict[str, Any],
    clean_question: str
) -> List[Dict[str, Any]]:
    """Aggregate candidates from all services, apply boost, and sort by score."""
    logger.info("[AGGREGATE] Collecting candidates from 3 services")
    logger.info("-" * 80)
    
    is_admin = is_administrative_question(clean_question)
    
    text_candidates = []
    doc_candidates = []
    web_candidates = []
    
    text_result = service_results.get("text", {})
    if text_result.get("status") == "has_candidates":
        text_candidates = text_result.get("data", {}).get("results", [])
        logger.info(f"[AGGREGATE] Text: {len(text_candidates)} candidates")
    
    doc_result = service_results.get("document", {})
    if doc_result.get("status") == "has_candidates":
        doc_candidates = doc_result.get("data", {}).get("results", [])
        logger.info(f"[AGGREGATE] Document: {len(doc_candidates)} candidates")
    
    web_result = service_results.get("web", {})
    if web_result.get("status") == "has_candidates":
        web_candidates = web_result.get("data", {}).get("results", [])
        logger.info(f"[AGGREGATE] Web: {len(web_candidates)} candidates")
    
    apply_administrative_boost(text_candidates, doc_candidates, is_admin)
    
    all_candidates = text_candidates + doc_candidates + web_candidates
    
    logger.info(f"[AGGREGATE] Total candidates: {len(all_candidates)}")
    logger.info("=" * 80)
    
    if all_candidates:
        all_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        logger.info("[SORT] Top candidates by score:")
        logger.info("-" * 80)
        for i, c in enumerate(all_candidates[:5]):
            logger.info(f"  [{i+1}] {c.get('source', '?').upper()} | score={c.get('final_score', 0):.4f}")
        logger.info("=" * 80)
    
    return all_candidates
