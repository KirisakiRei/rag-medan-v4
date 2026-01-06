"""
RAG Medan v3 - Shared Module
Common utilities, schemas, and configurations
"""
from shared.db import get_variable, execute_query, execute_update
from shared.utils import (
    detect_category,
    normalize_text,
    clean_location_terms,
    expand_terms,
    tokenize_and_filter,
    keyword_overlap,
    hard_filter_local,
    safe_parse_answer_id,
    format_for_display,
    calculate_final_score,
    STOPWORDS,
    SYNONYMS,
    CATEGORY_KEYWORDS,
    CATEGORY_NAMES
)
from shared.filtering import (
    ai_pre_filter,
    ai_check_relevance,
    ai_pre_filter_usulan,
    ai_relevance_usulan,
    ai_rerank_results
)
from shared.logging_config import setup_logging, get_logger
from shared.prompts import (
    PROMPT_PRE_FILTER_RAG,
    PROMPT_PRE_FILTER_USULAN,
    PROMPT_RELEVANCE_RAG,
    PROMPT_RELEVANCE_USULAN,
    PROMPT_RERANK
)
from shared.ocr_utils import (
    extract_text_from_file,
    get_ocr_engine,
    clean_ocr_text,
    format_for_display as ocr_format_for_display,
    calculate_file_hash,
    calculate_content_hash
)
from shared.summarizer_utils import summarize_text

__all__ = [
    # Database
    "get_variable",
    "execute_query",
    "execute_update",
    
    # Utils
    "detect_category",
    "normalize_text",
    "clean_location_terms",
    "expand_terms",
    "tokenize_and_filter",
    "keyword_overlap",
    "hard_filter_local",
    "safe_parse_answer_id",
    "format_for_display",
    "calculate_final_score",
    "STOPWORDS",
    "SYNONYMS",
    "CATEGORY_KEYWORDS",
    "CATEGORY_NAMES",
    
    # Filtering
    "ai_pre_filter",
    "ai_check_relevance",
    "ai_pre_filter_usulan",
    "ai_relevance_usulan",
    "ai_rerank_results",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Prompts
    "PROMPT_PRE_FILTER_RAG",
    "PROMPT_PRE_FILTER_USULAN",
    "PROMPT_RELEVANCE_RAG",
    "PROMPT_RELEVANCE_USULAN",
    "PROMPT_RERANK",
    
    # OCR Utils
    "extract_text_from_file",
    "get_ocr_engine",
    "clean_ocr_text",
    "ocr_format_for_display",
    "calculate_file_hash",
    "calculate_content_hash",
    
    # Summarizer Utils
    "summarize_text"
]
