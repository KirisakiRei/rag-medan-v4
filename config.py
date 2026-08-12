"""
RAG Medan v3 - Configuration
Unified configuration untuk orchestrator dan semua services
"""
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()


def _env(key, default=None, cast=str):
    val = os.getenv(key, default)
    if cast is int:
        try:
            return int(val)
        except Exception:
            return int(default) if default is not None else None
    if cast is float:
        try:
            return float(val)
        except Exception:
            return float(default) if default is not None else None
    if cast is bool:
        return str(val).lower() in ("true", "1", "yes")
    if cast is list:
        return [x.strip() for x in str(val).split(",") if x.strip()] if val else []
    return val


class Config:
    """Main configuration class untuk RAG Medan v3"""
    
    # API Configuration
    API_HOST = _env("API_HOST", "0.0.0.0")
    
    # Service Ports (internal communication)
    ORCHESTRATOR_PORT = _env("ORCHESTRATOR_PORT", 5000, int)
    TEXT_SERVICE_PORT = _env("TEXT_SERVICE_PORT", 5010, int)
    DOCUMENT_SERVICE_PORT = _env("DOCUMENT_SERVICE_PORT", 5011, int)
    WEB_SERVICE_PORT = _env("WEB_SERVICE_PORT", 5012, int)
    USULAN_SERVICE_PORT = _env("USULAN_SERVICE_PORT", 5013, int)
    
    # Service URLs (for orchestrator to call)
    TEXT_SERVICE_URL = _env("TEXT_SERVICE_URL", "http://localhost:5010")
    DOCUMENT_SERVICE_URL = _env("DOCUMENT_SERVICE_URL", "http://localhost:5011")
    WEB_SERVICE_URL = _env("WEB_SERVICE_URL", "http://localhost:5012")
    USULAN_SERVICE_URL = _env("USULAN_SERVICE_URL", "http://localhost:5013")
    
    # Embedding Models
    EMBEDDING_MODEL_PATH = _env("EMB_MODEL_PATH")
    EMBEDDING_MODEL_PATH_LARGE = _env("EMB_LARGE_PATH")
    EMBEDDING_DIMENSION = _env("EMBEDDING_DIMENSION", 384, int)
    EMBEDDING_DIMENSION_LARGE = _env("EMBEDDING_DIMENSION_LARGE", 1024, int)
    
    # Qdrant Configuration
    QDRANT_HOST = _env("QDRANT_HOST", "localhost")
    QDRANT_PORT = _env("QDRANT_PORT", 6333, int)
    QDRANT_API_KEY = _env("QDRANT_API_KEY", None)
    
    # Collection Names
    COLLECTION_TEXT = _env("COLLECTION_TEXT", "knowledge_bank")
    COLLECTION_DOCUMENT = _env("COLLECTION_DOCUMENT", "document_bank")
    COLLECTION_WEB = _env("COLLECTION_WEB", "web_scraping_bank")
    COLLECTION_WEB_STATE = _env("COLLECTION_WEB_STATE", "web_scraping_state")
    COLLECTION_USULAN = _env("COLLECTION_USULAN", "usulan_bank")
    COLLECTION_DOCUMENT_DEDUP_REGISTRY = _env("COLLECTION_DOCUMENT_DEDUP_REGISTRY", "document_dedup_registry")
    
    # LLM Configuration
    LLM_BASE_URL = _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")
    LLM_API_KEY = _env("LLM_API_KEY", "")
    LLM_MODEL = _env("LLM_MODEL", "gemini-2.0-flash")
    LLM_TIMEOUT = _env("LLM_TIMEOUT_SEC", 60, int)
    LLM_PROVIDER = _env("LLM_PROVIDER", "gemini") # "gemini" atau "router"
    RELEVANCE_MODE = _env("RELEVANCE_MODE", "single")
    
    ENABLE_CITATION = _env("ENABLE_CITATION", "true", bool)
    
    # Database Configuration
    DB_HOST = _env("DB_HOST", "localhost")
    DB_PORT = _env("DB_PORT", 3306, int)
    DB_DATABASE = _env("DB_DATABASE", "")
    DB_USERNAME = _env("DB_USERNAME", "")
    DB_PASSWORD = _env("DB_PASSWORD", "")
    
    # OCR Configuration
    OCR_TIMEOUT = _env("OCR_TIMEOUT", 1800, int)  # default 30 menit
    OCR_STALL_TIMEOUT = _env("OCR_STALL_TIMEOUT", 300, int)
    OCR_HARD_TIMEOUT = _env("OCR_HARD_TIMEOUT", 21600, int)
    OCR_QUEUE_LOG_INTERVAL = _env("OCR_QUEUE_LOG_INTERVAL", 30, int)
    OCR_PDF_DPI = _env("OCR_PDF_DPI", 200, int)         # 200 untuk ebook/majalah; 150 cukup utk surat biasa
    OCR_PDF_DPI_RETRY = _env("OCR_PDF_DPI_RETRY", 250, int)  # retry jika hasil OCR < 20 char
    OCR_DOWNLOAD_PROGRESS_MB = _env("OCR_DOWNLOAD_PROGRESS_MB", 5, int)

    # OCR Mode: "local" (PaddleOCR) atau "api" (LLM via Router API)
    OCR_MODE = _env("OCR_MODE", "local")

    # Router API Configuration (untuk OCR_MODE=api)
    ROUTER_API_URL = _env("ROUTER_API_URL", "http://localhost:20128/v1/chat/completions")
    ROUTER_API_KEY = _env("ROUTER_API_KEY", "")

    # LLM OCR Model Settings
    OCR_LLM_MODEL = _env("OCR_LLM_MODEL", "ag/gemini-3-flash")
    OCR_LLM_MAX_TOKENS = _env("OCR_LLM_MAX_TOKENS", 8192, int)
    OCR_DELAY = _env("OCR_DELAY", 2, int)     # Jeda (detik) antar request ke Router API per halaman
    OCR_RETRIES = _env("OCR_RETRIES", 3, int) # Jumlah maksimum percobaan ulang jika request gagal

    # RAG Configuration
    USE_POST_SUMMARY = _env("USE_POST_SUMMARY", "false", bool)
    POST_SUMMARY_TOP_K = _env("POST_SUMMARY_TOP_K", 2, int)
    
    # Chunking Configuration (Document & Web)
    # Nilai dikalibrasi untuk model E5 Large (multilingual, batas 512 token).
    # Estimator internal menggunakan rasio 3 char/token (vs 4 untuk Inggris)
    # karena subword tokenization Bahasa Indonesia lebih banyak.
    # Child chunk: ~380 est. token → ~330-360 token nyata (aman + margin heading prefix)
    # Parent chunk: tidak di-embed, digunakan untuk retrieval context expansion
    DOC_CHILD_CHUNK_SIZE = _env("DOC_CHILD_CHUNK_SIZE", 380, int)
    DOC_PARENT_CHUNK_SIZE = _env("DOC_PARENT_CHUNK_SIZE", 1100, int)
    DOC_CHUNK_OVERLAP = _env("DOC_CHUNK_OVERLAP", 70, int)
    WEB_CHILD_CHUNK_SIZE = _env("WEB_CHILD_CHUNK_SIZE", 380, int)
    WEB_PARENT_CHUNK_SIZE = _env("WEB_PARENT_CHUNK_SIZE", 950, int)
    ENABLE_SEMANTIC_MERGE = _env("ENABLE_SEMANTIC_MERGE", "true", bool)
    SEMANTIC_MERGE_SIM_THRESHOLD = _env("SEMANTIC_MERGE_SIM_THRESHOLD", 0.32, float)
    RETRIEVAL_CONTEXT_EXPANSION = _env("RETRIEVAL_CONTEXT_EXPANSION", "true", bool)
    
    # Web Scraping Configuration
    SCRAPING_TIMEOUT = _env("SCRAPING_TIMEOUT", 30, int)
    SCRAPING_MAX_RETRIES = _env("SCRAPING_MAX_RETRIES", 3, int)
    SCRAPING_USER_AGENT = _env("SCRAPING_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    # Auto-detect JS threshold: jika clean content < nilai ini (chars), coba Playwright
    AUTO_DETECT_MIN_CONTENT = _env("AUTO_DETECT_MIN_CONTENT", 300, int)
    # Playwright navigation timeout (milliseconds)
    PLAYWRIGHT_TIMEOUT = _env("PLAYWRIGHT_TIMEOUT", 30000, int)
    # Delay (detik) antar request ke domain yang sama (rate limiting)
    RATE_LIMIT_DELAY = _env("RATE_LIMIT_DELAY", 2.0, float)
    # Max retry attempts untuk webhook callback
    WEBHOOK_RETRY_ATTEMPTS = _env("WEBHOOK_RETRY_ATTEMPTS", 3, int)
    
    # Webhook Configuration (shared: document dan web scraping service)
    # Backward-compatible aliases:
    # - WEB_MANAJEMEN_BASE_URL -> DOCUMENT_CALLBACK_URL
    # - WEB_MANAJEMEN_CALLBACK_URL -> WEB_CALLBACK_URL
    DOCUMENT_CALLBACK_URL = _env("DOCUMENT_CALLBACK_URL") or _env("WEB_MANAJEMEN_BASE_URL", "")
    WEB_CALLBACK_URL = _env("WEB_CALLBACK_URL") or _env("WEB_MANAJEMEN_CALLBACK_URL", "")
    WEB_MANAJEMEN_API_KEY = _env("WEB_MANAJEMEN_API_KEY", "")
    
    # Logging
    LOG_LEVEL = _env("LOG_LEVEL", "INFO")
    LOG_DIR = _env("LOG_DIR", "logs")
    
    # CORS
    CORS_ORIGINS: List[str] = _env("CORS_ORIGINS", "*", list) or ["*"]

    # Internal API Key (autentikasi antar service via header X-API-Key)
    # Fail-closed: jika kosong, semua request selain allowlist ditolak 401.
    # Tidak mengubah struktur payload; key hanya dikirim via header.
    INTERNAL_API_KEY = _env("INTERNAL_API_KEY", "")

    # ============== OPTIMIZATION CONFIGS ==============
    
    # Gemini concurrency limiter (max parallel Gemini API calls)
    GEMINI_MAX_CONCURRENT = _env("GEMINI_MAX_CONCURRENT", 20, int)
    
    # Prompt cache TTL in seconds (how long DB prompt overrides are cached)
    PROMPT_CACHE_TTL = _env("PROMPT_CACHE_TTL", 300, int)
    
    # Early exit threshold for adaptive fan-out (orchestrator)
    EARLY_EXIT_THRESHOLD = _env("EARLY_EXIT_THRESHOLD", 0.92, float)
    
    # Model idle timeout in seconds (lazy load / idle unload)
    MODEL_IDLE_TIMEOUT = _env("MODEL_IDLE_TIMEOUT", 600, int)
    
    # Shared Embedding Service
    USE_SHARED_EMBEDDING = _env("USE_SHARED_EMBEDDING", "true", bool)
    SHARED_EMBEDDING_URL = _env("SHARED_EMBEDDING_URL", "http://localhost:5014")
    EMBEDDING_SERVICE_PORT = _env("EMBEDDING_SERVICE_PORT", 5014, int)
    EMBEDDING_THREAD_POOL_SIZE = _env("EMBEDDING_THREAD_POOL_SIZE", 2, int)
    LARGE_MODEL_IDLE_TIMEOUT = _env("LARGE_MODEL_IDLE_TIMEOUT", 1800, int)

    # ============== LIGHTRAG ADAPTER (v4) ==============
    RAG_SEARCH_ENGINE = _env("RAG_SEARCH_ENGINE", "legacy")  # legacy|lightrag|shadow
    LIGHTRAG_ADAPTER_URL = _env("LIGHTRAG_ADAPTER_URL", "http://localhost:5015")
    LIGHTRAG_ADAPTER_PORT = _env("LIGHTRAG_ADAPTER_PORT", 5015, int)
    LIGHTRAG_BASE_URL = _env("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621")
    LIGHTRAG_API_KEY = _env("LIGHTRAG_API_KEY", "")
    LIGHTRAG_WORKSPACE = _env("LIGHTRAG_WORKSPACE", "medan-main")
    LIGHTRAG_QUERY_MODE = _env("LIGHTRAG_QUERY_MODE", "mix")  # naive|local|global|hybrid|mix
    LIGHTRAG_TOP_K = _env("LIGHTRAG_TOP_K", 10, int)
    LIGHTRAG_RERANK_ENABLED = _env("LIGHTRAG_RERANK_ENABLED", "false", bool)
    LIGHTRAG_FALLBACK_TO_LEGACY = _env("LIGHTRAG_FALLBACK_TO_LEGACY", "true", bool)
    LIGHTRAG_INDEX_TEXT = _env("LIGHTRAG_INDEX_TEXT", "true", bool)
    LIGHTRAG_INDEX_DOCUMENT = _env("LIGHTRAG_INDEX_DOCUMENT", "true", bool)
    LIGHTRAG_INDEX_WEB = _env("LIGHTRAG_INDEX_WEB", "true", bool)
    LIGHTRAG_TIMEOUT_SEC = _env("LIGHTRAG_TIMEOUT_SEC", 120, int)
    LIGHTRAG_MAX_RETRIES = _env("LIGHTRAG_MAX_RETRIES", 3, int)


# Canonical configuration instance used across the codebase
config = Config()
