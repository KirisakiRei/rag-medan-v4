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
    DEBUG = _env("DEBUG", "false", bool)
    
    # Service Ports (internal communication)
    ORCHESTRATOR_PORT = _env("ORCHESTRATOR_PORT", 5100, int)
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
    
    # Aliases for backward compatibility
    EMB_MODEL_PATH = EMBEDDING_MODEL_PATH
    EMB_LARGE_PATH = EMBEDDING_MODEL_PATH_LARGE
    EMB_DIM_SMALL = EMBEDDING_DIMENSION
    EMB_DIM_LARGE = EMBEDDING_DIMENSION_LARGE
    
    # Qdrant Configuration
    QDRANT_HOST = _env("QDRANT_HOST", "localhost")
    QDRANT_PORT = _env("QDRANT_PORT", 6333, int)
    QDRANT_API_KEY = _env("QDRANT_API_KEY", None)
    
    # Collection Names
    COLLECTION_TEXT = _env("COLLECTION_TEXT", "knowledge_bank")
    COLLECTION_DOCUMENT = _env("COLLECTION_DOCUMENT", "document_bank")
    COLLECTION_WEB = _env("COLLECTION_WEB", "web_scraping_bank")
    COLLECTION_USULAN = _env("COLLECTION_USULAN", "usulan_bank")
    
    # LLM Configuration (Gemini)
    LLM_BASE_URL = _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")
    LLM_API_KEY = _env("LLM_API_KEY", "")
    LLM_MODEL = _env("LLM_MODEL", "gemini-2.0-flash")
    LLM_TIMEOUT = _env("LLM_TIMEOUT_SEC", 60, int)
    LLM_PROVIDER = _env("LLM_PROVIDER", "gemini")
    
    # Database Configuration
    DB_HOST = _env("DB_HOST", "localhost")
    DB_PORT = _env("DB_PORT", 3306, int)
    DB_DATABASE = _env("DB_DATABASE", "")
    DB_USERNAME = _env("DB_USERNAME", "")
    DB_PASSWORD = _env("DB_PASSWORD", "")
    
    # OCR Configuration
    OCR_ENGINE = _env("OCR_ENGINE", "paddle")
    OCR_LANG = _env("OCR_LANG", "id")
    OCR_TIMEOUT = _env("OCR_TIMEOUT", 1800, int)  # default 30 menit
    OCR_STALL_TIMEOUT = _env("OCR_STALL_TIMEOUT", 300, int)
    OCR_HARD_TIMEOUT = _env("OCR_HARD_TIMEOUT", 21600, int)
    OCR_QUEUE_LOG_INTERVAL = _env("OCR_QUEUE_LOG_INTERVAL", 30, int)
    OCR_PDF_DPI = _env("OCR_PDF_DPI", 150, int)
    OCR_PDF_DPI_RETRY = _env("OCR_PDF_DPI_RETRY", 200, int)
    OCR_DOWNLOAD_PROGRESS_MB = _env("OCR_DOWNLOAD_PROGRESS_MB", 5, int)

    # RAG Configuration
    USE_POST_SUMMARY = _env("USE_POST_SUMMARY", "false", bool)
    POST_SUMMARY_TOP_K = _env("POST_SUMMARY_TOP_K", 2, int)
    
    # Search Thresholds
    TEXT_DENSE_THRESHOLD = _env("TEXT_DENSE_THRESHOLD", 0.83, float)
    TEXT_OVERLAP_THRESHOLD = _env("TEXT_OVERLAP_THRESHOLD", 0.15, float)
    DOCUMENT_SCORE_THRESHOLD = _env("DOCUMENT_SCORE_THRESHOLD", 0.4, float)
    WEB_SCORE_THRESHOLD = _env("WEB_SCORE_THRESHOLD", 0.5, float)
    USULAN_SCORE_THRESHOLD = _env("USULAN_SCORE_THRESHOLD", 0.85, float)
    
    # Reranking Configuration
    RERANK_TOP_K = _env("RERANK_TOP_K", 5, int)
    RERANK_WEIGHT_TEXT = _env("RERANK_WEIGHT_TEXT", 0.4, float)
    RERANK_WEIGHT_DOC = _env("RERANK_WEIGHT_DOC", 0.35, float)
    RERANK_WEIGHT_WEB = _env("RERANK_WEIGHT_WEB", 0.25, float)
    
    # Chunking Configuration (Document & Web)
    CHUNK_SIZE = _env("CHUNK_SIZE", 1200, int)
    CHUNK_OVERLAP = _env("CHUNK_OVERLAP", 150, int)
    
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


# Singleton instance
config = Config()

# Legacy CONFIG dict for backward compatibility
CONFIG = {
    "api": {
        "host": config.API_HOST,
        "port": config.ORCHESTRATOR_PORT
    },
    "embeddings": {
        "model_path": config.EMBEDDING_MODEL_PATH,
        "model_path_large": config.EMBEDDING_MODEL_PATH_LARGE
    },
    "qdrant": {
        "host": config.QDRANT_HOST,
        "port": config.QDRANT_PORT
    },
    "llm": {
        "base_url": config.LLM_BASE_URL,
        "api_key": config.LLM_API_KEY,
        "model": config.LLM_MODEL,
        "timeout_sec": config.LLM_TIMEOUT,
        "provider": config.LLM_PROVIDER
    },
    "db": {
        "host": config.DB_HOST,
        "port": config.DB_PORT,
        "database": config.DB_DATABASE,
        "username": config.DB_USERNAME,
        "password": config.DB_PASSWORD
    },
    "ocr": {
        "engine": config.OCR_ENGINE,
        "lang": config.OCR_LANG,
        "timeout": config.OCR_TIMEOUT
    },
    "rag": {
        "use_post_summary": config.USE_POST_SUMMARY,
        "post_summary_top_k": config.POST_SUMMARY_TOP_K
    }
}
