"""
RAG Medan v4 - LightRAG Adapter — Configuration.

Menggunakan pola yang sama dengan config.py utama (class-based + _env helper).
Mengambil beberapa nilai dari app_config untuk konsistensi.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config as app_config


def _env(key, default=None, cast=str):
    """Environment variable reader — konsisten dengan config.py utama."""
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
    return val


class LightRAGAdapterConfig:
    """
    Konfigurasi LightRAG Adapter service.

    Menggabungkan:
    - Variable LIGHTRAG_* spesifik adapter
    - Reuse nilai dari app_config (TEXT_SERVICE_URL, dll) untuk konsistensi
    """

    # === Server ===
    PORT = _env("LIGHTRAG_ADAPTER_PORT", 5015, int)

    # === LightRAG Server Connection ===
    BASE_URL = _env("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621")
    API_KEY = _env("LIGHTRAG_API_KEY", "")
    WORKSPACE = _env("LIGHTRAG_WORKSPACE", "medan-main")

    # === Query ===
    QUERY_MODE = _env("LIGHTRAG_QUERY_MODE", "mix")       # naive|local|global|hybrid|mix
    TOP_K = _env("LIGHTRAG_TOP_K", 10, int)
    RERANK_ENABLED = _env("LIGHTRAG_RERANK_ENABLED", "false", bool)

    # === Feature Flags ===
    FALLBACK_TO_LEGACY = _env("LIGHTRAG_FALLBACK_TO_LEGACY", "true", bool)
    INDEX_TEXT = _env("LIGHTRAG_INDEX_TEXT", "true", bool)
    INDEX_DOCUMENT = _env("LIGHTRAG_INDEX_DOCUMENT", "true", bool)
    INDEX_WEB = _env("LIGHTRAG_INDEX_WEB", "true", bool)

    # === Client Tuning ===
    TIMEOUT_SEC = _env("LIGHTRAG_TIMEOUT_SEC", 120.0, float)
    MAX_RETRIES = _env("LIGHTRAG_MAX_RETRIES", 3, int)
    HEALTH_CHECK_INTERVAL = _env("LIGHTRAG_HEALTH_INTERVAL", 10, int)
    INDEX_TIMEOUT_SEC = _env("LIGHTRAG_INDEX_TIMEOUT_SEC", 600.0, float)
    INDEX_POLL_INTERVAL_SEC = _env("LIGHTRAG_INDEX_POLL_INTERVAL_SEC", 2.0, float)

    # === Legacy Fallback URLs (reuse dari app_config) ===
    LEGACY_TEXT_URL = app_config.TEXT_SERVICE_URL
    LEGACY_DOCUMENT_URL = app_config.DOCUMENT_SERVICE_URL
    LEGACY_WEB_URL = app_config.WEB_SERVICE_URL

    # === Embedding (reuse dari app_config) ===
    EMBEDDING_DIM = app_config.EMBEDDING_DIMENSION


# Singleton instance
adapter_config = LightRAGAdapterConfig()
