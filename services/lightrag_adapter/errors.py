"""
RAG Medan v4 - LightRAG Adapter — Custom Exceptions.

Sentralisasi error types untuk semua operasi LightRAG.
"""


class LightRAGError(Exception):
    """Base exception untuk semua operasi LightRAG."""
    pass


class LightRAGConnectionError(LightRAGError):
    """LightRAG Server tidak bisa dihubungi (unreachable / circuit open)."""
    pass


class LightRAGTimeoutError(LightRAGError):
    """Request ke LightRAG Server melebihi batas waktu."""
    pass


class LightRAGIndexError(LightRAGError):
    """Operasi indexing/sync gagal."""
    pass


class LightRAGSearchError(LightRAGError):
    """Operasi search/query gagal."""
    pass


class SourceMappingError(LightRAGError):
    """Gagal resolve source ID atau URI mapping."""
    pass
