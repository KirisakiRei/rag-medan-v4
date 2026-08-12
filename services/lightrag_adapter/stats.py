"""
RAG Medan v4 - LightRAG Adapter — In-memory Statistics.

Mencatat hitungan operasi (query, sync, delete, error) + latency rata-rata.
Data in-memory per proses; reset saat restart service.
"""
import time
import logging
from typing import Dict, Any

logger = logging.getLogger("lightrag_adapter.stats")


class Stats:
    def __init__(self) -> None:
        self._started_at = time.time()
        self._query_count = 0
        self._query_error_count = 0
        self._query_total_sec = 0.0
        self._sync_counts: Dict[str, int] = {"text": 0, "document": 0, "web": 0}
        self._sync_error_counts: Dict[str, int] = {"text": 0, "document": 0, "web": 0}
        self._delete_counts: Dict[str, int] = {"text": 0, "document": 0, "web": 0}

    def record_query(self, duration_sec: float) -> None:
        self._query_count += 1
        self._query_total_sec += duration_sec

    def record_query_error(self) -> None:
        self._query_error_count += 1

    def record_sync(self, source_type: str, ok: bool) -> None:
        if ok:
            self._sync_counts[source_type] = self._sync_counts.get(source_type, 0) + 1
        else:
            self._sync_error_counts[source_type] = self._sync_error_counts.get(source_type, 0) + 1

    def record_delete(self, source_type: str) -> None:
        self._delete_counts[source_type] = self._delete_counts.get(source_type, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        avg_query_sec = (
            round(self._query_total_sec / self._query_count, 3)
            if self._query_count
            else 0.0
        )
        return {
            "uptime_sec": round(time.time() - self._started_at, 1),
            "query": {
                "count": self._query_count,
                "error_count": self._query_error_count,
                "avg_sec": avg_query_sec,
            },
            "sync": {
                "success": dict(self._sync_counts),
                "error": dict(self._sync_error_counts),
            },
            "delete": dict(self._delete_counts),
        }


stats = Stats()
