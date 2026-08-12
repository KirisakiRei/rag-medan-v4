"""
RAG Medan v4 - LightRAG Adapter — HTTP Client for LightRAG Server.

Menyediakan async HTTP client dengan:
- Connection pooling (httpx.AsyncClient)
- Retry dengan exponential backoff (1s → 2s → 4s)
- Circuit breaker (5 consecutive failures → open 30s → half-open probe)
- Health probe caching (interval 10s)
- Sanitized logging (no secrets, no full bodies)

Semua komunikasi dengan LightRAG Server melewati module ini.
"""
import asyncio
import time
import logging
from typing import Optional, Dict, Any, List

import httpx

from services.lightrag_adapter.config import adapter_config
from services.lightrag_adapter.errors import (
    LightRAGConnectionError,
    LightRAGTimeoutError,
    LightRAGSearchError,
)

logger = logging.getLogger("lightrag_adapter.client")


# ============== CIRCUIT BREAKER ==============

class CircuitBreaker:
    """
    Simple circuit breaker: closed → open → half_open → closed.

    States:
    - CLOSED    : Normal, semua request diizinkan.
    - OPEN      : Setelah N failure berturut-turut, block semua request.
    - HALF_OPEN : Setelah recovery_timeout, izinkan 1 probe request.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self.state = self.CLOSED

    def record_success(self) -> None:
        """Reset counter dan tutup circuit."""
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self) -> None:
        """Increment counter; buka circuit jika threshold tercapai."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning(
                f"[CIRCUIT] Opened after {self.failure_count} consecutive failures"
            )

    def allow_request(self) -> bool:
        """Check apakah request diizinkan."""
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = self.HALF_OPEN
                logger.info("[CIRCUIT] Half-open, allowing probe request")
                return True
            return False
        # HALF_OPEN — izinkan satu probe
        return True


# ============== LIGHT RAG CLIENT ==============

class LightRAGClient:
    """
    Async HTTP client untuk LightRAG Server.

    Usage:
        client = LightRAGClient()
        await client.start()
        result = await client.query("pertanyaan", mode="mix")
        await client.stop()
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
        )
        self._healthy: bool = False
        self._last_health_check: float = 0.0

    @property
    def base_url(self) -> str:
        return adapter_config.BASE_URL.rstrip("/")

    @property
    def headers(self) -> Dict[str, str]:
        """Build request headers — menggunakan X-API-Key (sama dengan existing)."""
        h = {"Content-Type": "application/json"}
        if adapter_config.API_KEY:
            h["X-API-Key"] = adapter_config.API_KEY
        return h

    # ============== LIFECYCLE ==============

    async def start(self) -> None:
        """Initialize HTTP client (dipanggil saat lifespan startup)."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
            timeout=httpx.Timeout(adapter_config.TIMEOUT_SEC),
        )
        logger.info(f"LightRAG client initialized: {self.base_url}")

    async def stop(self) -> None:
        """Close HTTP client (dipanggil saat lifespan shutdown)."""
        if self._client:
            await self._client.aclose()
            logger.info("LightRAG client closed")

    # ============== CORE REQUEST ==============

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute HTTP request dengan retry + circuit breaker.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path (e.g. "/query")
            json_data: Request body (JSON)
            timeout: Override timeout untuk request ini

        Returns:
            Parsed JSON response dict.

        Raises:
            LightRAGConnectionError: Client belum siap atau server unreachable.
            LightRAGSearchError: Server mengembalikan 4xx client error.
        """
        if self._client is None:
            raise LightRAGConnectionError(
                "LightRAG HTTP client belum diinisialisasi. "
                "Pastikan start() dipanggil saat lifespan startup."
            )

        if not self._circuit.allow_request():
            raise LightRAGConnectionError(
                "Circuit breaker is OPEN — LightRAG server deemed unavailable"
            )

        last_error: Optional[Exception] = None

        for attempt in range(1, adapter_config.MAX_RETRIES + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=json_data,
                    timeout=timeout or adapter_config.TIMEOUT_SEC,
                )
                response.raise_for_status()
                self._circuit.record_success()

                logger.debug(
                    f"[LR-HTTP] {method} {path} → {response.status_code} "
                    f"(attempt={attempt})"
                )
                return response.json()

            except httpx.ConnectError as e:
                last_error = e
                self._circuit.record_failure()
                logger.warning(
                    f"[LR-HTTP] Connection refused: {path} "
                    f"(attempt={attempt}/{adapter_config.MAX_RETRIES})"
                )

            except httpx.TimeoutException as e:
                last_error = e
                self._circuit.record_failure()
                logger.warning(
                    f"[LR-HTTP] Timeout: {path} "
                    f"(attempt={attempt}/{adapter_config.MAX_RETRIES})"
                )

            except httpx.HTTPStatusError as e:
                # 4xx = client error, jangan retry
                if 400 <= e.response.status_code < 500:
                    self._circuit.record_success()  # Server reachable
                    body = (e.response.text or "")[:300]
                    logger.error(
                        f"[LR-HTTP] Client error {e.response.status_code}: {body}"
                    )
                    raise LightRAGSearchError(
                        f"LightRAG returned {e.response.status_code}: {body}"
                    ) from e
                # 5xx = server error, retry
                last_error = e
                self._circuit.record_failure()
                logger.warning(
                    f"[LR-HTTP] Server error {e.response.status_code}: {path} "
                    f"(attempt={attempt}/{adapter_config.MAX_RETRIES})"
                )

            except Exception as e:
                last_error = e
                self._circuit.record_failure()
                logger.error(f"[LR-HTTP] Unexpected error: {path}: {e}")

            # Exponential backoff: 1s, 2s, 4s
            if attempt < adapter_config.MAX_RETRIES:
                backoff = 2 ** (attempt - 1)
                logger.info(f"[LR-HTTP] Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        raise LightRAGConnectionError(
            f"LightRAG unreachable after {adapter_config.MAX_RETRIES} attempts: "
            f"{last_error}"
        )

    # ============== HIGH-LEVEL API ==============

    async def query(
        self,
        query_text: str,
        mode: str = "mix",
        top_k: int = 10,
        include_references: bool = True,
        include_chunk_content: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute search query terhadap LightRAG Server.

        POST /query

        include_chunk_content=True agar `references[].content` berisi teks
        chunk asli (bukan hanya metadata file). Tanpa ini jawaban tidak
        bisa diverifikasi / ditampilkan sumbernya.
        """
        payload = {
            "query": query_text,
            "mode": mode,
            "top_k": top_k,
            "include_references": include_references,
            "include_chunk_content": include_chunk_content,
            "enable_rerank": adapter_config.RERANK_ENABLED,
        }
        return await self._request("POST", "/query", json_data=payload)

    async def query_context_only(
        self,
        query_text: str,
        mode: str = "mix",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute search dan return HANYA context tanpa LLM generation.

        Berguna untuk Phase 1 dimana LightRAG hanya sebagai retriever,
        dan answer generation tetap di existing pipeline.

        POST /query dengan response_type=multiple
        """
        payload = {
            "query": query_text,
            "mode": mode,
            "top_k": top_k,
            "response_type": "multiple",
            "include_references": True,
            "enable_rerank": adapter_config.RERANK_ENABLED,
        }
        return await self._request("POST", "/query", json_data=payload)

    async def insert_text(
        self,
        text: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Insert text document ke LightRAG index.

        POST /documents/text
        """
        payload: Dict[str, Any] = {"text": text}
        if description:
            payload["description"] = description
        return await self._request("POST", "/documents/text", json_data=payload)

    async def insert_texts_batch(
        self,
        texts: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Insert multiple text documents sekaligus.

        POST /documents/texts
        Each item: {"text": "...", "description": "..."}
        """
        payload = {"texts": texts}
        return await self._request("POST", "/documents/texts", json_data=payload)

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete document dari LightRAG index.

        DELETE /documents/{document_id}
        """
        return await self._request("DELETE", f"/documents/{document_id}")

    async def get_documents_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Get paginated document list.

        GET /documents/paginated
        """
        return await self._request(
            "GET",
            f"/documents/paginated?page={page}&page_size={page_size}",
        )

    async def check_health(self) -> bool:
        """
        Check apakah LightRAG Server healthy.

        Hasil di-cache selama HEALTH_CHECK_INTERVAL detik untuk
        menghindari health check yang terlalu sering.

        Returns:
            True jika server healthy, False jika tidak atau client belum siap.
        """
        if self._client is None:
            # Client belum diinisialisasi — bukan error, hanya belum siap
            self._healthy = False
            return False

        now = time.time()
        if (now - self._last_health_check) < adapter_config.HEALTH_CHECK_INTERVAL:
            return self._healthy

        try:
            response = await self._client.get("/health", timeout=5.0)
            self._healthy = response.status_code == 200
        except Exception:
            self._healthy = False

        self._last_health_check = now
        return self._healthy

    @property
    def is_circuit_open(self) -> bool:
        """Check apakah circuit breaker sedang terbuka."""
        return self._circuit.state == CircuitBreaker.OPEN

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state."""
        return self._circuit.state


# Singleton instance — diinitialize di main.py lifespan
lightrag_client = LightRAGClient()
