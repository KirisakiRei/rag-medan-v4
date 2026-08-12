"""
RAG Medan v4 - Orchestrator — Search Provider Abstraction.

Strategy pattern untuk retrieval engine. Memungkinkan switching
antara Legacy RAG dan LightRAG tanpa mengubah flow Orchestrator.

Providers:
- LegacySearchProvider  : Wraps existing parallel fan-out + aggregation
- LightRAGSearchProvider: Delegates to LightRAG Adapter service
- ShadowSearchProvider  : Runs both, returns legacy, logs LightRAG for comparison

Pemilihan provider dikontrol oleh env var RAG_SEARCH_ENGINE:
- "legacy"  → LegacySearchProvider
- "lightrag"→ LightRAGSearchProvider (with optional legacy fallback)
- "shadow"  → ShadowSearchProvider (legacy to user, lightrag for evaluation)
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from config import config

logger = logging.getLogger("orchestrator.search_provider")


# ============== ABSTRACT BASE ==============

class SearchProvider(ABC):
    """Abstract search provider interface."""

    @abstractmethod
    async def search(
        self,
        normalized_question: str,
        user_question: str,
        wa_number: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute search dan return result dict.

        Args:
            normalized_question: Cleaned/normalized question.
            user_question: Original user question.
            wa_number: WhatsApp number.
            top_k: Number of results per service.

        Returns:
            Dict dengan minimal: status, engine, candidates/contexts, timing
        """
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check apakah provider ini operational."""
        pass


# ============== LEGACY PROVIDER ==============

class LegacySearchProvider(SearchProvider):
    """
    Wraps existing parallel fan-out + aggregation + relevance check.

    Ini adalah provider default yang mengimplementasikan
    logic dari search_handler.py yang sudah berjalan.
    """

    async def search(self, normalized_question, user_question, wa_number, top_k=3):
        from orchestrator.search_handler import parallel_search_services
        from orchestrator.aggregation import aggregate_and_sort_candidates

        service_results, parallel_duration, services_queried = (
            await parallel_search_services(
                normalized_question,
                user_question,
                wa_number,
                top_k=top_k,
            )
        )

        all_candidates = aggregate_and_sort_candidates(
            service_results,
            normalized_question,
        )

        return {
            "status": "success" if all_candidates else "no_results",
            "engine": "legacy",
            "candidates": all_candidates,
            "services_queried": services_queried,
            "timing": {
                "parallel_search_sec": round(parallel_duration, 3),
            },
        }

    async def is_healthy(self) -> bool:
        # Legacy selalu available selama services berjalan
        return True


# ============== LIGHTRAG PROVIDER ==============

class LightRAGSearchProvider(SearchProvider):
    """
    Delegates search ke LightRAG Adapter service.

    Menggunakan httpx via call_service (sama seperti existing pattern).
    Jika LightRAG gagal dan LIGHTRAG_FALLBACK_TO_LEGACY=true,
    otomatis fallback ke LegacySearchProvider.
    """

    def __init__(self):
        self.adapter_url = config.LIGHTRAG_ADAPTER_URL
        self._legacy_fallback = LegacySearchProvider()

    async def search(self, normalized_question, user_question, wa_number, top_k=3):
        from orchestrator.service_client import call_service

        try:
            result = await call_service(
                self.adapter_url,
                "/internal/search",
                "POST",
                {
                    "query": normalized_question,
                    "knowledge_base_id": config.LIGHTRAG_WORKSPACE,
                    "mode": config.LIGHTRAG_QUERY_MODE,
                    "top_k": config.LIGHTRAG_TOP_K,
                    "include_references": True,
                },
                timeout=float(config.LIGHTRAG_TIMEOUT_SEC),
            )

            # Jika adapter return error status, cek apakah perlu fallback
            if result.get("status") == "error" and config.LIGHTRAG_FALLBACK_TO_LEGACY:
                logger.warning(
                    f"[LR-PROVIDER] LightRAG returned error, "
                    f"falling back to legacy: {result.get('error', '?')[:100]}"
                )
                return await self._do_fallback(
                    normalized_question, user_question, wa_number, top_k,
                    reason=result.get("error", "lightrag_error"),
                )

            # Tag dengan engine identifier
            result["engine"] = "lightrag"
            return result

        except Exception as e:
            # call_service atau network error
            if config.LIGHTRAG_FALLBACK_TO_LEGACY:
                logger.warning(
                    f"[LR-PROVIDER] LightRAG adapter unreachable "
                    f"({type(e).__name__}), falling back to legacy"
                )
                return await self._do_fallback(
                    normalized_question, user_question, wa_number, top_k,
                    reason=str(e),
                )
            # Fallback dinonaktifkan — propagate error
            logger.error(f"[LR-PROVIDER] LightRAG failed, fallback disabled: {e}")
            return {
                "status": "error",
                "engine": "lightrag",
                "error": str(e),
                "candidates": [],
                "contexts": [],
                "timing": {},
            }

    async def _do_fallback(
        self,
        normalized_question: str,
        user_question: str,
        wa_number: str,
        top_k: int,
        reason: str = "",
    ) -> dict:
        """Jalankan legacy fallback dan tandai hasilnya."""
        result = await self._legacy_fallback.search(
            normalized_question, user_question, wa_number, top_k
        )
        result["status"] = "fallback"
        result["fallback_reason"] = reason
        return result

    async def is_healthy(self) -> bool:
        from orchestrator.service_client import call_service
        try:
            result = await call_service(
                self.adapter_url, "/health", "GET", timeout=5.0
            )
            return result.get("status") == "healthy"
        except Exception:
            return False


# ============== SHADOW PROVIDER ==============

class ShadowSearchProvider(SearchProvider):
    """
    Shadow mode: jalankan Legacy + LightRAG secara paralel.

    - Return Legacy result ke user (production traffic).
    - Log LightRAG result untuk comparison/evaluation.
    - Tidak mempengaruhi response ke client.

    Digunakan untuk Phase 1 — Shadow Mode benchmarking.
    """

    def __init__(self):
        self.legacy = LegacySearchProvider()
        self.lightrag = LightRAGSearchProvider()

    async def search(self, normalized_question, user_question, wa_number, top_k=3):
        # Jalankan kedua provider secara paralel
        legacy_task = asyncio.create_task(
            self.legacy.search(
                normalized_question, user_question, wa_number, top_k
            )
        )
        lightrag_task = asyncio.create_task(
            self.lightrag.search(
                normalized_question, user_question, wa_number, top_k
            )
        )

        # Tunggu legacy result (yang akan dikembalikan ke user)
        legacy_result = await legacy_task

        # Log comparison asynchronously (tidak blocking response)
        asyncio.create_task(
            self._log_comparison(
                normalized_question, legacy_result, lightrag_task
            )
        )

        # Tag sebagai shadow mode
        legacy_result["shadow_mode"] = True
        return legacy_result

    async def _log_comparison(
        self,
        query: str,
        legacy_result: Dict[str, Any],
        lightrag_task: asyncio.Task,
    ) -> None:
        """Log perbandingan hasil Legacy vs LightRAG."""
        try:
            lightrag_result = await lightrag_task

            legacy_count = len(legacy_result.get("candidates", []))
            lightrag_count = len(lightrag_result.get("contexts", []))
            lightrag_status = lightrag_result.get("status", "?")

            logger.info(
                f"[SHADOW] query='{query[:60]}' | "
                f"legacy_candidates={legacy_count} | "
                f"lightrag_candidates={lightrag_count} | "
                f"lightrag_status={lightrag_status}"
            )
        except Exception as e:
            logger.warning(f"[SHADOW] LightRAG comparison failed: {e}")

    async def is_healthy(self) -> bool:
        # Shadow healthy jika legacy healthy
        return await self.legacy.is_healthy()


# ============== FACTORY ==============

def create_search_provider() -> SearchProvider:
    """
    Factory: buat search provider berdasarkan RAG_SEARCH_ENGINE config.

    Returns:
        Instance dari SearchProvider subclass yang sesuai.
    """
    engine = config.RAG_SEARCH_ENGINE

    if engine == "lightrag":
        logger.info("Search engine: LightRAG (primary)")
        return LightRAGSearchProvider()
    elif engine == "shadow":
        logger.info("Search engine: Shadow (legacy primary + lightrag evaluation)")
        return ShadowSearchProvider()
    else:
        logger.info("Search engine: Legacy (default)")
        return LegacySearchProvider()
