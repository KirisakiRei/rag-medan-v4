"""RAG Web Service - Per-domain async rate limiter."""
import asyncio
import time
import logging
from urllib.parse import urlparse
from typing import Dict

logger = logging.getLogger("rag_web.rate_limiter")


class DomainRateLimiter:
    """Rate limiter per domain dengan asyncio Lock."""

    def __init__(self):
        # Dict: domain → timestamp request terakhir
        self._last_request: Dict[str, float] = {}
        # Dict: domain → asyncio.Lock (satu lock per domain)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()  # Untuk proteksi akses ke _locks dict itu sendiri

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        """Get atau create Lock untuk domain tertentu (thread-safe)."""
        if domain not in self._locks:
            async with self._meta_lock:
                if domain not in self._locks:
                    self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def wait_for_domain(self, url: str, delay: float = None) -> None:
        """Tunggu delay minimum sebelum request ke domain yang sama."""
        from config import config as app_config

        effective_delay = delay if delay is not None else app_config.RATE_LIMIT_DELAY

        try:
            parsed = urlparse(url)
            domain = parsed.netloc or url
        except Exception:
            domain = url

        lock = await self._get_lock(domain)

        async with lock:
            last = self._last_request.get(domain, 0.0)
            elapsed = time.time() - last

            if elapsed < effective_delay:
                wait_time = effective_delay - elapsed
                logger.debug(
                    f"[RATE_LIMITER] Domain '{domain}' — menunggu {wait_time:.2f}s "
                    f"(last request {elapsed:.2f}s lalu)"
                )
                await asyncio.sleep(wait_time)

            self._last_request[domain] = time.time()

    def get_last_request_time(self, url: str) -> float:
        """Kembalikan timestamp request terakhir ke domain URL ini (0 jika belum pernah)."""
        try:
            domain = urlparse(url).netloc or url
        except Exception:
            domain = url
        return self._last_request.get(domain, 0.0)

    def reset_domain(self, url: str) -> None:
        """Reset tracking untuk domain tertentu (untuk testing/debugging)."""
        try:
            domain = urlparse(url).netloc or url
        except Exception:
            domain = url
        self._last_request.pop(domain, None)
        logger.debug(f"[RATE_LIMITER] Reset domain: {domain}")


# Singleton instance
rate_limiter = DomainRateLimiter()
