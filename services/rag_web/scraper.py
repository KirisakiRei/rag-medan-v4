"""RAG Web Service - Scraper Module"""
import os
import sys
import logging
import asyncio
from typing import Dict, Any, Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

logger = logging.getLogger("rag_web.scraper")


class WebScrapeError(Exception):
    """Structured scraping exception to distinguish target vs system failures."""

    def __init__(
        self,
        code: str,
        user_message: str,
        *,
        source: str = "target_web",
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.source = source
        self.detail = detail or user_message


class Scraper:
    """Web scraper dengan retry logic, mendukung httpx dan Playwright."""
    
    def __init__(self):
        self.timeout = config.SCRAPING_TIMEOUT
        self.max_retries = config.SCRAPING_MAX_RETRIES
        self.headers = {
            "User-Agent": config.SCRAPING_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    
    async def scrape(
        self,
        url: str,
        use_js_renderer: Optional[bool] = None,
        wait_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scrape URL → raw HTML. use_js_renderer: True=Playwright, False=httpx, None=auto."""
        if use_js_renderer is True:
            logger.info(f"[SCRAPER] Mode: Playwright (forced) — {url}")
            return await self._scrape_with_playwright(url, wait_selector)

        logger.info(f"[SCRAPER] Mode: httpx — {url}")
        try:
            result = await self._scrape_with_httpx(url)
        except WebScrapeError as exc:
            # Auto mode: HTTP clients sering ditolak 401/403/429 oleh WAF/CDN,
            # sementara browser sungguhan tetap diizinkan. Coba Playwright
            # sekali sebelum menyatakan pipeline gagal.
            browser_fallback_statuses = {"HTTP 401", "HTTP 403", "HTTP 429"}
            should_try_browser = (
                use_js_renderer is None
                and exc.code == "http_status_error"
                and any(status in exc.detail for status in browser_fallback_statuses)
            )
            if not should_try_browser:
                raise

            logger.warning(
                f"[SCRAPER] httpx ditolak ({exc.detail}); "
                f"fallback ke Playwright — {url}"
            )
            return await self._scrape_with_playwright(url, wait_selector)

        if use_js_renderer is False:
            return result

        # auto-detect: import here to avoid circular import
        from services.rag_web.cleaner import cleaner
        from services.rag_web.js_renderer import js_renderer

        raw_html = result.get("raw_html", "")
        try:
            quick_text = cleaner.clean(raw_html, url)
            clean_len = len(quick_text.strip())
        except Exception:
            clean_len = len(raw_html) // 10  # estimasi kasar

        if js_renderer.detect_needs_js(
            url=url,
            raw_html=raw_html,
            clean_text_length=clean_len,
            min_content_chars=config.AUTO_DETECT_MIN_CONTENT
        ):
            logger.info(
                f"[SCRAPER] Auto-detect: konten httpx terlalu pendek ({clean_len} chars), "
                f"retry dengan Playwright..."
            )
            try:
                js_result = await self._scrape_with_playwright(url, wait_selector)
                return js_result
            except Exception as e:
                logger.warning(
                    f"[SCRAPER] Playwright gagal ({e}), pakai hasil httpx sebagai fallback"
                )
                return result

        return result

    async def _scrape_with_httpx(self, url: str) -> Dict[str, Any]:
        """Scrape menggunakan httpx dengan retry exponential backoff."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True
                ) as client:
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()
                    
                    logger.info(f"[SCRAPER] httpx success: {url} (HTTP {response.status_code})")
                    return {
                        "url": str(url),
                        "raw_html": response.text,
                        "status_code": response.status_code
                    }
                    
            except httpx.ConnectTimeout as e:
                logger.warning(f"[SCRAPER] Timeout attempt {attempt}/{self.max_retries}: {url}")
                last_error = e
                if attempt == self.max_retries:
                    raise WebScrapeError(
                        "connect_timeout",
                        f"Koneksi ke server target timeout setelah {self.timeout} detik",
                        source="target_web",
                        detail=str(e),
                    )
                await asyncio.sleep(2 ** attempt)

            except httpx.ReadTimeout as e:
                logger.warning(f"[SCRAPER] Timeout attempt {attempt}/{self.max_retries}: {url}")
                last_error = e
                if attempt == self.max_retries:
                    raise WebScrapeError(
                        "read_timeout",
                        f"Server target tidak merespons dalam {self.timeout} detik",
                        source="target_web",
                        detail=str(e),
                    )
                await asyncio.sleep(2 ** attempt)

            except httpx.TimeoutException as e:
                logger.warning(f"[SCRAPER] Timeout attempt {attempt}/{self.max_retries}: {url}")
                last_error = e
                if attempt == self.max_retries:
                    raise WebScrapeError(
                        "timeout",
                        f"Akses ke server target timeout setelah {self.timeout} detik",
                        source="target_web",
                        detail=str(e),
                    )
                await asyncio.sleep(2 ** attempt)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"[SCRAPER] HTTP error {e.response.status_code}: {url}")
                status = e.response.status_code
                if status == 403:
                    user_message = "Akses ke halaman ditolak oleh server target (HTTP 403)"
                elif status == 404:
                    user_message = "Halaman target tidak ditemukan (HTTP 404)"
                elif 500 <= status <= 599:
                    user_message = f"Server target mengalami gangguan (HTTP {status})"
                else:
                    user_message = f"Gagal mengakses halaman target (HTTP {status})"
                raise WebScrapeError(
                    "http_status_error",
                    user_message,
                    source="target_web",
                    detail=f"HTTP {status}",
                )

            except httpx.ConnectError as e:
                logger.error(f"[SCRAPER] Connect error attempt {attempt}/{self.max_retries}: {e}")
                last_error = e
                if attempt == self.max_retries:
                    raise WebScrapeError(
                        "connect_error",
                        "Koneksi ke server target gagal dibuka",
                        source="target_web",
                        detail=str(e),
                    )
                await asyncio.sleep(2 ** attempt)

            except httpx.RequestError as e:
                logger.error(f"[SCRAPER] Request error attempt {attempt}/{self.max_retries}: {e}")
                last_error = e
                if attempt == self.max_retries:
                    raise WebScrapeError(
                        "request_error",
                        "Permintaan ke server target gagal diproses",
                        source="target_web",
                        detail=str(e),
                    )
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"[SCRAPER] Error attempt {attempt}/{self.max_retries}: {e}")
                last_error = e
                if attempt == self.max_retries:
                    raise WebScrapeError(
                        "scraper_internal_error",
                        "Terjadi kesalahan internal saat mengambil halaman web",
                        source="rag_system",
                        detail=str(e),
                    )
                await asyncio.sleep(2 ** attempt)

        raise WebScrapeError(
            "unknown_scrape_error",
            "Terjadi kesalahan yang tidak diketahui saat mengambil halaman web",
            source="rag_system",
            detail=str(last_error) if last_error else None,
        )

    async def _scrape_with_playwright(
        self,
        url: str,
        wait_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scrape menggunakan Playwright (JavaScript rendering)."""
        from services.rag_web.js_renderer import js_renderer

        try:
            raw_html = await js_renderer.render(
                url=url,
                wait_selector=wait_selector,
                timeout_ms=config.PLAYWRIGHT_TIMEOUT,
            )
            logger.info(f"[SCRAPER] Playwright success: {url} ({len(raw_html)} chars)")
            return {
                "url": str(url),
                "raw_html": raw_html,
                "status_code": 200  # Playwright tidak expose status code secara langsung
            }
        except WebScrapeError:
            raise
        except Exception as e:
            detail = str(e)
            lower_detail = detail.lower()
            source = "rag_system" if any(
                token in lower_detail for token in ["browser", "playwright", "executable", "install"]
            ) else "target_web"
            user_message = (
                "Renderer JavaScript internal gagal dijalankan"
                if source == "rag_system"
                else "Halaman JavaScript target gagal dirender"
            )
            raise WebScrapeError(
                "playwright_render_error",
                user_message,
                source=source,
                detail=detail,
            )


# Singleton instance
scraper = Scraper()
