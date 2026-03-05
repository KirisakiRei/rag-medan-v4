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
        result = await self._scrape_with_httpx(url)

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
                    
            except httpx.TimeoutException:
                logger.warning(f"[SCRAPER] Timeout attempt {attempt}/{self.max_retries}: {url}")
                if attempt == self.max_retries:
                    raise Exception(f"Timeout setelah {self.max_retries} percobaan")
                await asyncio.sleep(2 ** attempt)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"[SCRAPER] HTTP error {e.response.status_code}: {url}")
                raise Exception(f"HTTP error: {e.response.status_code}")
                
            except Exception as e:
                logger.error(f"[SCRAPER] Error attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _scrape_with_playwright(
        self,
        url: str,
        wait_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scrape menggunakan Playwright (JavaScript rendering)."""
        from services.rag_web.js_renderer import js_renderer

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


# Singleton instance
scraper = Scraper()
