"""RAG Web Service - Playwright Chromium renderer for SPA/JS-heavy pages."""
import re
import logging
import asyncio
from typing import Optional

logger = logging.getLogger("rag_web.js_renderer")

# SPA marker patterns di raw HTML yang mengindikasikan halaman butuh JS
_SPA_ROOT_PATTERN = re.compile(
    r'<(div|section)\s[^>]*(id|class)=["\'][^"\']*\b(app|root|__next|__nuxt|vue-app)\b',
    re.IGNORECASE,
)
_BUNDLE_SCRIPT_PATTERN = re.compile(
    r'<script[^>]+src=["\'][^"\']*\b(chunk|bundle|vendor|main|app)\b[^"\']*\.js',
    re.IGNORECASE,
)
_GENERATOR_PATTERN = re.compile(
    r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_KNOWN_SPA_GENERATORS = {"next.js", "nuxt", "gatsby", "create react app", "vite", "angular"}


class JSRenderer:
    """Playwright-based JS renderer, singleton Chromium browser."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._lock = asyncio.Lock()

    async def _get_browser(self):
        """Lazy singleton: launch Chromium jika belum ada."""
        if self._browser is not None:
            return self._browser

        async with self._lock:
            if self._browser is not None:
                return self._browser

            try:
                from playwright.async_api import async_playwright

                logger.info("[JS_RENDERER] Launching Chromium browser (one-time)...")
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--disable-background-networking",
                        "--disable-default-apps",
                        "--mute-audio",
                    ],
                )
                logger.info("[JS_RENDERER] Chromium browser launched successfully")
            except ImportError:
                raise RuntimeError(
                    "Playwright tidak terinstall. Jalankan: pip install playwright && playwright install chromium"
                )
            except Exception as e:
                logger.error(f"[JS_RENDERER] Gagal launch browser: {e}")
                raise

        return self._browser

    async def render(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> str:
        """Render URL dengan Playwright, kembalikan HTML setelah JS dieksekusi."""
        browser = await self._get_browser()
        page = None

        try:
            page = await browser.new_page()

            await page.set_extra_http_headers({
                "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            })

            logger.info(f"[JS_RENDERER] Rendering: {url}")
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)

            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=5000)
                    logger.debug(f"[JS_RENDERER] wait_selector '{wait_selector}' ditemukan")
                except Exception:
                    logger.warning(
                        f"[JS_RENDERER] wait_selector '{wait_selector}' tidak muncul dalam 5 detik, "
                        "tetap lanjut ambil HTML"
                    )

            html = await page.content()
            logger.info(f"[JS_RENDERER] Render selesai: {len(html)} chars dari {url}")
            return html

        except Exception as e:
            logger.error(f"[JS_RENDERER] Error rendering {url}: {e}")
            raise
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass

    def detect_needs_js(
        self,
        url: str,
        raw_html: str,
        clean_text_length: int,
        min_content_chars: int = 300,
    ) -> bool:
        """Heuristik deteksi apakah halaman butuh JS rendering."""
        # Cek 1: Konten terlalu pendek
        if clean_text_length < min_content_chars:
            logger.info(
                f"[JS_RENDERER] detect_needs_js=True: konten pendek "
                f"({clean_text_length} < {min_content_chars} chars)"
            )
            return True

        # Cek 2: SPA root container
        if _SPA_ROOT_PATTERN.search(raw_html):
            logger.info("[JS_RENDERER] detect_needs_js=True: SPA root div ditemukan")
            return True

        # Cek 3: bundle script + konten relatif pendek
        if _BUNDLE_SCRIPT_PATTERN.search(raw_html):
            if clean_text_length < min_content_chars * 2:
                logger.info("[JS_RENDERER] detect_needs_js=True: bundle script + konten relatif pendek")
                return True

        # Cek 4: Meta generator SPA
        match = _GENERATOR_PATTERN.search(raw_html)
        if match:
            generator = match.group(1).lower()
            if any(spa in generator for spa in _KNOWN_SPA_GENERATORS):
                logger.info(f"[JS_RENDERER] detect_needs_js=True: generator={match.group(1)}")
                return True

        return False

    async def close(self):
        """Tutup browser dan Playwright instance. Dipanggil saat service shutdown."""
        if self._browser:
            try:
                await self._browser.close()
                logger.info("[JS_RENDERER] Browser closed")
            except Exception as e:
                logger.warning(f"[JS_RENDERER] Error saat close browser: {e}")
            finally:
                self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
                logger.info("[JS_RENDERER] Playwright stopped")
            except Exception as e:
                logger.warning(f"[JS_RENDERER] Error saat stop playwright: {e}")
            finally:
                self._playwright = None


# Singleton instance
js_renderer = JSRenderer()
