import asyncio
import unittest
from unittest.mock import AsyncMock

from services.rag_web.scraper import Scraper, WebScrapeError


class WebScraperFallbackTests(unittest.TestCase):
    def test_auto_mode_falls_back_to_playwright_on_403(self):
        scraper = Scraper()
        scraper._scrape_with_httpx = AsyncMock(side_effect=WebScrapeError(
            "http_status_error",
            "Akses ditolak",
            detail="HTTP 403",
        ))
        scraper._scrape_with_playwright = AsyncMock(return_value={
            "url": "https://example.test",
            "raw_html": "<html>browser content</html>",
            "status_code": 200,
        })

        result = asyncio.run(scraper.scrape("https://example.test"))

        self.assertEqual(result["status_code"], 200)
        scraper._scrape_with_playwright.assert_awaited_once()

    def test_forced_httpx_does_not_fallback_on_403(self):
        scraper = Scraper()
        scraper._scrape_with_httpx = AsyncMock(side_effect=WebScrapeError(
            "http_status_error",
            "Akses ditolak",
            detail="HTTP 403",
        ))
        scraper._scrape_with_playwright = AsyncMock()

        with self.assertRaises(WebScrapeError):
            asyncio.run(scraper.scrape(
                "https://example.test",
                use_js_renderer=False,
            ))

        scraper._scrape_with_playwright.assert_not_awaited()

    def test_auto_mode_does_not_fallback_on_404(self):
        scraper = Scraper()
        scraper._scrape_with_httpx = AsyncMock(side_effect=WebScrapeError(
            "http_status_error",
            "Halaman tidak ditemukan",
            detail="HTTP 404",
        ))
        scraper._scrape_with_playwright = AsyncMock()

        with self.assertRaises(WebScrapeError):
            asyncio.run(scraper.scrape("https://example.test"))

        scraper._scrape_with_playwright.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
