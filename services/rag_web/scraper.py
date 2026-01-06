"""
RAG Web Service - Scraper Module
Web scraping dengan httpx
"""
import os
import sys
import logging
import asyncio
from typing import Dict, Any

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

logger = logging.getLogger("rag_web.scraper")


class Scraper:
    """Web scraper dengan retry logic."""
    
    def __init__(self):
        self.timeout = config.SCRAPING_TIMEOUT
        self.max_retries = config.SCRAPING_MAX_RETRIES
        self.headers = {
            "User-Agent": config.SCRAPING_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    
    async def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape URL dengan retry.
        
        Returns:
            Dict dengan url, raw_html, status_code
        """
        logger.info(f"[SCRAPER] Scraping: {url}")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True
                ) as client:
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()
                    
                    logger.info(f"[SCRAPER] Success: {url} ({response.status_code})")
                    
                    return {
                        "url": str(url),
                        "raw_html": response.text,
                        "status_code": response.status_code
                    }
                    
            except httpx.TimeoutException:
                logger.warning(f"[SCRAPER] Timeout attempt {attempt}/{self.max_retries}")
                if attempt == self.max_retries:
                    raise Exception(f"Timeout after {self.max_retries} attempts")
                await asyncio.sleep(2 ** attempt)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"[SCRAPER] HTTP error: {e.response.status_code}")
                raise Exception(f"HTTP error: {e.response.status_code}")
                
            except Exception as e:
                logger.error(f"[SCRAPER] Error: {e}")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)


# Singleton instance
scraper = Scraper()
