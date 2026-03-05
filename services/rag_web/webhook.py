"""RAG Web Service - Webhook callback for web-banks scrape status."""
import asyncio
import logging
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger("rag_web.webhook")

_STATUS_MAP = {
    "scraping": "scraping",
    "completed": "scraped",
    "failed": "failed",
    "skipped": "scraped",
}


def _build_message(status: str, result: Dict[str, Any]) -> str:
    """Bangun scrape_message berdasarkan status dan result."""
    if status == "scraping":
        return "Sedang mengambil konten dari halaman web..."
    if status == "completed":
        chunks = result.get("chunks_count", 0)
        content_type = result.get("content_type", "general")
        faq_count = result.get("faq_pairs_count")
        if faq_count:
            return f"Halaman berhasil di-scrape. {faq_count} pasangan Q&A, {chunks} chunk ({content_type})"
        return f"Halaman berhasil di-scrape. {chunks} chunk ({content_type})"
    if status == "failed":
        error = result.get("error", "unknown error")
        return f"Gagal memproses halaman web: {error}"
    if status == "skipped":
        return "Konten sudah tersedia dan tidak di-scrape ulang. Gunakan force_rescrape=True untuk memperbarui."
    return ""


async def send_callback(
    link_id: str,
    url: str,
    status: str,
    result: Dict[str, Any],
    callback_url: Optional[str] = None,
) -> bool:
    """Kirim webhook PUT ke web-banks/scrape-status/{link_id}. Return True jika berhasil."""
    from config import config as app_config

    web_callback_url = app_config.WEB_CALLBACK_URL
    if not web_callback_url:
        logger.debug("[WEBHOOK] WEB_CALLBACK_URL tidak dikonfigurasi, skip")
        return False

    target_url = f"{web_callback_url.rstrip('/')}/{link_id}"
    payload = {
        "scrape_status": _STATUS_MAP.get(status, "failed"),
        "scrape_message": _build_message(status, result),
    }

    headers = {"Content-Type": "application/json"}
    if app_config.WEB_MANAJEMEN_API_KEY:
        headers["X-API-Key"] = app_config.WEB_MANAJEMEN_API_KEY

    max_retries = app_config.WEBHOOK_RETRY_ATTEMPTS
    delays = [2 ** i for i in range(max_retries)]

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(target_url, json=payload, headers=headers)
                response.raise_for_status()

            logger.info(
                f"[WEBHOOK] Callback berhasil (attempt {attempt}/{max_retries}): "
                f"link_id={link_id}, scrape_status={payload['scrape_status']}, HTTP {response.status_code}"
            )
            return True

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[WEBHOOK] HTTP error {e.response.status_code} attempt {attempt}/{max_retries} "
                f"link_id={link_id}"
            )
        except httpx.RequestError as e:
            logger.warning(
                f"[WEBHOOK] Request error attempt {attempt}/{max_retries} link_id={link_id}: {e}"
            )
        except Exception as e:
            logger.warning(
                f"[WEBHOOK] Error attempt {attempt}/{max_retries} link_id={link_id}: {e}"
            )

        if attempt < max_retries:
            delay = delays[attempt - 1] if attempt - 1 < len(delays) else 8
            await asyncio.sleep(delay)

    logger.error(f"[WEBHOOK] Semua {max_retries} retry gagal untuk link_id={link_id}")
    return False
