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
}


def _build_message(status: str, result: Dict[str, Any]) -> str:
    """Bangun scrape_message berdasarkan status dan result."""
    if status == "scraping":
        return result.get("scrape_message") or "Sedang mengambil konten dari halaman web..."

    if status == "completed":
        if result.get("dedup_skipped"):
            return "Halaman berhasil di-scrape. Konten tidak berubah, indeks tidak diperbarui"

        page_count = result.get("page_count", 1)
        paragraph_count = result.get("paragraph_count", 0)
        chunks = result.get("chunks_count", 0)
        return (
            "Halaman berhasil di-scrape. "
            f"Total konten: {page_count} halaman, {paragraph_count} paragraf, {chunks} chunk"
        )

    if status == "failed":
        error = (result.get("error") or "URL tidak dapat dijangkau").strip()
        if error.lower().startswith("gagal "):
            return error
        return f"Gagal mengakses halaman web: {error}"

    return ""


def build_callback_payload(status: str, result: Dict[str, Any]) -> Dict[str, str]:
    """Bangun payload callback tanpa mengirim request."""
    return {
        "scrape_status": _STATUS_MAP.get(status, "failed"),
        "scrape_message": _build_message(status, result),
    }


async def send_callback(
    web_bank_id: str,
    url: str,
    status: str,
    result: Dict[str, Any],
    callback_url: Optional[str] = None,
) -> bool:
    """Kirim webhook PUT ke web-banks/scrape-status/{web_bank_id}. Return True jika berhasil."""
    from config import config as app_config

    web_callback_url = callback_url or app_config.WEB_CALLBACK_URL
    if not web_callback_url:
        logger.debug("[WEBHOOK] WEB_CALLBACK_URL tidak dikonfigurasi, skip")
        return False

    target_url = f"{web_callback_url.rstrip('/')}/{web_bank_id}"
    payload = build_callback_payload(status, result)

    headers = {"Content-Type": "application/json"}
    if app_config.WEB_MANAJEMEN_API_KEY:
        headers["X-API-KEY"] = app_config.WEB_MANAJEMEN_API_KEY

    max_retries = app_config.WEBHOOK_RETRY_ATTEMPTS
    delays = [2 ** i for i in range(max_retries)]

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                verify=app_config.WEBHOOK_SSL_VERIFY,
            ) as client:
                response = await client.put(target_url, json=payload, headers=headers)
                response.raise_for_status()

            logger.info(
                f"[WEBHOOK] Callback berhasil (attempt {attempt}/{max_retries}): "
                f"web_bank_id={web_bank_id}, scrape_status={payload['scrape_status']}, "
                f"HTTP {response.status_code}"
            )
            return True

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[WEBHOOK] HTTP error {e.response.status_code} attempt {attempt}/{max_retries} "
                f"web_bank_id={web_bank_id}"
            )
        except httpx.RequestError as e:
            logger.warning(
                f"[WEBHOOK] Request error attempt {attempt}/{max_retries} "
                f"web_bank_id={web_bank_id}: {e}"
            )
        except Exception as e:
            logger.warning(
                f"[WEBHOOK] Error attempt {attempt}/{max_retries} "
                f"web_bank_id={web_bank_id}: {e}"
            )

        if attempt < max_retries:
            delay = delays[attempt - 1] if attempt - 1 < len(delays) else 8
            await asyncio.sleep(delay)

    logger.error(f"[WEBHOOK] Semua {max_retries} retry gagal untuk web_bank_id={web_bank_id}")
    return False
