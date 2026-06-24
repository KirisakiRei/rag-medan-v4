"""
RAG Medan v3 - LLM OCR Client
Klien HTTP untuk memanggil LLM via Router API (OpenAI-compatible) untuk OCR.
Mendukung retry dengan Exponential Backoff untuk HTTP 429 dan 5xx.
"""
import base64
import logging
import time
from typing import Optional

import requests

from config import config

logger = logging.getLogger("llm_ocr")

# Timeout per request ke Router API (detik)
_REQUEST_TIMEOUT = 120


def call_llm_ocr(
    image_bytes: bytes,
    prompt: Optional[str] = None,
) -> str:
    """
    Kirim gambar halaman (PNG bytes) ke LLM via Router API untuk di-OCR.

    Args:
        image_bytes: Bytes gambar halaman yang akan di-OCR (format PNG).
        prompt: Prompt instruksi OCR. Jika None, gunakan PROMPT_LLM_OCR dari
                prompts.py (dengan mekanisme DB override jika tersedia).

    Returns:
        Teks hasil OCR dalam format Markdown. Kembalikan string kosong
        jika seluruh percobaan gagal.
    """
    if prompt is None:
        prompt = _get_ocr_prompt()

    # Encode gambar ke base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": config.OCR_LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": config.OCR_LLM_MAX_TOKENS,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {config.ROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    max_retries = max(1, config.OCR_RETRIES)

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"[LLM-OCR] Mengirim request ke Router API "
                f"(attempt {attempt + 1}/{max_retries}, model={config.OCR_LLM_MODEL})"
            )
            response = requests.post(
                config.ROUTER_API_URL,
                json=payload,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )

            # --- Penanganan Rate Limit (429) ---
            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning(
                    f"[LLM-OCR] Rate limit (HTTP 429). "
                    f"Menunggu {wait}s sebelum retry (attempt {attempt + 1}/{max_retries})."
                )
                time.sleep(wait)
                continue

            # --- Penanganan Server Error (5xx) ---
            if response.status_code >= 500:
                wait = 5 * (attempt + 1)
                logger.warning(
                    f"[LLM-OCR] Server error (HTTP {response.status_code}). "
                    f"Menunggu {wait}s sebelum retry (attempt {attempt + 1}/{max_retries})."
                )
                time.sleep(wait)
                continue

            # --- Penanganan Client Error (4xx selain 429) ---
            if response.status_code >= 400:
                logger.error(
                    f"[LLM-OCR] Client error (HTTP {response.status_code}). "
                    f"Tidak ada retry. Response: {response.text[:300]}"
                )
                return ""

            # --- Success: Parse response ---
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )

            logger.info(
                f"[LLM-OCR] Berhasil dari Router API. Panjang hasil: {len(content)} karakter."
            )

            # Jeda antar request untuk menjaga stabilitas kuota
            if config.OCR_DELAY > 0:
                time.sleep(config.OCR_DELAY)

            return content.strip()

        except requests.exceptions.Timeout:
            logger.warning(
                f"[LLM-OCR] Timeout ({_REQUEST_TIMEOUT}s) pada attempt {attempt + 1}/{max_retries}."
            )
            time.sleep(5 * (attempt + 1))

        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                f"[LLM-OCR] Connection error pada attempt {attempt + 1}/{max_retries}: {exc}"
            )
            time.sleep(5 * (attempt + 1))

        except Exception as exc:
            logger.error(
                f"[LLM-OCR] Error tidak terduga pada attempt {attempt + 1}/{max_retries}: {exc}",
                exc_info=True,
            )
            time.sleep(5 * (attempt + 1))

    logger.error(
        f"[LLM-OCR] Semua {max_retries} percobaan gagal. "
        "Mengembalikan string kosong untuk halaman ini."
    )
    return ""


def _get_ocr_prompt() -> str:
    """
    Ambil prompt OCR dari cache DB jika tersedia, fallback ke konstanta
    PROMPT_LLM_OCR di shared/prompts.py.
    """
    try:
        from shared.filtering import get_cached_variable
        from shared.prompts import PROMPT_LLM_OCR

        cached = get_cached_variable("prompt_llm_ocr")
        if cached:
            logger.debug("[LLM-OCR] Menggunakan prompt OCR dari tabel variables (DB).")
            return cached
        else:
            logger.debug("[LLM-OCR] Menggunakan prompt OCR default dari prompts.py.")
            return PROMPT_LLM_OCR
    except Exception as exc:
        logger.warning(f"[LLM-OCR] Gagal query DB untuk prompt ({exc}), fallback ke prompts.py.")
        from shared.prompts import PROMPT_LLM_OCR
        return PROMPT_LLM_OCR

