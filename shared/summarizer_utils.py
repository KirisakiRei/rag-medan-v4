"""
RAG Medan v3 - Shared Summarizer Utils
Utilities untuk summarization menggunakan LLM
"""
import logging
import requests
from config import config

logger = logging.getLogger("summarizer_utils")


def _call_gemini_summarizer(system_prompt: str, user_message: str, temperature: float = 0.4, max_tokens: int = 400):
    """
    Helper function untuk memanggil Gemini API untuk summarization.
    """
    try:
        url = f"{config.LLM_BASE_URL}/{config.LLM_MODEL}:generateContent?key={config.LLM_API_KEY}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt.strip()},
                        {"text": user_message.strip()}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.7,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT)
        
        if response.status_code != 200:
            logger.error(f"[GEMINI-SUMMARIZER] HTTP {response.status_code}: {response.text[:200]}")
            return None
            
        response_data = response.json()
        candidates = response_data.get("candidates", [])
        if not candidates:
            return None
            
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return None
            
        return parts[0].get("text", "").strip()
        
    except Exception as e:
        logger.error(f"[GEMINI-SUMMARIZER] Error: {e}")
        return None


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Ringkas teks dengan Gemini LLM.
    """
    try:
        # batasi panjang input agar tidak boros token
        text_snippet = text.strip()[:4000]
        
        system_prompt = "Anda adalah asisten yang ahli dalam meringkas dokumen panjang menjadi versi singkat yang mudah dipahami."
        user_prompt = (
            f"Ringkas teks berikut menjadi maksimal {max_sentences} kalimat yang padat, jelas, "
            f"dan tetap mempertahankan konteks penting.\n\nTeks:\n{text_snippet}"
        )

        logger.info(f"[SUMMARIZER] Mengirim teks ({len(text_snippet)} chars) ke Gemini model '{config.LLM_MODEL}'")

        generated_summary = _call_gemini_summarizer(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.4,
            max_tokens=400
        )

        if not generated_summary:
            logger.warning("[SUMMARIZER] Gemini API error atau empty response, pakai fallback.")
            generated_summary = text_snippet[:350] + "..."

        return generated_summary

    except Exception as e:
        logger.error(f"[SUMMARIZER] Gagal meringkas: {e}")
        return text.strip()[:350] + "..."
