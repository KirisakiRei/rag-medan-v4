"""
RAG Medan v3 - Internal API Key Authentication Middleware

Melindungi orchestrator dan semua service internal menggunakan header
`X-API-Key`. Struktur payload API tidak berubah; autentikasi murni
berjalan di level header HTTP.

Aturan:
- Key dibaca dari config.INTERNAL_API_KEY per-request (bisa dirotasi
  tanpa restart, dan mudah di-test).
- Fail-closed: jika key kosong, semua request non-allowlist ditolak 401.
- Allowlist: `/`, `/health`, dan path dokumentasi FastAPI tetap terbuka.
- Metode OPTIONS dibiarkan lewat agar CORS preflight tetap berfungsi.
- Perbandingan key memakai secrets.compare_digest (constant-time).
- Nilai key tidak pernah ditulis ke log.
"""
import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import config

logger = logging.getLogger("security")

_ALLOWED_EXACT_PATHS = {"/", "/health"}
_ALLOWED_PREFIX_PATHS = ("/docs", "/redoc", "/openapi.json")


def _is_allowed_path(path: str) -> bool:
    """Kembalikan True jika path boleh diakses tanpa X-API-Key."""
    if path in _ALLOWED_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _ALLOWED_PREFIX_PATHS)


class InternalAuthMiddleware(BaseHTTPMiddleware):
    """Validasi header X-API-Key pada seluruh request masuk."""

    def __init__(self, app):
        super().__init__(app)
        if not config.INTERNAL_API_KEY:
            logger.warning(
                "INTERNAL_API_KEY kosong! Fail-closed aktif: "
                "semua request non-allowlist akan ditolak (HTTP 401). "
                "Set INTERNAL_API_KEY di .env sebelum menjalankan service."
            )

    async def dispatch(self, request: Request, call_next):
        # CORS preflight tidak membawa header custom → harus dibiarkan.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Endpoint publik / monitoring / dokumentasi tetap terbuka.
        if _is_allowed_path(request.url.path):
            return await call_next(request)

        expected = config.INTERNAL_API_KEY
        provided = request.headers.get("X-API-Key", "")
        if (
            not expected
            or not provided
            or not secrets.compare_digest(provided, expected)
        ):
            client_host = request.client.host if request.client else "unknown"
            logger.warning(
                f"[AUTH] Unauthorized request ditolak: "
                f"path={request.url.path} method={request.method} client={client_host}"
            )
            return JSONResponse(
                {"status": "error", "message": "Unauthorized"},
                status_code=401,
            )

        return await call_next(request)
