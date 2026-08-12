"""
RAG Medan v4 - Shared LightRAG Real-time Sync.

Menyediakan fungsi fire-and-forget untuk sinkronisasi data
ke LightRAG Adapter secara real-time setiap ada add/update/delete
di service existing (rag_text, rag_document, rag_web).

Prinsip:
- Non-blocking: menggunakan asyncio.create_task() atau threading
- Non-breaking: error di LightRAG sync tidak menggagalkan operasi utama
- Conditional: hanya aktif jika RAG_SEARCH_ENGINE bukan "legacy"

Usage di setiap service:
    from shared.lightrag_sync import fire_lightrag_sync_text

    # Setelah Qdrant berhasil:
    fire_lightrag_sync_text(
        source_id=str(item["question_rag_id"]),
        title=item.get("question", ""),
        content="",
        content_hash="",
        is_active=True,
        category=item.get("category_id"),
        question=item.get("question"),
        answer=None,
    )
"""
import asyncio
import logging
import threading
from typing import Optional

import httpx

from config import config

logger = logging.getLogger("lightrag_sync")

# ── Shared HTTP client (lazy init) ──
_http_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Get or create shared async HTTP client for LightRAG Adapter."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=config.LIGHTRAG_ADAPTER_URL,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": config.INTERNAL_API_KEY,
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )
    return _http_client


def _should_sync() -> bool:
    """Check apakah LightRAG sync perlu dijalankan."""
    engine = config.RAG_SEARCH_ENGINE.lower()
    return engine in ("lightrag", "shadow")


# ============== FIRE-AND-FORGET ENTRY POINTS ==============

def fire_lightrag_sync_text(
    source_id: str,
    title: str,
    content: str = "",
    content_hash: str = "",
    is_active: bool = True,
    category: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
) -> None:
    """
    Fire-and-forget: sync text/FAQ ke LightRAG Adapter.

    Panggil setelah Qdrant upsert berhasil.
    Tidak blocking, tidak raise exception.
    """
    if not _should_sync():
        return

    asyncio.create_task(_sync_text_task(
        source_id=source_id,
        title=title,
        content=content,
        content_hash=content_hash,
        is_active=is_active,
        category=category,
        question=question,
        answer=answer,
    ))


def fire_lightrag_sync_document(
    source_id: str,
    title: str,
    normalized_content: str,
    file_name: Optional[str] = None,
    content_hash: str = "",
    is_active: bool = True,
    organization_id: Optional[str] = None,
) -> None:
    """
    Fire-and-forget: sync dokumen ke LightRAG Adapter.

    Panggil setelah Qdrant upsert berhasil.
    Tidak blocking, tidak raise exception.
    """
    if not _should_sync():
        return

    asyncio.create_task(_sync_document_task(
        source_id=source_id,
        title=title,
        normalized_content=normalized_content,
        file_name=file_name,
        content_hash=content_hash,
        is_active=is_active,
        organization_id=organization_id,
    ))


def fire_lightrag_sync_web(
    source_id: str,
    url: str,
    title: str,
    clean_content: str,
    content_hash: str = "",
    is_active: bool = True,
) -> None:
    """
    Fire-and-forget: sync web page ke LightRAG Adapter.

    Panggil setelah Qdrant upsert berhasil.
    Tidak blocking, tidak raise exception.
    """
    if not _should_sync():
        return

    asyncio.create_task(_sync_web_task(
        source_id=source_id,
        url=url,
        title=title,
        clean_content=clean_content,
        content_hash=content_hash,
        is_active=is_active,
    ))


def fire_lightrag_delete(
    source_type: str,
    source_id: str,
) -> None:
    """
    Fire-and-forget: hapus source dari LightRAG.

    Args:
        source_type: "text" | "document" | "web"
        source_id: Application primary key.

    Panggil setelah Qdrant delete berhasil.
    Tidak blocking, tidak raise exception.
    """
    if not _should_sync():
        return

    asyncio.create_task(_delete_task(
        source_type=source_type,
        source_id=source_id,
    ))


# ============== INTERNAL TASK IMPLEMENTATIONS ==============

async def _sync_text_task(
    source_id: str,
    title: str,
    content: str,
    content_hash: str,
    is_active: bool,
    category: Optional[str],
    question: Optional[str],
    answer: Optional[str],
) -> None:
    """Background task: POST /internal/sync/text ke adapter."""
    try:
        client = _get_client()
        payload = {
            "source_id": source_id,
            "title": title,
            "content": content,
            "content_hash": content_hash,
            "is_active": is_active,
            "category": category,
            "question": question,
            "answer": answer,
        }
        response = await client.post("/internal/sync/text", json=payload)
        if response.status_code == 200:
            logger.debug(f"[LR-SYNC] Text synced: {source_id}")
        else:
            logger.warning(
                f"[LR-SYNC] Text sync failed for {source_id}: "
                f"HTTP {response.status_code}"
            )
    except Exception as exc:
        logger.warning(
            f"[LR-SYNC] Text sync error for {source_id}: "
            f"{type(exc).__name__}: {exc}"
        )


async def _sync_document_task(
    source_id: str,
    title: str,
    normalized_content: str,
    file_name: Optional[str],
    content_hash: str,
    is_active: bool,
    organization_id: Optional[str],
) -> None:
    """Background task: POST /internal/sync/document ke adapter."""
    try:
        client = _get_client()
        payload = {
            "source_id": source_id,
            "title": title,
            "normalized_content": normalized_content,
            "file_name": file_name,
            "content_hash": content_hash,
            "is_active": is_active,
            "organization_id": organization_id,
        }
        response = await client.post("/internal/sync/document", json=payload)
        if response.status_code == 200:
            logger.debug(f"[LR-SYNC] Document synced: {source_id}")
        else:
            logger.warning(
                f"[LR-SYNC] Document sync failed for {source_id}: "
                f"HTTP {response.status_code}"
            )
    except Exception as exc:
        logger.warning(
            f"[LR-SYNC] Document sync error for {source_id}: "
            f"{type(exc).__name__}: {exc}"
        )


async def _sync_web_task(
    source_id: str,
    url: str,
    title: str,
    clean_content: str,
    content_hash: str,
    is_active: bool,
) -> None:
    """Background task: POST /internal/sync/web ke adapter."""
    try:
        client = _get_client()
        payload = {
            "source_id": source_id,
            "url": url,
            "title": title,
            "clean_content": clean_content,
            "content_hash": content_hash,
            "is_active": is_active,
        }
        response = await client.post("/internal/sync/web", json=payload)
        if response.status_code == 200:
            logger.debug(f"[LR-SYNC] Web synced: {source_id}")
        else:
            logger.warning(
                f"[LR-SYNC] Web sync failed for {source_id}: "
                f"HTTP {response.status_code}"
            )
    except Exception as exc:
        logger.warning(
            f"[LR-SYNC] Web sync error for {source_id}: "
            f"{type(exc).__name__}: {exc}"
        )


async def _delete_task(
    source_type: str,
    source_id: str,
) -> None:
    """Background task: DELETE /internal/source/{type}/{id} ke adapter."""
    try:
        client = _get_client()
        response = await client.delete(
            f"/internal/source/{source_type}/{source_id}"
        )
        if response.status_code == 200:
            logger.debug(f"[LR-SYNC] Deleted {source_type}:{source_id}")
        else:
            logger.warning(
                f"[LR-SYNC] Delete failed for {source_type}:{source_id}: "
                f"HTTP {response.status_code}"
            )
    except Exception as exc:
        logger.warning(
            f"[LR-SYNC] Delete error for {source_type}:{source_id}: "
            f"{type(exc).__name__}: {exc}"
        )


# ============== SYNCHRONOUS FIRE-AND-FORGET (untuk worker thread) ==============

def fire_lightrag_sync_document_sync(
    source_id: str,
    title: str,
    normalized_content: str,
    file_name: Optional[str] = None,
    content_hash: str = "",
    is_active: bool = True,
    organization_id: Optional[str] = None,
) -> None:
    """
    Fire-and-forget (threading): sync dokumen ke LightRAG Adapter.

    Versi synchronous untuk dipakai di document worker (sync context).
    Menjalankan HTTP call di background thread.
    """
    if not _should_sync():
        return

    thread = threading.Thread(
        target=_sync_document_thread,
        args=(source_id, title, normalized_content, file_name,
              content_hash, is_active, organization_id),
        daemon=True,
    )
    thread.start()


def fire_lightrag_delete_sync(
    source_type: str,
    source_id: str,
) -> None:
    """
    Fire-and-forget (threading): hapus source dari LightRAG.

    Versi synchronous untuk dipakai di document worker (sync context).
    """
    if not _should_sync():
        return

    thread = threading.Thread(
        target=_delete_thread,
        args=(source_type, source_id),
        daemon=True,
    )
    thread.start()


def _sync_document_thread(
    source_id: str,
    title: str,
    normalized_content: str,
    file_name: Optional[str],
    content_hash: str,
    is_active: bool,
    organization_id: Optional[str],
) -> None:
    """Background thread: POST /internal/sync/document ke adapter (sync HTTP)."""
    try:
        url = f"{config.LIGHTRAG_ADAPTER_URL}/internal/sync/document"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": config.INTERNAL_API_KEY,
        }
        payload = {
            "source_id": source_id,
            "title": title,
            "normalized_content": normalized_content,
            "file_name": file_name,
            "content_hash": content_hash,
            "is_active": is_active,
            "organization_id": organization_id,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            logger.debug(f"[LR-SYNC] Document synced: {source_id}")
        else:
            logger.warning(
                f"[LR-SYNC] Document sync failed for {source_id}: "
                f"HTTP {response.status_code}"
            )
    except Exception as exc:
        logger.warning(
            f"[LR-SYNC] Document sync error for {source_id}: "
            f"{type(exc).__name__}: {exc}"
        )


def _delete_thread(
    source_type: str,
    source_id: str,
) -> None:
    """Background thread: DELETE /internal/source/{type}/{id} ke adapter (sync HTTP)."""
    try:
        url = f"{config.LIGHTRAG_ADAPTER_URL}/internal/source/{source_type}/{source_id}"
        headers = {"X-API-Key": config.INTERNAL_API_KEY}
        with httpx.Client(timeout=30.0) as client:
            response = client.delete(url, headers=headers)
        if response.status_code == 200:
            logger.debug(f"[LR-SYNC] Deleted {source_type}:{source_id}")
        else:
            logger.warning(
                f"[LR-SYNC] Delete failed for {source_type}:{source_id}: "
                f"HTTP {response.status_code}"
            )
    except Exception as exc:
        logger.warning(
            f"[LR-SYNC] Delete error for {source_type}:{source_id}: "
            f"{type(exc).__name__}: {exc}"
        )
