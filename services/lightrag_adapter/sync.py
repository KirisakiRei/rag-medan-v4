"""
RAG Medan v4 - LightRAG Adapter — Sync Logic.

Bertanggung jawab:
- Menerima normalized content dari source processors (text/document/web)
- Membangun deterministic document ID
- Mengirim konten ke LightRAG untuk indexing
- Menangani deletion (is_active=false)
- Return sync status

Setiap sync operation bersifat idempotent — content_hash check
dilakukan di level pemanggil (source processor) sebelum memanggil adapter.
"""
import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, List

from services.lightrag_adapter.client import lightrag_client
from services.lightrag_adapter.source_mapper import (
    make_document_id,
    normalize_text_content,
    normalize_document_content,
    normalize_web_content,
    normalize_usulan_content,
)
from services.lightrag_adapter.stats import stats
from services.lightrag_adapter.config import adapter_config

logger = logging.getLogger("lightrag_adapter.sync")


# ============== SYNC — TEXT ==============

async def sync_text(
    source_id: str,
    knowledge_base_id: str,
    title: str,
    content: str,
    content_hash: str,
    is_active: bool = True,
    category: str = None,
    answer_id: List[str] = None,
    question: str = None,
    answer: str = None,
) -> Dict[str, Any]:
    """
    Sync text/FAQ knowledge ke LightRAG.

    Args:
        source_id: Application primary key (question_rag_id).
        knowledge_base_id: Workspace (e.g. "medan-main").
        title: FAQ title / question short form.
        content: Pre-formatted content (jika ada).
        content_hash: Hash konten — diterima untuk traceability response,
                      idempotency check (skip jika hash sama) dilakukan
                      oleh source processor sebelum memanggil endpoint ini.
        is_active: Jika False, hapus dari LightRAG.
        category: FAQ category (optional metadata).
        question: Original question text.
        answer: Original answer text.

    Returns:
        SyncResponse dict.
    """
    doc_id = make_document_id(knowledge_base_id, "text", source_id)

    if not is_active:
        return await _delete_source(doc_id, "text", source_id)

    normalized = normalize_text_content(
        title=title,
        question=question,
        answer=answer,
        category=category,
        raw_content=content if content else None,
    )

    metadata: Dict[str, Any] = {
        "category_id": category,
        "answer_id": answer_id or [],
    }
    result = await _index_document(doc_id, normalized, "text", source_id, metadata=metadata)
    result["content_hash"] = content_hash
    stats.record_sync("text", result.get("status") == "success")
    return result


# ============== SYNC — DOCUMENT ==============

async def sync_document(
    source_id: str,
    knowledge_base_id: str,
    title: str,
    normalized_content: str,
    file_name: str = None,
    content_hash: str = "",
    is_active: bool = True,
    organization_id: str = None,
) -> Dict[str, Any]:
    """
    Sync document knowledge ke LightRAG.

    Konten yang dikirim adalah hasil ekstraksi Document Worker
    (sudah melalui OCR, layout extraction, dll).

    Args:
        source_id: Application primary key (doc_id).
        knowledge_base_id: Workspace (e.g. "medan-main").
        title: Document title.
        normalized_content: Full extracted text dari Document Worker.
        file_name: Original file name (untuk citation).
        content_hash: Hash konten — diterima untuk traceability response,
                      idempotency check dilakukan oleh source processor
                      sebelum memanggil endpoint ini.
        is_active: Jika False, hapus dari LightRAG.
        organization_id: OPD/organization identifier.

    Returns:
        SyncResponse dict.
    """
    doc_id = make_document_id(knowledge_base_id, "document", source_id)

    if not is_active:
        return await _delete_source(doc_id, "document", source_id)

    normalized = normalize_document_content(
        title=title,
        normalized_content=normalized_content,
        organization_id=organization_id,
        file_name=file_name,
    )

    result = await _index_document(doc_id, normalized, "document", source_id)
    result["content_hash"] = content_hash
    stats.record_sync("document", result.get("status") == "success")
    return result


# ============== SYNC — WEB ==============

async def sync_web(
    source_id: str,
    knowledge_base_id: str,
    url: str,
    title: str,
    clean_content: str,
    content_hash: str = "",
    is_active: bool = True,
) -> Dict[str, Any]:
    """
    Sync web page knowledge ke LightRAG.

    Konten yang dikirim adalah hasil cleaning dari Web Scraper
    (sudah melalui main-content extraction, nav/footer removal, dll).

    Args:
        source_id: Application primary key (web_bank_id).
        knowledge_base_id: Workspace (e.g. "medan-main").
        url: Source URL (untuk citation).
        title: Page title.
        clean_content: Cleaned web content.
        content_hash: Hash konten — diterima untuk traceability response,
                      idempotency check dilakukan oleh source processor
                      sebelum memanggil endpoint ini.
        is_active: Jika False, hapus dari LightRAG.

    Returns:
        SyncResponse dict.
    """
    doc_id = make_document_id(knowledge_base_id, "web", source_id)

    if not is_active:
        return await _delete_source(doc_id, "web", source_id)

    normalized = normalize_web_content(
        title=title,
        url=url,
        clean_content=clean_content,
    )

    result = await _index_document(doc_id, normalized, "web", source_id)
    result["content_hash"] = content_hash
    stats.record_sync("web", result.get("status") == "success")
    return result


# ============== SYNC — USULAN ==============

async def sync_usulan(
    source_id: str,
    knowledge_base_id: str = "usulan-main",
    title: str = "",
    content: str = "",
    content_hash: str = "",
    is_active: bool = True,
    organization_id: str = None,
    request_id: str = None,
    request_name: str = None,
    question: str = None,
) -> Dict[str, Any]:
    """
    Sync usulan knowledge ke LightRAG.

    Payload mirip text: question-only (pertanyaan warga), tanpa jawaban.
    Metadata usulan (organization_id, request_id, request_name) di-embed
    ke header konten sebagai provenance.

    Args:
        source_id: Application primary key (request_rag_id).
        knowledge_base_id: Workspace logis (default "usulan-main").
        title: Request title / short form.
        content: Pre-formatted content (jika ada).
        content_hash: Hash konten — untuk traceability.
        is_active: Jika False, hapus dari LightRAG.
        organization_id: OPD identifier (metadata).
        request_id: Application request id (metadata).
        request_name: Full request name (metadata).
        question: Request question text (request_rag_name).

    Returns:
        SyncResponse dict.
    """
    doc_id = make_document_id(knowledge_base_id, "usulan", source_id)

    if not is_active:
        return await _delete_source(doc_id, "usulan", source_id)

    normalized = normalize_usulan_content(
        title=title,
        question=question or title,
        organization_id=organization_id,
        request_id=request_id,
        request_name=request_name,
    )

    metadata: Dict[str, Any] = {
        "category_id": organization_id,
        "request_id": request_id,
        "request_name": request_name,
    }
    result = await _index_document(doc_id, normalized, "usulan", source_id, metadata=metadata)
    result["content_hash"] = content_hash
    stats.record_sync("usulan", result.get("status") == "success")
    return result


# ============== DELETE ==============

async def delete_source(    source_type: str,
    source_id: str,
    knowledge_base_id: str = "medan-main",
) -> Dict[str, Any]:
    """
    Delete source dari LightRAG index.

    Args:
        source_type: "text" | "document" | "web"
        source_id: Application primary key.
        knowledge_base_id: Workspace.

    Returns:
        SyncResponse dict.
    """
    doc_id = make_document_id(knowledge_base_id, source_type, source_id)
    result = await _delete_source(doc_id, source_type, source_id)
    stats.record_delete(source_type)
    return result


# ============== INTERNAL HELPERS ==============

async def _index_document(
    doc_id: str,
    content: str,
    source_type: str,
    source_id: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Kirim normalized content ke LightRAG untuk indexing.

    Args:
        doc_id: Deterministic document ID.
        content: Normalized text content.
        source_type: "text" | "document" | "web"
        source_id: Application primary key.

    Returns:
        SyncResponse dict.
    """
    logger.info(
        f"[LR-SYNC] Indexing {source_type}:{source_id} "
        f"(doc_id={doc_id}, content_len={len(content)})"
    )

    try:
        source_descriptor = f"{source_type}:{source_id}"
        # LightRAG juga melakukan deduplikasi berdasarkan hash konten. Dua FAQ
        # dapat memiliki pertanyaan identik tetapi answer_id/source_id berbeda;
        # tanpa provenance marker, source kedua ditolak sebagai identical content.
        # Marker ini metadata internal, bukan jawaban FAQ, dan menjaga hubungan
        # satu-ke-satu antara retrieved source dengan application source ID.
        header_parts = [f"Source-ID: {source_descriptor}"]
        if metadata:
            category = metadata.get("category_id")
            if category:
                header_parts.append(f"Category-ID: {category}")
            answer_ids = metadata.get("answer_id")
            if answer_ids:
                header_parts.append(
                    "Answer-ID: " + ",".join(str(a) for a in answer_ids if str(a).strip())
                )
        indexed_content = "\n".join(header_parts) + "\n" + content
        existing = await _find_document_by_source(source_descriptor)
        if existing:
            await _delete_actual_document(
                str(existing.get("id") or _actual_document_id(source_descriptor)),
                source_descriptor,
            )
        result = await lightrag_client.insert_text(
            text=indexed_content,
            file_source=source_descriptor,
            metadata=metadata,
        )
        track_id = str(result.get("track_id") or "").strip()
        if result.get("status") != "success" or not track_id:
            raise RuntimeError(
                f"LightRAG tidak mengonfirmasi enqueue: {result}"
            )

        tracked = await _wait_until_indexed(track_id, source_descriptor)
        actual_doc_id = tracked.get("id")
        logger.info(
            f"[LR-SYNC] Indexed {source_descriptor} successfully "
            f"(doc_id={actual_doc_id}, track_id={track_id})"
        )
        return {
            "status": "success",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": actual_doc_id,
            "logical_document_id": doc_id,
            "file_source": source_descriptor,
            "track_id": track_id,
            "message": "Indexed successfully",
        }

    except Exception as e:
        logger.error(f"[LR-SYNC] Failed to index {source_type}:{source_id}: {e}")
        return {
            "status": "error",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": None,
            "logical_document_id": doc_id,
            "message": str(e),
        }


def _actual_document_id(file_source: str) -> str:
    """Mirror LightRAG's stable doc ID algorithm for known file_source."""
    return "doc-" + hashlib.md5(file_source.encode("utf-8")).hexdigest()


async def _find_document_by_source(file_source: str) -> Dict[str, Any] | None:
    """Find an exact source in LightRAG's paginated document registry."""
    page = 1
    while True:
        result = await lightrag_client.get_documents_paginated(page=page, page_size=100)
        for document in result.get("documents") or []:
            if str(document.get("file_path") or "").strip() == file_source:
                return document
        pagination = result.get("pagination") or {}
        if not pagination.get("has_next"):
            return None
        page += 1


async def _delete_actual_document(actual_doc_id: str, file_source: str) -> None:
    """Start deletion and wait until the exact source disappears."""
    deadline = time.monotonic() + adapter_config.INDEX_TIMEOUT_SEC
    while time.monotonic() < deadline:
        result = await lightrag_client.delete_document(actual_doc_id)
        status = str(result.get("status") or "").lower()
        if status in {"deletion_started", "success", "deleted"}:
            break
        if status == "busy":
            await asyncio.sleep(adapter_config.INDEX_POLL_INTERVAL_SEC)
            continue
        raise RuntimeError(f"LightRAG menolak delete {file_source}: {result}")
    else:
        raise TimeoutError(f"Timeout memulai delete LightRAG untuk {file_source}")

    while time.monotonic() < deadline:
        if await _find_document_by_source(file_source) is None:
            return
        await asyncio.sleep(adapter_config.INDEX_POLL_INTERVAL_SEC)
    raise TimeoutError(f"Timeout menunggu delete LightRAG untuk {file_source}")


async def _wait_until_indexed(track_id: str, expected_file_source: str) -> Dict[str, Any]:
    """Poll LightRAG until the enqueued source reaches a terminal status."""
    deadline = time.monotonic() + adapter_config.INDEX_TIMEOUT_SEC
    terminal_success = {"processed", "success", "completed"}
    terminal_failure = {"failed", "error", "cancelled", "canceled"}

    while time.monotonic() < deadline:
        result = await lightrag_client.get_track_status(track_id)
        documents = result.get("documents") or []
        if documents:
            failures = []
            all_done = True
            for document in documents:
                status = str(document.get("status") or "").lower()
                if status in terminal_failure:
                    failures.append(document.get("error_msg") or status)
                elif status not in terminal_success:
                    all_done = False
            if failures:
                raise RuntimeError(
                    f"LightRAG indexing gagal untuk {expected_file_source}: "
                    + "; ".join(str(item) for item in failures)
                )
            if all_done:
                matching = next(
                    (
                        doc for doc in documents
                        if str(doc.get("file_path") or "").strip() == expected_file_source
                    ),
                    documents[0],
                )
                return matching
        await asyncio.sleep(adapter_config.INDEX_POLL_INTERVAL_SEC)

    raise TimeoutError(
        f"LightRAG indexing timeout untuk {expected_file_source} "
        f"(track_id={track_id})"
    )


async def _delete_source(
    doc_id: str,
    source_type: str,
    source_id: str,
) -> Dict[str, Any]:
    """
    Hapus document dari LightRAG index.

    Args:
        doc_id: Deterministic document ID.
        source_type: "text" | "document" | "web"
        source_id: Application primary key.

    Returns:
        SyncResponse dict.
    """
    source_descriptor = f"{source_type}:{source_id}"
    logger.info(f"[LR-SYNC] Deleting {source_descriptor} (logical_id={doc_id})")

    try:
        existing = await _find_document_by_source(source_descriptor)
        actual_doc_id = str(
            (existing or {}).get("id") or _actual_document_id(source_descriptor)
        )
        if existing:
            await _delete_actual_document(actual_doc_id, source_descriptor)
        logger.info(f"[LR-SYNC] Deleted {source_descriptor}")
        return {
            "status": "success",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": actual_doc_id,
            "logical_document_id": doc_id,
            "file_source": source_descriptor,
            "message": "Deleted from LightRAG",
        }

    except Exception as e:
        logger.error(f"[LR-SYNC] Failed to delete {source_type}:{source_id}: {e}")
        return {
            "status": "error",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": _actual_document_id(source_descriptor),
            "message": str(e),
        }
