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
import logging
from typing import Dict, Any

from services.lightrag_adapter.client import lightrag_client
from services.lightrag_adapter.source_mapper import (
    make_document_id,
    normalize_text_content,
    normalize_document_content,
    normalize_web_content,
)
from services.lightrag_adapter.stats import stats

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

    result = await _index_document(doc_id, normalized, "text", source_id)
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


# ============== DELETE ==============

async def delete_source(
    source_type: str,
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
        result = await lightrag_client.insert_text(
            text=content,
            description=f"{source_type}:{source_id}",
        )
        logger.info(f"[LR-SYNC] Indexed {source_type}:{source_id} successfully")
        return {
            "status": "success",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": doc_id,
            "message": "Indexed successfully",
        }

    except Exception as e:
        logger.error(f"[LR-SYNC] Failed to index {source_type}:{source_id}: {e}")
        return {
            "status": "error",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": doc_id,
            "message": str(e),
        }


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
    logger.info(f"[LR-SYNC] Deleting {source_type}:{source_id} (doc_id={doc_id})")

    try:
        await lightrag_client.delete_document(doc_id)
        logger.info(f"[LR-SYNC] Deleted {source_type}:{source_id}")
        return {
            "status": "success",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": doc_id,
            "message": "Deleted from LightRAG",
        }

    except Exception as e:
        logger.error(f"[LR-SYNC] Failed to delete {source_type}:{source_id}: {e}")
        return {
            "status": "error",
            "source_id": source_id,
            "source_type": source_type,
            "lightrag_document_id": doc_id,
            "message": str(e),
        }
