"""Sync module for web_scraping_bank."""
import hashlib
import os
import sys
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.lightrag_sync import fire_lightrag_sync_web, fire_lightrag_delete
from services.rag_document.chunker import ChunkItem
from services.rag_web.scraper import scraper, WebScrapeError
from services.rag_web.cleaner import cleaner
from services.rag_web import chunker as web_chunker

logger = logging.getLogger("rag_web.sync")

qdrant: AsyncQdrantClient = None

_STATE_VECTOR = [0.0]
_active_jobs: set[str] = set()
_active_jobs_lock = threading.Lock()


def reserve_job(web_bank_id: str) -> bool:
    """Reserve an in-flight scraping slot for a web bank."""
    with _active_jobs_lock:
        if web_bank_id in _active_jobs:
            return False
        _active_jobs.add(web_bank_id)
        return True


def release_job(web_bank_id: str) -> None:
    """Release an in-flight scraping slot for a web bank."""
    with _active_jobs_lock:
        _active_jobs.discard(web_bank_id)


def set_instances(qdrant_client: AsyncQdrantClient):
    """Set global Qdrant instance."""
    global qdrant
    qdrant = qdrant_client


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def _log_stage(
    web_bank_id: str,
    stage: str,
    message: str,
    *,
    job_id: Optional[str] = None,
    started_at: Optional[float] = None,
    level: str = "info",
    **extra: Any,
) -> None:
    """Log progress stage for scraping monitoring."""
    parts = [f"web_bank_id={web_bank_id}", f"stage={stage}"]
    if job_id:
        parts.append(f"job_id={job_id}")
    if started_at is not None:
        parts.append(f"elapsed={time.monotonic() - started_at:.2f}s")
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    log_line = f"[PROGRESS] {' | '.join(parts)} | {message}"
    getattr(logger, level)(log_line)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_paragraphs(text: str) -> int:
    return len([part for part in text.split("\n\n") if part.strip()])


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _state_point_id(web_bank_id: str) -> str:
    """Map external web_bank_id to a valid deterministic Qdrant point ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"web-state:{web_bank_id}"))


def _classify_processing_error(
    exc: Exception,
    *,
    current_stage: str,
    css_selector: Optional[str] = None,
) -> Dict[str, str]:
    """Classify scrape failures into target-web vs system-processing categories."""
    if isinstance(exc, WebScrapeError):
        prefix = (
            "Gagal mengakses target web"
            if exc.source == "target_web"
            else "Gagal pada sistem scraping RAG"
        )
        return {
            "error_source": exc.source,
            "error_code": exc.code,
            "user_message": f"{prefix}: {exc.user_message}",
            "detail_message": exc.detail,
        }

    error_text = (str(exc) or "").strip()
    lowered = error_text.lower()

    if "konten terlalu pendek atau kosong setelah cleaning" in lowered:
        hint = " Periksa URL target atau CSS selector." if css_selector else ""
        return {
            "error_source": "target_web",
            "error_code": "content_empty_after_cleaning",
            "user_message": (
                "Gagal mengambil konten yang cukup dari halaman target setelah proses cleaning."
                f"{hint}"
            ),
            "detail_message": error_text,
        }

    if "tidak ada chunk yang berhasil dibuat" in lowered:
        hint = " Coba ganti CSS selector atau gunakan halaman yang lebih spesifik." if css_selector else ""
        return {
            "error_source": "target_web",
            "error_code": "chunk_extraction_failed",
            "user_message": (
                "Konten berhasil diambil, tetapi struktur halaman tidak dapat diubah menjadi chunk RAG."
                f"{hint}"
            ),
            "detail_message": error_text,
        }

    return {
        "error_source": "rag_system",
        "error_code": "internal_processing_error",
        "user_message": (
            f"Terjadi kesalahan internal pada tahap {current_stage} saat memproses hasil scraping."
        ),
        "detail_message": error_text or current_stage,
    }


def _chunk_filter(web_bank_id: str, include_deleted: bool = False) -> qdrant_models.Filter:
    conditions = [
        qdrant_models.FieldCondition(
            key="web_bank_id",
            match=qdrant_models.MatchValue(value=web_bank_id)
        ),
        qdrant_models.FieldCondition(
            key="link_id",
            match=qdrant_models.MatchValue(value=web_bank_id)
        ),
    ]
    must_conditions: List[qdrant_models.FieldCondition] = []
    if not include_deleted:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="is_deleted",
                match=qdrant_models.MatchValue(value=False)
            )
        )

    return qdrant_models.Filter(
        must=must_conditions or None,
        min_should=qdrant_models.MinShould(conditions=conditions, min_count=1),
    )


async def get_web_state(web_bank_id: str) -> Optional[Dict[str, Any]]:
    """Get persisted web scraping state for a web bank."""
    points = await qdrant.retrieve(
        collection_name=config.COLLECTION_WEB_STATE,
        ids=[_state_point_id(web_bank_id)],
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        results, _ = await qdrant.scroll(
            collection_name=config.COLLECTION_WEB_STATE,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="web_bank_id",
                        match=qdrant_models.MatchValue(value=web_bank_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        points = results
    if not points:
        return None
    return dict(points[0].payload or {})


async def upsert_web_state(web_bank_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge and persist web scraping state."""
    existing = await get_web_state(web_bank_id) or {}
    payload = {
        **existing,
        **updates,
        "web_bank_id": web_bank_id,
        "updated_at": _utcnow_iso(),
    }
    payload.setdefault("is_active", True)
    payload = {k: v for k, v in payload.items() if v is not None}

    await qdrant.upsert(
        collection_name=config.COLLECTION_WEB_STATE,
        points=[PointStruct(id=_state_point_id(web_bank_id), vector=_STATE_VECTOR, payload=payload)],
    )
    return payload


async def store_chunks(
    web_bank_id: str,
    name: str,
    opd_id: str,
    url: str,
    title: Optional[str],
    child_chunks: List[ChunkItem],
    content_hash: str,
    is_active: bool = True,
) -> int:
    """Forward chunks ke LightRAG (ingestion murni, tanpa Qdrant legacy)."""

    # Fire-and-forget: sync ke LightRAG
    clean_content = "\n\n".join(
        chunk.text for chunk in child_chunks if chunk.text and len(chunk.text) >= 20
    )
    fire_lightrag_sync_web(
        source_id=web_bank_id,
        url=url,
        title=title or name,
        clean_content=clean_content,
        content_hash=content_hash,
        is_active=is_active,
    )

    logger.info(
        f"[SYNC] Synced {len(child_chunks)} chunks to LightRAG "
        f"for web_bank_id={web_bank_id}"
    )
    return len(child_chunks)


async def get_chunks_by_web_bank_id(
    web_bank_id: str,
    include_deleted: bool = False,
    chunk_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get chunks by web_bank_id."""
    base_filter = _chunk_filter(web_bank_id, include_deleted=include_deleted)
    must_conditions = list(base_filter.must or [])
    if chunk_level:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="chunk_level",
                match=qdrant_models.MatchValue(value=chunk_level)
            )
        )

    results = await qdrant.scroll(
        collection_name=config.COLLECTION_WEB,
        scroll_filter=qdrant_models.Filter(
            must=must_conditions or None,
            min_should=base_filter.min_should,
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    chunks = [{"id": point.id, **(point.payload or {})} for point in results[0]]
    chunks.sort(key=lambda item: (item.get("chunk_level") != "parent", item.get("chunk_index", 0)))
    return chunks


async def restore_chunks_by_web_bank_id(web_bank_id: str, is_active: bool = True) -> int:
    """Restore previously soft-deleted chunks for a web_bank_id."""
    chunks = await get_chunks_by_web_bank_id(web_bank_id, include_deleted=True)
    if not chunks:
        return 0

    point_ids = [chunk["id"] for chunk in chunks]
    now = _utcnow_iso()
    await qdrant.set_payload(
        collection_name=config.COLLECTION_WEB,
        payload={"is_deleted": False, "is_active": is_active, "deleted_at": None, "updated_at": now},
        points=point_ids,
    )
    logger.info(f"[SYNC] Restored {len(point_ids)} soft-deleted chunks for web_bank_id={web_bank_id}")
    return len(point_ids)


async def update_chunk_metadata(
    web_bank_id: str,
    *,
    name: Optional[str] = None,
    opd_id: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> int:
    """Update top-level metadata fields on existing chunks without rewriting content."""
    chunks = await get_chunks_by_web_bank_id(web_bank_id, include_deleted=True)
    if not chunks:
        return 0

    payload = {"updated_at": _utcnow_iso()}
    if name is not None:
        payload["name"] = name
    if opd_id is not None:
        payload["opd_id"] = opd_id
    if url is not None:
        payload["url"] = url
    if title is not None:
        payload["title"] = title
    if is_active is not None:
        payload["is_active"] = is_active

    point_ids = [chunk["id"] for chunk in chunks]
    await qdrant.set_payload(
        collection_name=config.COLLECTION_WEB,
        payload=payload,
        points=point_ids,
    )
    logger.info(f"[SYNC] Updated metadata on {len(point_ids)} chunks for web_bank_id={web_bank_id}")
    return len(point_ids)


async def soft_delete_by_web_bank_id(web_bank_id: str) -> int:
    """Soft delete chunks by web_bank_id."""
    logger.info(f"[SYNC] Soft deleted (LightRAG only) chunks for web_bank_id={web_bank_id}")

    # Fire-and-forget: hapus dari LightRAG
    fire_lightrag_delete(source_type="web", source_id=web_bank_id)

    return 1


async def hard_delete_by_web_bank_id(web_bank_id: str) -> int:
    """Hard delete chunks by web_bank_id."""
    logger.info(f"[SYNC] Hard deleted (LightRAG only) chunks for web_bank_id={web_bank_id}")

    # Fire-and-forget: hapus dari LightRAG
    fire_lightrag_delete(source_type="web", source_id=web_bank_id)

    return 1


async def _send_callback_once(
    sent_callbacks: set[tuple[str, str]],
    web_bank_id: str,
    url: str,
    status: str,
    result: Dict[str, Any],
    *,
    job_id: Optional[str] = None,
    started_at: Optional[float] = None,
) -> bool:
    from services.rag_web.webhook import build_callback_payload, send_callback

    payload = build_callback_payload(status, result)
    signature = (payload["scrape_status"], payload["scrape_message"])
    if signature in sent_callbacks:
        logger.info(
            f"[CALLBACK] Skip duplicate callback web_bank_id={web_bank_id} "
            f"status={payload['scrape_status']}"
        )
        return False

    _log_stage(
        web_bank_id,
        "callback",
        "Mengirim callback ke WA manajemen",
        job_id=job_id,
        started_at=started_at,
        scrape_status=payload["scrape_status"],
    )
    sent_callbacks.add(signature)
    callback_ok = await send_callback(web_bank_id, url, status, result)
    _log_stage(
        web_bank_id,
        "callback",
        "Callback selesai diproses",
        job_id=job_id,
        started_at=started_at,
        status_sent=payload["scrape_status"],
        delivered=callback_ok,
    )
    return callback_ok


async def register_inactive_web_bank(
    web_bank_id: str,
    name: str,
    opd_id: str,
    url: str,
    css_selector: Optional[str] = None,
    scrape_interval: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register or update a web bank as inactive without scraping."""
    existing_state = await get_web_state(web_bank_id) or {}
    await update_chunk_metadata(
        web_bank_id,
        name=name,
        opd_id=opd_id,
        is_active=False,
    )
    await upsert_web_state(
        web_bank_id,
        {
            "name": name,
            "opd_id": opd_id,
            "url": url,
            "css_selector": css_selector,
            "scrape_interval": scrape_interval,
            "is_active": False,
            "last_scrape_status": existing_state.get("last_scrape_status", "inactive"),
            "last_scrape_message": "Web bank tersimpan dalam status nonaktif. Scraping tidak dijalankan.",
            "metadata": metadata if metadata is not None else existing_state.get("metadata") or {},
        },
    )
    logger.info(f"[SYNC] Registered inactive web bank without scraping: {web_bank_id}")
    return {
        "status": "inactive",
        "web_bank_id": web_bank_id,
        "message": "Web bank nonaktif. Metadata tersimpan tanpa scraping.",
    }


async def process_url(
    web_bank_id: str,
    name: str,
    opd_id: str,
    url: str,
    css_selector: Optional[str] = None,
    scrape_interval: Optional[int] = None,
    is_active: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Pipeline: scrape -> clean/extract -> dedup -> chunk -> embed -> store."""
    sent_callbacks: set[tuple[str, str]] = set()
    existing_state = await get_web_state(web_bank_id) or {}
    was_inactive = not bool(existing_state.get("is_active", True))
    started_at = time.monotonic()
    current_stage = "start"

    try:
        logger.info(
            f"[SYNC] process_url start: web_bank_id={web_bank_id}, opd_id={opd_id}, "
            f"url={url}, css_selector={css_selector}, scrape_interval={scrape_interval}"
        )
        _log_stage(
            web_bank_id,
            "start",
            "Pipeline scraping dimulai",
            job_id=job_id,
            started_at=started_at,
            url=url,
            css_selector=css_selector or "-",
            scrape_interval=scrape_interval,
        )

        await upsert_web_state(
            web_bank_id,
            {
                "name": name,
                "opd_id": opd_id,
                "url": url,
                "css_selector": css_selector,
                "scrape_interval": scrape_interval,
                "is_active": is_active,
                "last_job_id": job_id,
                "last_scrape_status": "scraping",
                "last_scrape_message": "Sedang mengambil konten dari halaman web...",
            },
        )
        await _send_callback_once(
            sent_callbacks,
            web_bank_id,
            url,
            "scraping",
            {"scrape_message": "Sedang mengambil konten dari halaman web..."},
            job_id=job_id,
            started_at=started_at,
        )

        try:
            from services.rag_web.rate_limiter import rate_limiter

            current_stage = "rate_limit"
            _log_stage(
                web_bank_id,
                "rate_limit",
                "Menunggu rate limiter per domain jika diperlukan",
                job_id=job_id,
                started_at=started_at,
            )
            await rate_limiter.wait_for_domain(url)
        except Exception as exc:
            logger.warning(f"[SYNC] Rate limiter error (ignored): {exc}")

        current_stage = "fetch"
        _log_stage(
            web_bank_id,
            "fetch",
            "Mengambil HTML halaman web",
            job_id=job_id,
            started_at=started_at,
        )
        scraped = await scraper.scrape(url=url, use_js_renderer=None, wait_selector=None)
        raw_html = scraped.get("raw_html", "")
        _log_stage(
            web_bank_id,
            "fetch",
            "HTML berhasil diambil",
            job_id=job_id,
            started_at=started_at,
            status_code=scraped.get("status_code"),
            html_chars=len(raw_html),
        )
        title = cleaner.extract_title(raw_html) or existing_state.get("title")
        _log_stage(
            web_bank_id,
            "extract_title",
            "Judul halaman berhasil diekstrak",
            job_id=job_id,
            started_at=started_at,
            title=(title or "-")[:80],
        )

        current_stage = "clean"
        _log_stage(
            web_bank_id,
            "clean",
            "Membersihkan dan mengekstrak konten utama",
            job_id=job_id,
            started_at=started_at,
            selector_mode=bool(css_selector),
        )
        if css_selector:
            clean_content = cleaner.clean_with_selector(raw_html, css_selector, url)
        else:
            clean_content = cleaner.clean(raw_html, url)

        clean_content = (clean_content or "").strip()
        _log_stage(
            web_bank_id,
            "clean",
            "Konten utama berhasil dibersihkan",
            job_id=job_id,
            started_at=started_at,
            clean_chars=len(clean_content),
        )
        if len(clean_content) < 50:
            raise Exception(
                f"konten terlalu pendek atau kosong setelah cleaning ({len(clean_content)} chars)"
            )

        paragraph_count = _count_paragraphs(clean_content)
        content_hash = _content_hash(clean_content)
        scraped_at = _utcnow_iso()
        _log_stage(
            web_bank_id,
            "hash",
            "Hash konten berhasil dihitung",
            job_id=job_id,
            started_at=started_at,
            paragraph_count=paragraph_count,
            content_hash=content_hash[:12],
        )

        previous_hash = existing_state.get("last_content_hash")
        previous_chunks = int(existing_state.get("chunks_count") or 0)
        _log_stage(
            web_bank_id,
            "dedup",
            "Membandingkan hash dengan scrape sukses terakhir",
            job_id=job_id,
            started_at=started_at,
            previous_hash=(previous_hash[:12] if previous_hash else "-"),
            previous_chunks=previous_chunks,
        )
        if previous_hash and previous_hash == content_hash:
            restored_chunks = 0
            if was_inactive:
                restored_chunks = await restore_chunks_by_web_bank_id(web_bank_id, is_active=is_active)
            updated_chunks = await update_chunk_metadata(
                web_bank_id,
                name=name,
                opd_id=opd_id,
                url=url,
                title=title,
                is_active=is_active,
            )
            result = {
                "status": "success",
                "web_bank_id": web_bank_id,
                "url": url,
                "title": title,
                "page_count": 1,
                "paragraph_count": paragraph_count,
                "chunks_count": previous_chunks,
                "content_hash": content_hash,
                "dedup_skipped": True,
                "restored_chunks": restored_chunks,
                "updated_chunks": updated_chunks,
            }
            await upsert_web_state(
                web_bank_id,
                {
                    "name": name,
                    "opd_id": opd_id,
                    "url": url,
                    "css_selector": css_selector,
                    "scrape_interval": scrape_interval,
                    "title": title,
                    "is_active": is_active,
                    "last_scrape_status": "scraped",
                    "last_scrape_message": "Konten tidak berubah, indeks tidak diperbarui.",
                    "last_scraped_at": scraped_at,
                    "last_success_at": existing_state.get("last_success_at") or scraped_at,
                    "last_content_hash": content_hash,
                    "indexed_url": url,
                    "indexed_css_selector": css_selector,
                    "paragraph_count": paragraph_count,
                    "chunks_count": previous_chunks,
                    "metadata": metadata or existing_state.get("metadata") or {},
                },
            )
            _log_stage(
                web_bank_id,
                "dedup",
                "Konten tidak berubah, re-index dilewati",
                job_id=job_id,
                started_at=started_at,
                paragraph_count=paragraph_count,
                chunks_count=previous_chunks,
                restored_chunks=restored_chunks,
            )
            await _send_callback_once(
                sent_callbacks,
                web_bank_id,
                url,
                "completed",
                result,
                job_id=job_id,
                started_at=started_at,
            )
            _log_stage(
                web_bank_id,
                "done",
                "Pipeline selesai dengan dedup skip",
                job_id=job_id,
                started_at=started_at,
            )
            return result

        current_stage = "chunking"
        _log_stage(
            web_bank_id,
            "chunking",
            "Mengekstrak block HTML dan menyusun parent-child chunk",
            job_id=job_id,
            started_at=started_at,
        )
        chunk_items = web_chunker.chunk_html(raw_html, css_selector=css_selector)
        parent_chunks = [item for item in chunk_items if item.chunk_level == "parent"]
        child_chunks = [item for item in chunk_items if item.chunk_level == "child"]
        if not child_chunks:
            raise Exception("tidak ada chunk yang berhasil dibuat")
        _log_stage(
            web_bank_id,
            "chunking",
            "Chunking selesai",
            job_id=job_id,
            started_at=started_at,
            parent_chunks=len(parent_chunks),
            chunks_count=len(child_chunks),
        )

        current_stage = "cleanup"
        _log_stage(
            web_bank_id,
            "cleanup",
            "Membersihkan index lama sebelum sync baru",
            job_id=job_id,
            started_at=started_at,
        )
        await hard_delete_by_web_bank_id(web_bank_id)
        current_stage = "upsert"
        _log_stage(
            web_bank_id,
            "upsert",
            "Mengirim chunk baru ke LightRAG",
            job_id=job_id,
            started_at=started_at,
            chunks_count=len(child_chunks),
        )
        await store_chunks(
            web_bank_id=web_bank_id,
            name=name,
            opd_id=opd_id,
            url=url,
            title=title,
            child_chunks=child_chunks,
            content_hash=content_hash,
            is_active=is_active,
        )

        result = {
            "status": "success",
            "web_bank_id": web_bank_id,
            "url": url,
            "title": title,
            "page_count": 1,
            "paragraph_count": paragraph_count,
            "chunks_count": len(child_chunks),
            "content_length": sum(len(chunk.text) for chunk in child_chunks),
            "content_hash": content_hash,
            "dedup_skipped": False,
        }

        await upsert_web_state(
            web_bank_id,
            {
                "name": name,
                "opd_id": opd_id,
                "url": url,
                "css_selector": css_selector,
                "scrape_interval": scrape_interval,
                "title": title,
                "is_active": is_active,
                "last_scrape_status": "scraped",
                "last_scrape_message": (
                    f"Halaman berhasil di-scrape. Total konten: 1 halaman, "
                    f"{paragraph_count} paragraf, {len(child_chunks)} chunk"
                ),
                "last_scraped_at": scraped_at,
                "last_success_at": scraped_at,
                "last_content_hash": content_hash,
                "indexed_url": url,
                "indexed_css_selector": css_selector,
                "paragraph_count": paragraph_count,
                "chunks_count": len(child_chunks),
                "metadata": metadata or {},
            },
        )
        _log_stage(
            web_bank_id,
            "state",
            "State scrape terakhir berhasil diperbarui",
            job_id=job_id,
            started_at=started_at,
            content_hash=content_hash[:12],
            paragraph_count=paragraph_count,
            chunks_count=len(child_chunks),
        )

        logger.info(
            f"[SYNC] Selesai: {len(parent_chunks)} parent + {len(child_chunks)} child chunks "
            f"untuk web_bank_id={web_bank_id} (paragraphs={paragraph_count})"
        )
        await _send_callback_once(
            sent_callbacks,
            web_bank_id,
            url,
            "completed",
            result,
            job_id=job_id,
            started_at=started_at,
        )
        _log_stage(
            web_bank_id,
            "done",
            "Pipeline scraping selesai",
            job_id=job_id,
            started_at=started_at,
            paragraph_count=paragraph_count,
            chunks_count=len(child_chunks),
        )
        return result

    except Exception as exc:
        logger.exception(f"[SYNC] Error processing URL web_bank_id={web_bank_id}: {exc}")
        failed_at = _utcnow_iso()
        failure = _classify_processing_error(
            exc,
            current_stage=current_stage,
            css_selector=css_selector,
        )
        error_message = failure["user_message"]
        detail_message = failure["detail_message"]

        await upsert_web_state(
            web_bank_id,
            {
                "name": name,
                "opd_id": opd_id,
                "url": url,
                "css_selector": css_selector,
                "scrape_interval": scrape_interval,
                "is_active": is_active,
                "last_scrape_status": "failed",
                "last_scrape_message": error_message,
                "last_scraped_at": failed_at,
                "last_error_source": failure["error_source"],
                "last_error_code": failure["error_code"],
                "last_error_detail": detail_message,
                "metadata": metadata or existing_state.get("metadata") or {},
            },
        )

        error_result = {
            "status": "failed",
            "web_bank_id": web_bank_id,
            "url": url,
            "error": error_message,
            "error_source": failure["error_source"],
            "error_code": failure["error_code"],
            "error_detail": detail_message,
            "failed_stage": current_stage,
        }
        _log_stage(
            web_bank_id,
            "failed",
            "Pipeline scraping gagal",
            job_id=job_id,
            started_at=started_at,
            level="error",
            error=error_message,
            error_source=failure["error_source"],
            error_code=failure["error_code"],
            failed_stage=current_stage,
        )
        await _send_callback_once(
            sent_callbacks,
            web_bank_id,
            url,
            "failed",
            error_result,
            job_id=job_id,
            started_at=started_at,
        )
        return error_result

    finally:
        release_job(web_bank_id)
        _log_stage(
            web_bank_id,
            "finalize",
            "Slot scraping dilepas",
            job_id=job_id,
            started_at=started_at,
        )


async def update_web_bank(
    web_bank_id: str,
    name: str,
    opd_id: str,
    url: str,
    css_selector: Optional[str] = None,
    scrape_interval: Optional[int] = None,
    is_active: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    background_tasks: Any = None,
) -> Dict[str, Any]:
    """Update metadata and only rescrape when source settings changed."""
    existing_state = await get_web_state(web_bank_id) or {}
    existing_chunks = await get_chunks_by_web_bank_id(web_bank_id, include_deleted=True, chunk_level="child")

    if not existing_state and not existing_chunks:
        return {
            "status": "not_found",
            "web_bank_id": web_bank_id,
            "message": "Web bank tidak ditemukan",
        }

    sample = existing_state or (existing_chunks[0] if existing_chunks else {})
    indexed_url = _normalize_optional_text(existing_state.get("indexed_url") or sample.get("url"))
    indexed_selector = _normalize_optional_text(existing_state.get("indexed_css_selector") or sample.get("css_selector"))
    new_url = _normalize_optional_text(url)
    new_selector = _normalize_optional_text(css_selector)
    has_indexed_content = bool(existing_chunks) or bool(existing_state.get("last_content_hash"))
    source_changed = (not has_indexed_content) or (indexed_url != new_url) or (indexed_selector != new_selector)
    metadata_updates = metadata if metadata is not None else existing_state.get("metadata") or {}

    if not is_active:
        updated_chunks = await update_chunk_metadata(
            web_bank_id,
            name=name,
            opd_id=opd_id,
            is_active=False,
        )
        await upsert_web_state(
            web_bank_id,
            {
                "name": name,
                "opd_id": opd_id,
                "url": url,
                "title": sample.get("title"),
                "css_selector": css_selector,
                "scrape_interval": scrape_interval,
                "is_active": False,
                "last_scrape_status": existing_state.get("last_scrape_status", "inactive"),
                "last_scrape_message": "Web bank dinonaktifkan. Data tidak ikut domain pencarian RAG.",
                "metadata": metadata_updates,
            },
        )
        logger.info(
            f"[UPDATE] Web bank set inactive for web_bank_id={web_bank_id} "
            f"(updated_chunks={updated_chunks})"
        )
        return {
            "status": "success",
            "web_bank_id": web_bank_id,
            "message": "Web bank berhasil dinonaktifkan tanpa menghapus data",
            "updated_chunks": updated_chunks,
        }

    if source_changed:
        if background_tasks is None:
            return {
                "status": "error",
                "web_bank_id": web_bank_id,
                "message": "Background task handler tidak tersedia untuk rescrape",
            }
        if not reserve_job(web_bank_id):
            return {
                "status": "skipped",
                "message": "Scraping untuk website ini masih berjalan",
                "web_bank_id": web_bank_id,
            }

        job_id = f"update-{web_bank_id}-{int(time.time())}"
        background_tasks.add_task(
            process_url,
            web_bank_id=web_bank_id,
            name=name,
            opd_id=opd_id,
            url=url,
            css_selector=css_selector,
            scrape_interval=scrape_interval,
            is_active=True,
            metadata=metadata,
            job_id=job_id,
        )
        logger.info(
            f"[UPDATE] Source changed for web_bank_id={web_bank_id}; scheduling rescrape "
            f"(url_changed={indexed_url != new_url}, css_changed={indexed_selector != new_selector})"
        )
        return {
            "status": "processing",
            "web_bank_id": web_bank_id,
            "job_id": job_id,
            "message": "Perubahan URL/CSS selector terdeteksi, scraping ulang dijalankan",
        }

    title = sample.get("title")
    updated_chunks = await update_chunk_metadata(
        web_bank_id,
        name=name,
        opd_id=opd_id,
        title=title,
        is_active=True,
    )
    if existing_state.get("is_active") is False:
        await restore_chunks_by_web_bank_id(web_bank_id, is_active=True)

    await upsert_web_state(
        web_bank_id,
        {
            "name": name,
            "opd_id": opd_id,
            "url": url,
            "title": title,
            "css_selector": css_selector,
            "scrape_interval": scrape_interval,
            "is_active": True,
            "last_scrape_status": existing_state.get("last_scrape_status", "scraped"),
            "last_scrape_message": "Metadata web bank berhasil diperbarui tanpa scraping ulang.",
            "indexed_url": existing_state.get("indexed_url") or sample.get("url"),
            "indexed_css_selector": existing_state.get("indexed_css_selector") or sample.get("css_selector"),
            "metadata": metadata_updates,
        },
    )
    logger.info(
        f"[UPDATE] Metadata updated without rescrape for web_bank_id={web_bank_id} "
        f"(updated_chunks={updated_chunks})"
    )
    return {
        "status": "success",
        "web_bank_id": web_bank_id,
        "message": "Metadata web bank berhasil diperbarui tanpa scraping ulang",
        "updated_chunks": updated_chunks,
    }


async def soft_delete_web_bank(web_bank_id: str) -> Dict[str, Any]:
    """Soft delete a web bank and its indexed chunks."""
    _log_stage(web_bank_id, "delete", "Melakukan soft delete web bank")
    deleted_chunks = await soft_delete_by_web_bank_id(web_bank_id)
    existing_state = await get_web_state(web_bank_id) or {}
    await upsert_web_state(
        web_bank_id,
        {
            "name": existing_state.get("name"),
            "opd_id": existing_state.get("opd_id"),
            "url": existing_state.get("url"),
            "css_selector": existing_state.get("css_selector"),
            "scrape_interval": existing_state.get("scrape_interval"),
            "is_active": False,
            "last_scrape_status": "deleted",
            "last_scrape_message": "Web bank dihapus dari WA manajemen (soft delete).",
            "metadata": existing_state.get("metadata") or {},
        },
    )

    return {
        "status": "success",
        "web_bank_id": web_bank_id,
        "deleted_chunks": deleted_chunks,
        "message": "Web bank berhasil di-soft-delete",
    }


async def disable_web_bank(web_bank_id: str) -> Dict[str, Any]:
    """Backward-compatible alias for soft delete flow."""
    return await soft_delete_web_bank(web_bank_id)


async def sync_edited_content(
    web_bank_id: str,
    edited_content: str,
) -> Dict[str, Any]:
    """Sync edited content (user-edited)."""
    started_at = time.monotonic()
    try:
        logger.info(f"[SYNC] Syncing edited content for web_bank_id={web_bank_id}")
        _log_stage(
            web_bank_id,
            "edit_sync",
            "Memulai sinkronisasi konten hasil edit",
            started_at=started_at,
        )
        existing_chunks = await get_chunks_by_web_bank_id(web_bank_id, include_deleted=True)
        existing_state = await get_web_state(web_bank_id) or {}

        if not existing_chunks and not existing_state:
            return {
                "status": "not_found",
                "web_bank_id": web_bank_id,
                "error": "Content not found",
            }

        sample = existing_chunks[0] if existing_chunks else existing_state
        url = sample.get("url", "")
        title = sample.get("title", "")
        name = sample.get("name", existing_state.get("name", ""))
        opd_id = sample.get("opd_id", existing_state.get("opd_id", ""))
        is_active = existing_state.get("is_active", sample.get("is_active", True))

        clean_content = (edited_content or "").strip()
        _log_stage(
            web_bank_id,
            "edit_sync",
            "Menyusun parent-child chunk dari konten hasil edit",
            started_at=started_at,
            clean_chars=len(clean_content),
        )
        chunk_items = web_chunker.chunk_text(clean_content)
        parent_chunks = [item for item in chunk_items if item.chunk_level == "parent"]
        child_chunks = [item for item in chunk_items if item.chunk_level == "child"]
        if not child_chunks:
            return {
                "status": "error",
                "web_bank_id": web_bank_id,
                "error": "No chunks created from edited content",
            }

        content_hash = _content_hash(clean_content)
        paragraph_count = _count_paragraphs(clean_content)
        scraped_at = _utcnow_iso()

        _log_stage(
            web_bank_id,
            "edit_sync",
            "Mengganti chunk lama dengan hasil edit",
            started_at=started_at,
            chunks_count=len(child_chunks),
        )
        await hard_delete_by_web_bank_id(web_bank_id)
        await store_chunks(
            web_bank_id=web_bank_id,
            name=name,
            opd_id=opd_id,
            url=url,
            title=title,
            child_chunks=child_chunks,
            content_hash=content_hash,
            is_active=is_active,
        )

        await upsert_web_state(
            web_bank_id,
            {
                "name": name,
                "opd_id": opd_id,
                "url": url,
                "title": title,
                "is_active": is_active,
                "last_scrape_status": "scraped",
                "last_scrape_message": "Konten hasil edit berhasil disinkronkan.",
                "last_scraped_at": scraped_at,
                "last_success_at": scraped_at,
                "last_content_hash": content_hash,
                "indexed_url": url,
                "indexed_css_selector": existing_state.get("css_selector"),
                "paragraph_count": paragraph_count,
                "chunks_count": len(child_chunks),
                "metadata": {"is_edited": True, "edited_at": scraped_at},
            },
        )

        logger.info(
            f"[SYNC] Updated: {len(parent_chunks)} parent + {len(child_chunks)} child chunks "
            f"for web_bank_id={web_bank_id}"
        )
        _log_stage(
            web_bank_id,
            "edit_sync",
            "Sinkronisasi konten hasil edit selesai",
            started_at=started_at,
            paragraph_count=paragraph_count,
            chunks_count=len(child_chunks),
        )
        return {
            "status": "success",
            "web_bank_id": web_bank_id,
            "chunks_count": len(child_chunks),
        }

    except Exception as exc:
        logger.exception(f"[SYNC] Error syncing edited content: {exc}")
        _log_stage(
            web_bank_id,
            "edit_sync_failed",
            "Sinkronisasi konten hasil edit gagal",
            started_at=started_at,
            level="error",
            error=str(exc),
        )
        return {
            "status": "error",
            "web_bank_id": web_bank_id,
            "error": str(exc),
        }


async def get_content(web_bank_id: str) -> Dict[str, Any]:
    """Get combined content by web_bank_id."""
    chunks = await get_chunks_by_web_bank_id(web_bank_id, chunk_level="child")
    if not chunks:
        return {
            "status": "not_found",
            "web_bank_id": web_bank_id,
        }

    clean_content = "\n\n".join(chunk.get("content", "") for chunk in chunks)
    sample = chunks[0]
    return {
        "status": "success",
        "web_bank_id": web_bank_id,
        "url": sample.get("url", ""),
        "title": sample.get("title", ""),
        "clean_content": clean_content,
        "chunks_count": len(chunks),
        "created_at": sample.get("created_at", ""),
        "updated_at": sample.get("updated_at", ""),
    }
