"""
Document Worker - RAG Medan v3
Subprocess OCR pipeline. Menerima params via stdin JSON, output final via stdout JSON.
"""
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests
import urllib3
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.ocr_utils import (
    build_blocks_from_extracted_pages,
    calculate_content_hash,
    calculate_file_hash,
    extract_blocks_from_file,
    extract_text_from_file,
)
from services.rag_document.chunker import ChunkItem, structure_chunk_document

logger = setup_logging("document_worker")

for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.setStream(sys.stderr)

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PROGRESS_PREFIX = "__PROGRESS__"
_EMBED_BATCH_SIZE = 32
_UPSERT_BATCH_SIZE = 50


def emit_progress(stage: str, message: str, **extra) -> None:
    payload = {
        "stage": stage,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    payload.update(extra)
    print(f"{_PROGRESS_PREFIX}{json.dumps(payload, ensure_ascii=False)}", file=sys.stderr, flush=True)


def _safe_pct(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return max(0, min(100, int((numerator / denominator) * 100)))


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-.]", "", name)
    name = re.sub(r"\s+", "-", name.strip())
    return name[:80]


def _resolve_file(
    url: str,
    doc_id: str = "doc",
    user_filename: Optional[str] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Tuple[str, bool]:
    """
    Resolve URL file ke path lokal.

    - File lokal: return (path, False)
    - File remote: download ke document_temp/{YYYY-MM-DD}/, return (path, True)
    """
    if not url.startswith(("http://", "https://")):
        return (url, False)

    parsed_url = urlparse(url)
    url_filename = unquote(Path(parsed_url.path).name)

    if user_filename:
        user_ext = os.path.splitext(user_filename)[1]
        url_ext = os.path.splitext(url_filename)[1]
        if not user_ext and url_ext:
            safe_name = _sanitize_filename(user_filename) + url_ext
        else:
            safe_name = _sanitize_filename(user_filename) + (user_ext or url_ext)
    else:
        safe_name = _sanitize_filename(url_filename) if url_filename else "document"

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H%M%S")
    temp_dir = _PROJECT_ROOT / "document_temp" / date_str
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_doc_id = doc_id.replace("/", "_")[:36]
    temp_file_path = str(temp_dir / f"{safe_doc_id}_{time_str}_{safe_name}")

    logger.info(f"[DOWNLOAD] Mengunduh dari {url}")
    logger.info(f"[DOWNLOAD] Simpan ke {temp_file_path}")
    if progress_callback:
        progress_callback(stage="downloading", message="Memulai download file dokumen...")

    try:
        http_response = requests.get(
            url,
            timeout=(10, config.OCR_TIMEOUT),
            verify=False,
            allow_redirects=True,
            stream=True,
        )

        if http_response.status_code == 404:
            raise FileNotFoundError(f"File tidak ditemukan di server (404): {url}")
        if http_response.status_code == 403:
            raise PermissionError(f"Akses ditolak (403): {url}")
        if http_response.status_code >= 500:
            raise ConnectionError(f"Server mengalami gangguan (HTTP {http_response.status_code}): {url}")
        if not http_response.ok:
            raise IOError(f"Gagal mengunduh file (HTTP {http_response.status_code}): {url}")

        downloaded_bytes = 0
        content_length = int(http_response.headers.get("content-length", "0") or 0)
        progress_step_bytes = max(1, config.OCR_DOWNLOAD_PROGRESS_MB) * 1024 * 1024
        next_progress_mark = progress_step_bytes
        last_progress_log = time.time()

        with open(temp_file_path, "wb") as f:
            for chunk in http_response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded_bytes += len(chunk)

                now = time.time()
                if progress_callback and (
                    downloaded_bytes >= next_progress_mark or
                    (now - last_progress_log) >= 10
                ):
                    if content_length > 0:
                        msg = (
                            f"Download dokumen {_safe_pct(downloaded_bytes, content_length)}% "
                            f"({downloaded_bytes // (1024 * 1024)}MB/{content_length // (1024 * 1024)}MB)"
                        )
                    else:
                        msg = f"Download dokumen {downloaded_bytes // (1024 * 1024)}MB"
                    progress_callback(
                        stage="downloading",
                        message=msg,
                        downloaded_bytes=downloaded_bytes,
                        total_bytes=content_length,
                    )
                    next_progress_mark = downloaded_bytes + progress_step_bytes
                    last_progress_log = now

        logger.info(f"[DOWNLOAD] Selesai: {downloaded_bytes / 1024:.1f} KB")
        if progress_callback:
            progress_callback(
                stage="downloading",
                message="Download dokumen selesai.",
                downloaded_bytes=downloaded_bytes,
                total_bytes=content_length,
            )
        return (temp_file_path, True)

    except requests.exceptions.ConnectTimeout:
        raise ConnectionError(f"Koneksi timeout saat menghubungi server: {url}")
    except requests.exceptions.ReadTimeout:
        raise TimeoutError(f"Download melebihi batas baca ({config.OCR_TIMEOUT} detik): {url}")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Tidak dapat terhubung ke server: {url} - {e}")
    except (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, IOError):
        raise
    except Exception as e:
        raise IOError(f"Gagal mengunduh file: {url} - {e}")


def check_duplicate_by_file_hash(
    qdrant: QdrantClient,
    collection_name: str,
    file_hash: str,
) -> dict:
    try:
        results, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="file_hash",
                        match=qdrant_models.MatchValue(value=file_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if results:
            is_deleted = results[0].payload.get("is_deleted", False)
            return {"exists": True, "point": results[0], "is_deleted": is_deleted}
    except Exception as e:
        logger.warning(f"[DEDUP] Gagal cek file_hash: {e}")
    return {"exists": False, "point": None, "is_deleted": False}


def check_duplicate_by_content_hash(
    qdrant: QdrantClient,
    collection_name: str,
    content_hash: str,
) -> dict:
    try:
        results, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="content_hash",
                        match=qdrant_models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if results:
            is_deleted = results[0].payload.get("is_deleted", False)
            return {"exists": True, "point": results[0], "is_deleted": is_deleted}
    except Exception as e:
        logger.warning(f"[DEDUP] Gagal cek content_hash: {e}")
    return {"exists": False, "point": None, "is_deleted": False}


def reactivate_document(
    qdrant: QdrantClient,
    collection_name: str,
    doc_id: str,
    *,
    is_active: bool = True,
) -> int:
    all_point_ids = []
    offset = None
    now = datetime.utcnow().isoformat()

    while True:
        results, next_offset = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="mysql_id",
                        match=qdrant_models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=False,
        )
        all_point_ids.extend([p.id for p in results])
        if next_offset is None:
            break
        offset = next_offset

    if all_point_ids:
        qdrant.set_payload(
            collection_name=collection_name,
            payload={
                "is_active": is_active,
                "is_deleted": False,
                "deleted_at": None,
                "reactivated_at": now,
            },
            points=all_point_ids,
        )
        logger.info(f"[REACTIVATE] {len(all_point_ids)} chunk dipulihkan untuk doc_id={doc_id}")

    return len(all_point_ids)


def _build_chunk_items(
    file_ext: str,
    blocks: List[dict],
    file_path: str,
) -> List[ChunkItem]:
    return structure_chunk_document(
        blocks,
        child_chunk_size=config.DOC_CHILD_CHUNK_SIZE,
        parent_chunk_size=config.DOC_PARENT_CHUNK_SIZE,
        overlap=config.DOC_CHUNK_OVERLAP,
        # PDF diizinkan semantic merge agar block OCR yang terfragmentasi
        # dapat digabung. Merge tetap dibatasi oleh heading_path di chunker.
        enable_semantic_merge=config.ENABLE_SEMANTIC_MERGE and file_ext not in [".jpg", ".jpeg", ".png", ".xlsx", ".xls"],
        similarity_threshold=config.SEMANTIC_MERGE_SIM_THRESHOLD,
    )


def _extract_structured_blocks_for_worker(
    file_ext: str,
    file_path: str,
    *,
    extracted_pages: dict,
    lang: str,
    progress_callback: Callable[..., None],
) -> List[dict]:
    if file_ext == ".pdf":
        return extract_blocks_from_file(
            file_path,
            lang=lang,
            progress_callback=progress_callback,
        )

    if file_ext in [".txt", ".jpg", ".jpeg", ".png"]:
        source_kind = "ocr" if file_ext in [".jpg", ".jpeg", ".png"] else "narrative"
        return build_blocks_from_extracted_pages(
            extracted_pages,
            source_kind=source_kind,
        )

    return extract_blocks_from_file(
        file_path,
        lang=lang,
        progress_callback=progress_callback,
    )


def _get_existing_doc_id(point: Optional[object]) -> Optional[str]:
    payload = getattr(point, "payload", None) or {}
    if isinstance(payload, dict):
        return payload.get("mysql_id")
    return None


def _embed_with_shared_service(texts: List[str]) -> List[List[float]]:
    response = requests.post(
        f"{config.SHARED_EMBEDDING_URL.rstrip('/')}/embed",
        json={
            "texts": texts,
            "prefix": "passage: ",
            "model_size": "large",
        },
        headers={"X-API-Key": config.INTERNAL_API_KEY},
        timeout=max(120, config.OCR_TIMEOUT),
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("embeddings", [])


def _embed_chunks(
    chunk_items: List[dict],
    progress_callback: Callable[..., None],
) -> List[List[float]]:
    total_chunks = len(chunk_items)
    all_embeddings: List[List[float]] = []
    local_model: Optional[SentenceTransformer] = None
    use_shared = config.USE_SHARED_EMBEDDING

    for start in range(0, total_chunks, _EMBED_BATCH_SIZE):
        end = min(start + _EMBED_BATCH_SIZE, total_chunks)
        batch_items = chunk_items[start:end]
        batch_texts = [item["text"] for item in batch_items]
        progress_callback(
            stage="embedding",
            message=f"Embedding chunk {start + 1}-{end}/{total_chunks}...",
            processed_chunks=start,
            total_chunks=total_chunks,
        )

        if use_shared:
            try:
                batch_embeddings = _embed_with_shared_service(batch_texts)
            except Exception as e:
                logger.warning(f"[EMBED] Shared embedding gagal, fallback lokal: {e}")
                progress_callback(
                    stage="embedding",
                    message="Shared embedding tidak tersedia, fallback ke model lokal...",
                    processed_chunks=start,
                    total_chunks=total_chunks,
                )
                use_shared = False
                local_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)
                batch_embeddings = local_model.encode(
                    [f"passage: {text}" for text in batch_texts],
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
        else:
            if local_model is None:
                local_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)
            batch_embeddings = local_model.encode(
                [f"passage: {text}" for text in batch_texts],
                batch_size=len(batch_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

        all_embeddings.extend(batch_embeddings)
        progress_callback(
            stage="embedding",
            message=f"Embedding selesai {end}/{total_chunks} chunk.",
            processed_chunks=end,
            total_chunks=total_chunks,
        )

    return all_embeddings


def process_document(
    task_id: str,
    doc_id: str,
    organization_id: Optional[str],
    filename: Optional[str],
    file_url: str,
    collection_name: str,
    is_active: bool = True,
    lang: str = "id",
    skip_dedup_check: bool = False,
) -> dict:
    logger.info(f"[WORKER] ========== START task={task_id} doc_id={doc_id} ==========")

    local_file_path = None
    is_temp_file = False

    def progress_callback(stage: str, message: str, **extra) -> None:
        logger.info(f"[PROGRESS] task={task_id} stage={stage} | {message}")
        emit_progress(stage, message, task_id=task_id, doc_id=doc_id, **extra)

    try:
        progress_callback("starting", "Pipeline dokumen dimulai.")

        try:
            local_file_path, is_temp_file = _resolve_file(
                file_url,
                doc_id,
                filename,
                progress_callback=progress_callback,
            )
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"[WORKER] File error: {e}")
            return {"status": "error", "message": str(e)}
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"[WORKER] Network error: {e}")
            return {"status": "error", "message": str(e)}
        except IOError as e:
            logger.error(f"[WORKER] IO error: {e}")
            return {"status": "error", "message": str(e)}

        if filename:
            doc_ext = os.path.splitext(local_file_path)[1]
            user_ext = os.path.splitext(filename)[1]
            document_filename = filename if user_ext else filename + doc_ext
        else:
            document_filename = Path(local_file_path).name

        file_ext = os.path.splitext(local_file_path)[1].lower()

        try:
            file_hash = calculate_file_hash(local_file_path)
            logger.info(f"[WORKER] file_hash={file_hash[:16]}...")
        except Exception as e:
            logger.warning(f"[WORKER] Gagal hitung file_hash: {e}")
            file_hash = ""

        progress_callback("dedup", "Menghubungkan ke Qdrant dan memeriksa duplikasi awal...")
        qdrant = _connect_qdrant()
        _ensure_collection(qdrant, collection_name)

        if not skip_dedup_check and file_hash:
            dedup_result = check_duplicate_by_file_hash(qdrant, collection_name, file_hash)
            if dedup_result["exists"]:
                existing_doc_id = _get_existing_doc_id(dedup_result.get("point")) or doc_id
                if dedup_result["is_deleted"]:
                    logger.info(f"[WORKER] file_hash match (soft-deleted) - reactivating doc_id={existing_doc_id}")
                    n = reactivate_document(qdrant, collection_name, existing_doc_id, is_active=is_active)
                    return {"status": "reactivated", "total_chunks": n, "message": ""}
                logger.info("[WORKER] file_hash match (aktif) - duplicate, skip OCR")
                return {"status": "duplicate", "total_chunks": 0, "message": ""}

        progress_callback("extracting", "Mengekstrak teks dokumen...")
        extracted_pages = extract_text_from_file(
            local_file_path,
            lang=lang,
            return_pages=True,
            progress_callback=progress_callback,
        )
        text = "\n\n".join(
            page_text for _, page_text in sorted(extracted_pages.items(), key=lambda x: x[0]) if page_text
        ).strip()

        if not text or len(text.strip()) < 50:
            return {"status": "error", "message": "Tidak ada teks yang berhasil diekstrak atau konten terlalu pendek."}

        logger.info(f"[WORKER] Ekstraksi selesai: {len(text)} karakter")
        content_hash = calculate_content_hash(text)

        if not skip_dedup_check:
            progress_callback("dedup", "Memeriksa duplikasi berdasarkan konten hasil ekstraksi...")
            content_dedup = check_duplicate_by_content_hash(qdrant, collection_name, content_hash)
            if content_dedup["exists"]:
                existing_doc_id = _get_existing_doc_id(content_dedup.get("point")) or doc_id
                if content_dedup["is_deleted"]:
                    logger.info(f"[WORKER] content_hash match (soft-deleted) - reactivating doc_id={existing_doc_id}")
                    n = reactivate_document(qdrant, collection_name, existing_doc_id, is_active=is_active)
                    return {"status": "reactivated", "total_chunks": n, "message": ""}
                logger.info("[WORKER] content_hash match (aktif) - duplicate")
                return {"status": "duplicate", "total_chunks": 0, "message": ""}

        progress_callback("chunking", "Mengekstrak block terstruktur dokumen...")
        structured_blocks = _extract_structured_blocks_for_worker(
            file_ext,
            local_file_path,
            extracted_pages=extracted_pages,
            lang=lang,
            progress_callback=progress_callback,
        )
        logger.info(f"[WORKER] Structured blocks: {len(structured_blocks)}")

        progress_callback("chunking", f"Menyusun parent-child chunk (ext={file_ext})...")
        chunk_items = _build_chunk_items(file_ext, structured_blocks, local_file_path)
        if not chunk_items:
            return {"status": "error", "message": "Tidak ada chunk yang berhasil dibuat dari konten dokumen."}

        parent_chunks = [item for item in chunk_items if item.chunk_level == "parent"]
        child_chunks = [item for item in chunk_items if item.chunk_level == "child"]
        total_chunks = len(child_chunks)
        logger.info(
            f"[WORKER] {len(parent_chunks)} parent chunk + {len(child_chunks)} child chunk dibuat"
        )
        progress_callback(
            "chunking",
            f"Chunking selesai: {len(parent_chunks)} parent, {len(child_chunks)} child chunk.",
            total_chunks=total_chunks,
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
        )

        embeddings = _embed_chunks(
            [{"page_number": item.page_start, "text": item.text} for item in child_chunks],
            progress_callback=progress_callback,
        )

        progress_callback("upserting", f"Menghapus chunk lama untuk doc_id={doc_id}...")
        try:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="mysql_id",
                                match=qdrant_models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
            )
        except Exception:
            pass

        now = datetime.utcnow().isoformat()
        points = []

        points_by_id = {}

        for parent_item in parent_chunks:
            parent_payload = {
                "mysql_id": doc_id,
                "organization_id": organization_id,
                "opd": organization_id,
                "filename": document_filename,
                "text": parent_item.text,
                "page_number": parent_item.page_start,
                "chunk_index": parent_item.block_order,
                "total_chunks": total_chunks,
                "section": parent_item.section_title,
                "summary": "",
                "file_hash": file_hash,
                "content_hash": content_hash,
                "chunk_id": parent_item.chunk_id,
                "chunk_level": parent_item.chunk_level,
                "chunk_kind": parent_item.chunk_kind,
                "source_kind": parent_item.source_kind,
                "section_title": parent_item.section_title,
                "heading_path": parent_item.heading_path,
                "page_start": parent_item.page_start,
                "page_end": parent_item.page_end,
                "parent_chunk_id": None,
                "window_prev_id": None,
                "window_next_id": None,
                "is_active": is_active,
                "is_deleted": False,
                "created_at": now,
                "updated_at": now,
                **parent_item.metadata,
            }
            parent_point = PointStruct(
                id=parent_item.chunk_id,
                vector=[0.0] * config.EMBEDDING_DIMENSION_LARGE,
                payload=parent_payload,
            )
            points_by_id[parent_item.chunk_id] = parent_point

        for i, (chunk_item, embedding) in enumerate(zip(child_chunks, embeddings)):
            payload = {
                "mysql_id": doc_id,
                "organization_id": organization_id,
                "opd": organization_id,
                "filename": document_filename,
                "text": chunk_item.text,
                "page_number": chunk_item.page_start,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "section": chunk_item.section_title,
                "summary": "",
                "file_hash": file_hash,
                "content_hash": content_hash,
                "chunk_id": chunk_item.chunk_id,
                "chunk_level": chunk_item.chunk_level,
                "chunk_kind": chunk_item.chunk_kind,
                "source_kind": chunk_item.source_kind,
                "section_title": chunk_item.section_title,
                "heading_path": chunk_item.heading_path,
                "page_start": chunk_item.page_start,
                "page_end": chunk_item.page_end,
                "parent_chunk_id": chunk_item.parent_chunk_id,
                "window_prev_id": chunk_item.window_prev_id,
                "window_next_id": chunk_item.window_next_id,
                "is_active": is_active,
                "is_deleted": False,
                "created_at": now,
                "updated_at": now,
                **chunk_item.metadata,
            }
            vector = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            points.append(PointStruct(id=chunk_item.chunk_id, vector=vector, payload=payload))

            if len(points) >= _UPSERT_BATCH_SIZE:
                qdrant.upsert(collection_name=collection_name, points=points)
                points = []
                progress_callback(
                    "upserting",
                    f"Upsert selesai {i + 1}/{total_chunks} chunk.",
                    processed_chunks=i + 1,
                    total_chunks=total_chunks,
                )

        if points_by_id:
            qdrant.upsert(collection_name=collection_name, points=list(points_by_id.values()))

        if points:
            qdrant.upsert(collection_name=collection_name, points=points)
            progress_callback(
                "upserting",
                f"Upsert selesai {total_chunks}/{total_chunks} chunk.",
                processed_chunks=total_chunks,
                total_chunks=total_chunks,
            )

        progress_callback("completed", f"Pipeline selesai. {total_chunks} chunk berhasil terindeks.", total_chunks=total_chunks)
        logger.info(f"[WORKER] ========== DONE task={task_id} chunks={total_chunks} ==========")
        return {"status": "ok", "total_chunks": total_chunks, "message": ""}

    except Exception as e:
        logger.exception(f"[WORKER] Unexpected error task={task_id}: {e}")
        return {"status": "error", "message": f"Kesalahan tidak terduga: {type(e).__name__}"}

    finally:
        if is_temp_file and local_file_path and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
                logger.info(f"[WORKER] Temp file dihapus: {local_file_path}")
            except Exception as e:
                logger.warning(f"[WORKER] Gagal hapus temp file: {e}")


def _connect_qdrant() -> QdrantClient:
    if config.QDRANT_API_KEY:
        return QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
        )
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def _ensure_collection(qdrant: QdrantClient, collection_name: str):
    collections = qdrant.get_collections()
    existing = [c.name for c in collections.collections]
    if collection_name not in existing:
        logger.info(f"[WORKER] Membuat collection: {collection_name}")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSION_LARGE,
                distance=Distance.COSINE,
            ),
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="mysql_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="is_deleted",
            field_schema=qdrant_models.PayloadSchemaType.BOOL,
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="is_active",
            field_schema=qdrant_models.PayloadSchemaType.BOOL,
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_level",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="parent_chunk_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )


def main():
    try:
        raw = sys.stdin.read()
        params = json.loads(raw)
    except Exception as e:
        result = {"status": "error", "message": f"Gagal parse input JSON: {e}"}
        print(json.dumps(result), flush=True)
        sys.exit(1)

    task_id = params.get("task_id", "unknown")
    doc_id = params.get("doc_id", "")
    organization_id = params.get("organization_id")
    filename = params.get("filename")
    file_url = params.get("file_url", "")
    collection_name = params.get("collection_name", config.COLLECTION_DOCUMENT)
    is_active = params.get("is_active", True)
    lang = params.get("lang", "id")
    skip_dedup = params.get("skip_dedup_check", False)

    if not doc_id or not file_url:
        result = {"status": "error", "message": "Parameter doc_id dan file_url wajib diisi."}
        print(json.dumps(result), flush=True)
        sys.exit(1)

    result = process_document(
        task_id=task_id,
        doc_id=doc_id,
        organization_id=organization_id,
        filename=filename,
        file_url=file_url,
        collection_name=collection_name,
        is_active=is_active,
        lang=lang,
        skip_dedup_check=skip_dedup,
    )

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
