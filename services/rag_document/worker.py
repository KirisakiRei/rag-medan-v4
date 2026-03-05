"""
Document Worker — RAG Medan v3
Subprocess OCR pipeline. Menerima params via stdin JSON, output via stdout JSON.

Ported & updated from rag-medan-v2 core/document_pipeline.py + core/ocr_worker.py

Bug fixes dari v3 lama:
  - B1: import extract_text_from_file (bukan extract_text_from_pdf/image/docx/xlsx yang tidak ada)
  - B2: worker tidak import dari sync.py — output via stdout JSON saja
  - B3: baca params dari stdin JSON (bukan argparse CLI args)
  - B4: payload Qdrant menggunakan field names yang seragam dengan search.py

Fitur baru dari v2:
  - _resolve_file(): SSL bypass, timeout terpisah, streaming write, error handling spesifik
  - Temp file di document_temp/{YYYY-MM-DD}/ dengan nama bermakna, auto-cleanup
  - Two-layer dedup: file_hash (pre-OCR) + content_hash (post-OCR)
  - reactivate_document(): pulihkan chunk soft-deleted tanpa re-OCR
  - Dispatch chunking: .xlsx -> chunk_xlsx(), lainnya -> semantic_chunk()
  - Batch embedding: model.encode(batch_size=32, normalize_embeddings=True)
  - Payload per chunk: mysql_id, organization_id, filename, text, file_hash, is_deleted, dll
"""
import os
import sys
import json
import uuid
import re
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from urllib.parse import urlparse, unquote

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.ocr_utils import extract_text_from_file, calculate_file_hash, calculate_content_hash
from services.rag_document.chunker import semantic_chunk, chunk_xlsx

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct, Distance, VectorParams

logger = setup_logging("document_worker")

# Root direktori project (2 level di atas services/)
_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================
# File Download & Temp Management
# ============================================================

def _sanitize_filename(name: str) -> str:
    """Sanitasi nama file agar aman untuk path."""
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '-', name.strip())
    return name[:80]  # potong jika terlalu panjang


def _resolve_file(
    url: str,
    doc_id: str = "doc",
    user_filename: Optional[str] = None
) -> Tuple[str, bool]:
    """
    Resolve URL file ke path lokal.

    - File lokal: return (path, False)
    - File remote: download ke document_temp/{YYYY-MM-DD}/, return (path, True)

    Returns:
        (local_path, is_temp) — is_temp=True berarti file harus dihapus setelah proses
    """
    # File lokal
    if not url.startswith(("http://", "https://")):
        return (url, False)

    # Tentukan nama file
    parsed_url = urlparse(url)
    url_filename = unquote(Path(parsed_url.path).name)

    if user_filename:
        # Pakai user_filename, tambah ekstensi dari URL jika belum ada
        user_ext = os.path.splitext(user_filename)[1]
        url_ext = os.path.splitext(url_filename)[1]
        if not user_ext and url_ext:
            safe_name = _sanitize_filename(user_filename) + url_ext
        else:
            safe_name = _sanitize_filename(user_filename) + (user_ext or url_ext)
    else:
        safe_name = _sanitize_filename(url_filename) if url_filename else "document"

    # Buat folder temp per tanggal
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H%M%S')
    temp_dir = _PROJECT_ROOT / "document_temp" / date_str
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_doc_id = doc_id.replace("/", "_")[:36]
    temp_file_path = str(temp_dir / f"{safe_doc_id}_{time_str}_{safe_name}")

    logger.info(f"[DOWNLOAD] Mengunduh dari {url}")
    logger.info(f"[DOWNLOAD] Simpan ke {temp_file_path}")

    try:
        http_response = requests.get(
            url,
            timeout=(10, 120),   # connect 10s, read 120s
            verify=False,         # bypass SSL cert untuk LAN self-signed
            allow_redirects=True,
            stream=True           # streaming — RAM konstan ~8 KB
        )

        # Error handling spesifik per HTTP status
        if http_response.status_code == 404:
            raise FileNotFoundError(f"File tidak ditemukan di server (404): {url}")
        elif http_response.status_code == 403:
            raise PermissionError(f"Akses ditolak (403): {url}")
        elif http_response.status_code >= 500:
            raise ConnectionError(f"Server mengalami gangguan (HTTP {http_response.status_code}): {url}")
        elif not http_response.ok:
            raise IOError(f"Gagal mengunduh file (HTTP {http_response.status_code}): {url}")

        # Tulis per chunk ke disk — RAM konstan berapapun ukuran file
        downloaded_bytes = 0
        with open(temp_file_path, "wb") as f:
            for chunk in http_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

        logger.info(f"[DOWNLOAD] Selesai: {downloaded_bytes / 1024:.1f} KB")
        return (temp_file_path, True)

    except requests.exceptions.ConnectTimeout:
        raise ConnectionError(f"Koneksi timeout saat menghubungi server: {url}")
    except requests.exceptions.ReadTimeout:
        raise TimeoutError(f"Download melebihi 120 detik (file terlalu besar atau koneksi lambat): {url}")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Tidak dapat terhubung ke server: {url} — {e}")
    except (FileNotFoundError, PermissionError, ConnectionError, TimeoutError, IOError):
        raise  # re-raise error yang sudah spesifik
    except Exception as e:
        raise IOError(f"Gagal mengunduh file: {url} — {e}")


# ============================================================
# Deduplication
# ============================================================

def check_duplicate_by_file_hash(
    qdrant: QdrantClient,
    collection_name: str,
    file_hash: str
) -> dict:
    """
    Cek duplikat berdasarkan file_hash di Qdrant (pre-OCR check).

    Returns:
        {"exists": bool, "point": PointStruct|None, "is_deleted": bool}
    """
    try:
        results, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="file_hash",
                        match=qdrant_models.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=True
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
    content_hash: str
) -> dict:
    """
    Cek duplikat berdasarkan content_hash di Qdrant (post-OCR check).

    Returns:
        {"exists": bool, "is_deleted": bool}
    """
    try:
        results, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="content_hash",
                        match=qdrant_models.MatchValue(value=content_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        if results:
            is_deleted = results[0].payload.get("is_deleted", False)
            return {"exists": True, "is_deleted": is_deleted}
    except Exception as e:
        logger.warning(f"[DEDUP] Gagal cek content_hash: {e}")
    return {"exists": False, "is_deleted": False}


# ============================================================
# Reactivate (tanpa re-OCR)
# ============================================================

def reactivate_document(qdrant: QdrantClient, collection_name: str, doc_id: str) -> int:
    """
    Pulihkan semua chunk soft-deleted milik doc_id.
    Menggunakan pagination loop — menangani dokumen dengan >100 chunk.

    Returns:
        Jumlah chunk yang dipulihkan
    """
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
                        match=qdrant_models.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=False
        )
        all_point_ids.extend([p.id for p in results])
        if next_offset is None:
            break
        offset = next_offset

    if all_point_ids:
        qdrant.set_payload(
            collection_name=collection_name,
            payload={"is_deleted": False, "deleted_at": None, "reactivated_at": now},
            points=all_point_ids
        )
        logger.info(f"[REACTIVATE] {len(all_point_ids)} chunk dipulihkan untuk doc_id={doc_id}")

    return len(all_point_ids)


# ============================================================
# XLSX row extraction untuk chunker
# ============================================================

def _extract_xlsx_rows(file_path: str):
    """
    Ekstrak rows dari XLSX menggunakan read_only=True (streaming).

    Returns:
        List[List[str]] — list baris, rows[0] = header
    """
    import openpyxl
    all_rows = []
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_data = [str(c) if c is not None else "" for c in row]
                all_rows.append(row_data)
        wb.close()
    except Exception as e:
        logger.warning(f"[XLSX] Gagal ekstrak rows dari {file_path}: {e}")
    return all_rows


# ============================================================
# Main pipeline
# ============================================================

def process_document(
    task_id: str,
    doc_id: str,
    organization_id: Optional[str],
    filename: Optional[str],
    file_url: str,
    collection_name: str,
    lang: str = "id",
    skip_dedup_check: bool = False
) -> dict:
    """
    Pipeline utama:
      Step 1  : Resolve file (download jika remote)
      Step 2.1: Layer 1 dedup — cek file_hash sebelum OCR
      Step 3  : OCR / extract text
      Step 4  : Layer 2 dedup — cek content_hash setelah OCR
      Step 5  : Chunking (xlsx -> chunk_xlsx, lainnya -> semantic_chunk)
      Step 6  : Batch embedding
      Step 7  : Upsert ke Qdrant
      cleanup : Hapus file temp via try/finally

    Returns:
        {"status": "ok|duplicate|reactivated|error", "total_chunks": int, "message": str}
    """
    logger.info(f"[WORKER] ========== START task={task_id} doc_id={doc_id} ==========")

    local_file_path = None
    is_temp_file = False

    try:
        # --- STEP 1: Resolve file ---
        try:
            local_file_path, is_temp_file = _resolve_file(file_url, doc_id, filename)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"[WORKER] File error: {e}")
            return {"status": "error", "message": str(e)}
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"[WORKER] Network error: {e}")
            return {"status": "error", "message": str(e)}
        except IOError as e:
            logger.error(f"[WORKER] IO error: {e}")
            return {"status": "error", "message": str(e)}

        # Tentukan document_filename (untuk payload)
        if filename:
            doc_ext = os.path.splitext(local_file_path)[1]
            user_ext = os.path.splitext(filename)[1]
            document_filename = filename if user_ext else filename + doc_ext
        else:
            document_filename = Path(local_file_path).name

        file_ext = os.path.splitext(local_file_path)[1].lower()

        # --- STEP 1.5: Hitung file_hash ---
        try:
            file_hash = calculate_file_hash(local_file_path)
            logger.info(f"[WORKER] file_hash={file_hash[:16]}...")
        except Exception as e:
            logger.warning(f"[WORKER] Gagal hitung file_hash: {e}")
            file_hash = ""

        # --- STEP 2.1: Layer 1 Dedup — file_hash (pre-OCR) ---
        logger.info(f"[WORKER] Menghubungkan ke Qdrant...")
        qdrant = _connect_qdrant()
        _ensure_collection(qdrant, collection_name)

        if not skip_dedup_check and file_hash:
            dedup_result = check_duplicate_by_file_hash(qdrant, collection_name, file_hash)
            if dedup_result["exists"]:
                if dedup_result["is_deleted"]:
                    # Reactivate tanpa re-OCR
                    logger.info(f"[WORKER] file_hash match (soft-deleted) — reactivating doc_id={doc_id}")
                    n = reactivate_document(qdrant, collection_name, doc_id)
                    return {"status": "reactivated", "total_chunks": n, "message": ""}
                else:
                    # Konten identik sudah aktif
                    logger.info(f"[WORKER] file_hash match (aktif) — duplicate, skip OCR")
                    return {"status": "duplicate", "total_chunks": 0, "message": ""}

        # --- STEP 3: Load model + OCR ---
        logger.info(f"[WORKER] Loading embedding model...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)

        logger.info(f"[WORKER] Mengekstrak teks dari {local_file_path}...")
        text = extract_text_from_file(local_file_path, lang=lang)

        if not text or len(text.strip()) < 50:
            return {"status": "error", "message": "Tidak ada teks yang berhasil diekstrak atau konten terlalu pendek."}

        logger.info(f"[WORKER] Ekstraksi selesai: {len(text)} karakter")

        # Hitung content_hash
        content_hash = calculate_content_hash(text)

        # --- STEP 4: Layer 2 Dedup — content_hash (post-OCR) ---
        if not skip_dedup_check:
            content_dedup = check_duplicate_by_content_hash(qdrant, collection_name, content_hash)
            if content_dedup["exists"]:
                if content_dedup["is_deleted"]:
                    logger.info(f"[WORKER] content_hash match (soft-deleted) — reactivating")
                    n = reactivate_document(qdrant, collection_name, doc_id)
                    return {"status": "reactivated", "total_chunks": n, "message": ""}
                else:
                    logger.info(f"[WORKER] content_hash match (aktif) — duplicate")
                    return {"status": "duplicate", "total_chunks": 0, "message": ""}

        # --- STEP 5: Chunking ---
        logger.info(f"[WORKER] Chunking (ext={file_ext})...")
        if file_ext in ['.xlsx', '.xls']:
            rows = _extract_xlsx_rows(local_file_path)
            chunks = chunk_xlsx(rows, rows_per_chunk=30, max_size=config.CHUNK_SIZE)
        else:
            chunks = semantic_chunk(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)

        if not chunks:
            return {"status": "error", "message": "Tidak ada chunk yang berhasil dibuat dari konten dokumen."}

        logger.info(f"[WORKER] {len(chunks)} chunk dibuat")

        # --- STEP 6: Batch embedding ---
        logger.info(f"[WORKER] Embedding {len(chunks)} chunks (batch_size=32)...")
        texts_with_prefix = [f"passage: {c}" for c in chunks]
        embeddings = model.encode(
            texts_with_prefix,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # --- STEP 7: Hapus chunk lama dan upsert ---
        logger.info(f"[WORKER] Menghapus chunk lama untuk doc_id={doc_id}...")
        try:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="mysql_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
        except Exception:
            pass  # Belum ada chunk — tidak masalah

        now = datetime.utcnow().isoformat()
        points = []

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "mysql_id":        doc_id,
                "organization_id": organization_id,
                "filename":        document_filename,
                "text":            chunk_text,
                "page_number":     1,
                "chunk_index":     i,
                "total_chunks":    len(chunks),
                "section":         "",
                "summary":         "",
                "file_hash":       file_hash,
                "content_hash":    content_hash,
                "is_deleted":      False,
                "created_at":      now,
                "updated_at":      now
            }
            points.append(PointStruct(id=point_id, vector=embedding.tolist(), payload=payload))

            # Batch upsert setiap 50 points
            if len(points) >= 50:
                qdrant.upsert(collection_name=collection_name, points=points)
                points = []
                logger.info(f"[WORKER] Upserted {i+1}/{len(chunks)} chunks...")

        # Final batch
        if points:
            qdrant.upsert(collection_name=collection_name, points=points)

        logger.info(f"[WORKER] ========== DONE task={task_id} chunks={len(chunks)} ==========")
        return {"status": "ok", "total_chunks": len(chunks), "message": ""}

    except Exception as e:
        logger.exception(f"[WORKER] Unexpected error task={task_id}: {e}")
        return {"status": "error", "message": f"Kesalahan tidak terduga: {type(e).__name__}"}

    finally:
        # Auto-cleanup file temp
        if is_temp_file and local_file_path and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
                logger.info(f"[WORKER] Temp file dihapus: {local_file_path}")
            except Exception as e:
                logger.warning(f"[WORKER] Gagal hapus temp file: {e}")


# ============================================================
# Qdrant helpers
# ============================================================

def _connect_qdrant() -> QdrantClient:
    """Buat koneksi sinkron ke Qdrant."""
    if config.QDRANT_API_KEY:
        return QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY
        )
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def _ensure_collection(qdrant: QdrantClient, collection_name: str):
    """Buat collection jika belum ada."""
    collections = qdrant.get_collections()
    existing = [c.name for c in collections.collections]
    if collection_name not in existing:
        logger.info(f"[WORKER] Membuat collection: {collection_name}")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSION_LARGE,
                distance=Distance.COSINE
            )
        )
        # Index payload fields untuk filter performa
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="mysql_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="is_deleted",
            field_schema=qdrant_models.PayloadSchemaType.BOOL
        )


# ============================================================
# Entry point — baca params dari stdin JSON
# ============================================================

def main():
    """
    Entry point subprocess.
    Menerima JSON params dari stdin, output JSON result ke stdout.
    """
    try:
        raw = sys.stdin.read()
        params = json.loads(raw)
    except Exception as e:
        result = {"status": "error", "message": f"Gagal parse input JSON: {e}"}
        print(json.dumps(result), flush=True)
        sys.exit(1)

    task_id        = params.get("task_id", "unknown")
    doc_id         = params.get("doc_id", "")
    organization_id = params.get("organization_id")
    filename       = params.get("filename")
    file_url       = params.get("file_url", "")
    collection_name = params.get("collection_name", config.COLLECTION_DOCUMENT)
    lang           = params.get("lang", "id")
    skip_dedup     = params.get("skip_dedup_check", False)

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
        lang=lang,
        skip_dedup_check=skip_dedup
    )

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
