"""
Document Sync — RAG Medan v3
Ported & updated from rag-medan-v2 routes/doc_sync_routes.py

Perubahan dari v3 lama:
  - organization_id + filename menggantikan opd_name
  - Semaphore BoundedSemaphore(2): max 2 subprocess OCR concurrent (cegah OOM)
  - task_id format: {doc_id}_{YYYYMMDD}_{HHMMSS} (bukan unix timestamp)
  - task_status struktur flat: {doc_id, task_id, status, message, total_chunks?, timestamp}
  - _finalize_task(): wrapper update_task_status + _send_callback
  - _send_callback(): PUT webhook ke wa manajemen, retry 3x, verify=False
  - _cleanup_old_tasks(): TTL 24 jam lazy cleanup (cegah memory leak)
  - Validasi ekstensi dari filename -> urlparse(file_url).path (fix URL query string)
  - OCR timeout dari config.OCR_TIMEOUT (bukan hardcoded 600s)
  - Status consistency fix: result["status"] == "error" -> outer "error"
"""
import os
import sys
import time
import logging
import subprocess
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

logger = logging.getLogger("rag_document.sync")

ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png']

# ============================================================
# Task status — in-memory, struktur flat
# ============================================================
task_status: Dict[str, dict] = {}
task_lock = threading.Lock()

# ============================================================
# Semaphore: max 2 subprocess OCR concurrent (cegah OOM)
# Setiap subprocess load e5-large (~1.3 GB RAM)
# ============================================================
_ocr_semaphore = threading.BoundedSemaphore(2)

# ============================================================
# Status mapping internal -> wa manajemen
# ============================================================
_SYNC_STATUS_MAP = {
    "completed":  "synced",
    "error":      "failed",
    "timeout":    "failed",
    "processing": "syncing",
    "queued":     "syncing",
}


# ============================================================
# Helpers
# ============================================================

def _cleanup_old_tasks(max_age_hours: int = 24):
    """Hapus entri task_status yang lebih tua dari max_age_hours (lazy cleanup)."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    with task_lock:
        to_delete = []
        for tid, entry in task_status.items():
            try:
                ts = datetime.fromisoformat(entry.get("timestamp", ""))
                if ts < cutoff:
                    to_delete.append(tid)
            except Exception:
                pass
        for tid in to_delete:
            del task_status[tid]
    if to_delete:
        logger.info(f"[CLEANUP] Hapus {len(to_delete)} task lama dari memory")


def update_task_status(
    task_id: str,
    doc_id: str,
    status: str,
    message: str = "",
    total_chunks: Optional[int] = None
):
    """Update task status dengan struktur flat."""
    entry = {
        "doc_id": doc_id,
        "task_id": task_id,
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    if total_chunks is not None:
        entry["total_chunks"] = total_chunks

    with task_lock:
        task_status[task_id] = entry

    logger.info(f"[TASK] {task_id} -> {status} | {message[:80] if message else ''}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status."""
    with task_lock:
        return task_status.get(task_id)


def get_all_tasks() -> Dict[str, dict]:
    """Get all tasks."""
    with task_lock:
        return {"tasks": dict(task_status)}


def _send_callback(
    doc_id: str,
    task_id: str,
    status: str,
    message: str,
    total_chunks: Optional[int] = None
):
    """Kirim webhook callback PUT ke file-banks/sync-status/{doc_id}."""
    callback_url = config.DOCUMENT_CALLBACK_URL
    if not callback_url:
        return

    import requests as req

    sync_status = _SYNC_STATUS_MAP.get(status, "failed")
    payload = {
        "sync_status": sync_status,
        "sync_message": message
    }

    headers = {"Content-Type": "application/json"}
    api_key = config.WEB_MANAJEMEN_API_KEY
    if api_key:
        headers["X-API-Key"] = api_key

    url = f"{callback_url.rstrip('/')}/{doc_id}"
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            resp = req.put(
                url,
                json=payload,
                headers=headers,
                timeout=10,
                verify=False  # LAN dengan self-signed cert
            )
            if resp.status_code < 400:
                logger.info(f"[CALLBACK] Berhasil ({resp.status_code}) untuk doc_id={doc_id} -> {sync_status}")
                return
            else:
                logger.warning(f"[CALLBACK] Attempt {attempt}/{max_attempts} HTTP {resp.status_code} untuk doc_id={doc_id}")
        except Exception as e:
            logger.warning(f"[CALLBACK] Attempt {attempt}/{max_attempts} error untuk doc_id={doc_id}: {e}")

        if attempt < max_attempts:
            time.sleep(5)

    logger.error(f"[CALLBACK] Semua {max_attempts} attempts gagal untuk doc_id={doc_id}")


def _finalize_task(
    task_id: str,
    doc_id: str,
    status: str,
    message: str,
    total_chunks: Optional[int] = None
):
    """Wrapper: update_task_status + _send_callback sekaligus."""
    update_task_status(task_id, doc_id, status, message, total_chunks)
    _send_callback(doc_id, task_id, status, message, total_chunks)


# ============================================================
# OCR Subprocess
# ============================================================

def run_ocr_subprocess(task_id: str, params: dict):
    """Run OCR dalam subprocess terpisah dengan semaphore kontrolling konkuren."""
    doc_id = params.get("doc_id", "")
    ocr_timeout = config.OCR_TIMEOUT  # default 1800s (30 menit)

    # Cleanup task lama sebelum mulai
    _cleanup_old_tasks()

    # Kirim callback awal: syncing
    _send_callback(
        doc_id=doc_id,
        task_id=task_id,
        status="processing",
        message="Dokumen sedang diproses oleh OCR pipeline."
    )

    # Acquire semaphore — tunggu jika sudah ada 2 subprocess berjalan
    logger.info(f"[SUBPROCESS] Mengantri semaphore untuk {task_id}...")
    acquired = _ocr_semaphore.acquire(timeout=ocr_timeout)
    if not acquired:
        msg = "Antrian OCR terlalu padat. Silakan coba lagi dalam beberapa menit."
        logger.error(f"[SUBPROCESS] Semaphore timeout untuk {task_id}")
        _finalize_task(task_id, doc_id, "error", msg)
        return

    try:
        logger.info(f"[SUBPROCESS] Semaphore acquired, memulai OCR untuk {task_id}")
        update_task_status(task_id, doc_id, "processing", "Subprocess OCR dimulai...")

        worker_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "worker.py")
        )
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(worker_script)))

        process = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            env={**os.environ, "PYTHONPATH": work_dir}
        )

        input_data = json.dumps(params).encode("utf-8")
        try:
            stdout, stderr = process.communicate(input=input_data, timeout=ocr_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            msg = (
                f"Proses OCR melampaui batas waktu ({ocr_timeout // 60} menit). "
                "Coba dengan file yang lebih kecil atau hubungi administrator."
            )
            logger.error(f"[SUBPROCESS] Timeout untuk {task_id}")
            _finalize_task(task_id, doc_id, "timeout", msg)
            return

        if stderr:
            stderr_text = stderr.decode("utf-8", errors="ignore")
            if stderr_text.strip():
                logger.debug(f"[SUBPROCESS] Stderr {task_id}: {stderr_text[:500]}")

        if process.returncode != 0:
            error_detail = stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
            logger.error(f"[SUBPROCESS] Process crashed (exit {process.returncode}): {error_detail[:300]}")
            _finalize_task(
                task_id, doc_id, "error",
                "Gagal memproses dokumen. Silakan coba lagi atau hubungi administrator."
            )
            return

        # Parse JSON output dari worker
        try:
            result = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as e:
            raw_out = stdout.decode("utf-8", errors="ignore")
            logger.error(f"[SUBPROCESS] JSON decode error: {e} | stdout: {raw_out[:300]}")
            _finalize_task(
                task_id, doc_id, "error",
                "Gagal memproses dokumen. Pastikan format file didukung dan coba lagi."
            )
            return

        pipeline_status = result.get("status")
        total_chunks = result.get("total_chunks")

        # Status consistency: mapping pipeline status -> outer task status
        if pipeline_status == "ok":
            n = total_chunks or 0
            _finalize_task(
                task_id, doc_id, "completed",
                f"Dokumen berhasil disinkronkan. {n} chunk berhasil terindeks.",
                total_chunks=n
            )
        elif pipeline_status == "duplicate":
            _finalize_task(
                task_id, doc_id, "completed",
                "Dokumen sudah tersinkronkan sebelumnya (konten duplikat), tidak ada perubahan."
            )
        elif pipeline_status == "reactivated":
            n = total_chunks or 0
            _finalize_task(
                task_id, doc_id, "completed",
                f"Dokumen berhasil diaktifkan kembali. {n} chunk telah dipulihkan.",
                total_chunks=n
            )
        elif pipeline_status == "error":
            internal_detail = result.get("message", "")
            logger.error(f"[SUBPROCESS] Pipeline error untuk {task_id}: {internal_detail}")
            _finalize_task(
                task_id, doc_id, "error",
                "Gagal memproses dokumen. Pastikan format file didukung dan coba lagi."
            )
        else:
            logger.warning(f"[SUBPROCESS] Status pipeline tidak dikenal: {pipeline_status}")
            _finalize_task(
                task_id, doc_id, "error",
                "Terjadi kesalahan pada server RAG. Silakan coba beberapa saat lagi."
            )

    except Exception as e:
        logger.exception(f"[SUBPROCESS] Unexpected error untuk {task_id}: {e}")
        _finalize_task(
            task_id, doc_id, "error",
            "Terjadi kesalahan pada server RAG. Silakan coba beberapa saat lagi."
        )
    finally:
        _ocr_semaphore.release()
        logger.info(f"[SUBPROCESS] Semaphore released untuk {task_id}")


# ============================================================
# Main sync entry point
# ============================================================

async def sync_document(
    doc_id: str,
    file_url: str,
    organization_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trigger OCR worker sebagai subprocess background.

    Returns immediately dengan status "queued".
    """
    logger.info(f"[API] ========== DOC-SYNC START ==========")
    logger.info(f"[API] doc_id={doc_id} | org={organization_id} | filename={filename}")
    logger.info(f"[API] file_url={file_url}")

    # Validasi ekstensi — prioritas dari filename -> urlparse(file_url).path
    file_ext = ""
    if filename:
        file_ext = os.path.splitext(filename)[1].lower()
    if not file_ext:
        parsed_url = urlparse(file_url)
        file_ext = os.path.splitext(parsed_url.path)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        return {
            "status": "error",
            "message": (
                f"Tipe file tidak didukung (ekstensi: '{file_ext}'). "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        }

    # Validasi file lokal (jika bukan URL remote)
    file_path = file_url
    if file_path.startswith("file://"):
        file_path = file_path.replace("file://", "")

    file_size_mb = 0
    if not file_path.startswith(("http://", "https://")):
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File tidak ditemukan: {file_path}"}
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        logger.info(f"[API] File lokal: {file_path} ({file_size_mb:.2f} MB)")

    # Generate task_id human-readable
    task_id = f"{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    update_task_status(task_id, doc_id, "queued", "Dokumen sedang antri untuk diproses.")

    # Params ke worker subprocess
    params = {
        "task_id": task_id,
        "doc_id": doc_id,
        "organization_id": organization_id,
        "filename": filename,
        "file_url": file_path,
        "lang": "id",
        "collection_name": config.COLLECTION_DOCUMENT
    }

    # Jalankan subprocess di background thread
    thread = threading.Thread(
        target=run_ocr_subprocess,
        args=(task_id, params),
        daemon=True
    )
    thread.start()

    logger.info(f"[API] Task queued: {task_id}")

    response = {
        "status": "queued",
        "task_id": task_id,
        "message": "Dokumen sedang diproses. Gunakan GET /internal/sync/status/{task_id} untuk cek status."
    }
    if file_size_mb:
        response["file_size_mb"] = file_size_mb
    return response
