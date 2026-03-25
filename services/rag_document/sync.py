"""
Document Sync - RAG Medan v3
Background OCR orchestration with real-time progress monitoring.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

logger = logging.getLogger("rag_document.sync")

ALLOWED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".jpg", ".jpeg", ".png"]
_PROGRESS_PREFIX = "__PROGRESS__"

task_status: Dict[str, dict] = {}
task_lock = threading.Lock()
_ocr_semaphore = threading.BoundedSemaphore(2)

_SYNC_STATUS_MAP = {
    "completed": "synced",
    "error": "failed",
    "timeout": "failed",
    "processing": "syncing",
    "queued": "syncing",
}


def _cleanup_old_tasks(max_age_hours: int = 24):
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
    total_chunks: Optional[int] = None,
):
    entry = {
        "doc_id": doc_id,
        "task_id": task_id,
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if total_chunks is not None:
        entry["total_chunks"] = total_chunks

    with task_lock:
        task_status[task_id] = entry

    logger.info(f"[TASK] {task_id} -> {status} | {message[:120] if message else ''}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    with task_lock:
        return task_status.get(task_id)


def get_all_tasks() -> Dict[str, dict]:
    with task_lock:
        return {"tasks": dict(task_status)}


def _send_callback(
    doc_id: str,
    task_id: str,
    status: str,
    message: str,
    total_chunks: Optional[int] = None,
):
    callback_url = config.DOCUMENT_CALLBACK_URL
    if not callback_url:
        return

    import requests as req

    sync_status = _SYNC_STATUS_MAP.get(status, "failed")
    payload = {
        "sync_status": sync_status,
        "sync_message": message,
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
                verify=False,
            )
            if resp.status_code < 400:
                logger.info(f"[CALLBACK] Berhasil ({resp.status_code}) untuk doc_id={doc_id} -> {sync_status}")
                return
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
    total_chunks: Optional[int] = None,
):
    update_task_status(task_id, doc_id, status, message, total_chunks)
    _send_callback(doc_id, task_id, status, message, total_chunks)


def _parse_progress_line(line: str) -> Optional[dict]:
    if not line.startswith(_PROGRESS_PREFIX):
        return None
    try:
        return json.loads(line[len(_PROGRESS_PREFIX):])
    except Exception:
        logger.warning(f"[SUBPROCESS] Gagal parse progress line: {line[:200]}")
        return None


def _parse_worker_result(stdout_text: str) -> Optional[dict]:
    text = (stdout_text or "").strip()
    if not text:
        return None

    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _consume_worker_stderr(
    stream,
    task_id: str,
    doc_id: str,
    state: dict,
):
    for raw_line in iter(stream.readline, ""):
        line = raw_line.strip()
        if not line:
            continue

        state["last_output_at"] = time.time()
        progress = _parse_progress_line(line)
        if progress:
            state["last_progress_at"] = time.time()
            state["last_progress_message"] = progress.get("message", "")
            progress_message = progress.get("message") or "Dokumen sedang diproses."
            total_chunks = progress.get("total_chunks")
            update_task_status(task_id, doc_id, "processing", progress_message, total_chunks=total_chunks)
            logger.info(f"[WORKER-PROGRESS] {task_id} | {progress_message}")
        else:
            logger.info(f"[WORKER-LOG] {task_id} | {line}")


def _wait_for_worker_slot(task_id: str, doc_id: str) -> None:
    logger.info(f"[SUBPROCESS] Mengantri semaphore untuk {task_id}...")
    queue_started_at = time.time()
    last_log_at = 0.0

    while True:
        acquired = _ocr_semaphore.acquire(timeout=1)
        if acquired:
            waited_sec = int(time.time() - queue_started_at)
            logger.info(f"[SUBPROCESS] Semaphore acquired untuk {task_id} setelah {waited_sec}s")
            return

        waited_sec = int(time.time() - queue_started_at)
        if waited_sec - last_log_at >= config.OCR_QUEUE_LOG_INTERVAL:
            last_log_at = waited_sec
            update_task_status(
                task_id,
                doc_id,
                "queued",
                f"Dokumen sedang antri untuk diproses. Menunggu slot OCR {waited_sec} detik.",
            )
            logger.info(f"[SUBPROCESS] {task_id} masih menunggu slot OCR ({waited_sec}s)")


def run_ocr_subprocess(task_id: str, params: dict):
    doc_id = params.get("doc_id", "")
    stall_timeout = max(60, config.OCR_STALL_TIMEOUT)
    hard_timeout = max(stall_timeout, config.OCR_HARD_TIMEOUT)
    acquired = False

    _cleanup_old_tasks()

    _send_callback(
        doc_id=doc_id,
        task_id=task_id,
        status="processing",
        message="Dokumen sedang diproses oleh OCR pipeline.",
    )

    try:
        _wait_for_worker_slot(task_id, doc_id)
        acquired = True
        update_task_status(task_id, doc_id, "processing", "Subprocess OCR dimulai...")

        worker_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "worker.py"))
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(worker_script)))

        process = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            env={**os.environ, "PYTHONPATH": work_dir},
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        process.stdin.write(json.dumps(params))
        process.stdin.close()

        state = {
            "last_progress_at": time.time(),
            "last_output_at": time.time(),
            "last_progress_message": "Subprocess OCR dimulai...",
        }

        stderr_thread = threading.Thread(
            target=_consume_worker_stderr,
            args=(process.stderr, task_id, doc_id, state),
            daemon=True,
        )
        stderr_thread.start()

        started_at = time.time()
        while process.poll() is None:
            time.sleep(2)
            now = time.time()
            if hard_timeout and (now - started_at) > hard_timeout:
                process.kill()
                msg = (
                    f"Proses OCR melampaui batas maksimum ({hard_timeout // 60} menit). "
                    "Silakan hubungi administrator."
                )
                logger.error(f"[SUBPROCESS] Hard timeout untuk {task_id}")
                _finalize_task(task_id, doc_id, "timeout", msg)
                return

            if (now - state["last_progress_at"]) > stall_timeout:
                process.kill()
                last_msg = state.get("last_progress_message") or "Tidak ada progress"
                msg = (
                    "Proses OCR berhenti memberi progress terlalu lama. "
                    f"Progress terakhir: {last_msg}"
                )
                logger.error(f"[SUBPROCESS] Stall timeout untuk {task_id}")
                _finalize_task(task_id, doc_id, "timeout", msg)
                return

        stderr_thread.join(timeout=5)
        stdout_text = process.stdout.read() if process.stdout else ""

        if process.returncode != 0:
            logger.error(f"[SUBPROCESS] Process crashed (exit {process.returncode}) untuk {task_id}")
            _finalize_task(
                task_id,
                doc_id,
                "error",
                "Gagal memproses dokumen. Silakan coba lagi atau hubungi administrator.",
            )
            return

        result = _parse_worker_result(stdout_text)
        if result is None:
            logger.error(f"[SUBPROCESS] JSON output worker tidak valid: {stdout_text[:300]}")
            _finalize_task(
                task_id,
                doc_id,
                "error",
                "Gagal memproses dokumen. Pastikan format file didukung dan coba lagi.",
            )
            return

        pipeline_status = result.get("status")
        total_chunks = result.get("total_chunks")

        if pipeline_status == "ok":
            n = total_chunks or 0
            _finalize_task(
                task_id,
                doc_id,
                "completed",
                f"Dokumen berhasil disinkronkan. {n} chunk berhasil terindeks.",
                total_chunks=n,
            )
        elif pipeline_status == "duplicate":
            _finalize_task(
                task_id,
                doc_id,
                "completed",
                "Dokumen sudah tersinkronkan sebelumnya (konten duplikat), tidak ada perubahan.",
            )
        elif pipeline_status == "reactivated":
            n = total_chunks or 0
            _finalize_task(
                task_id,
                doc_id,
                "completed",
                f"Dokumen berhasil diaktifkan kembali. {n} chunk telah dipulihkan.",
                total_chunks=n,
            )
        elif pipeline_status == "error":
            internal_detail = result.get("message", "")
            logger.error(f"[SUBPROCESS] Pipeline error untuk {task_id}: {internal_detail}")
            _finalize_task(
                task_id,
                doc_id,
                "error",
                "Gagal memproses dokumen. Pastikan format file didukung dan coba lagi.",
            )
        else:
            logger.warning(f"[SUBPROCESS] Status pipeline tidak dikenal: {pipeline_status}")
            _finalize_task(
                task_id,
                doc_id,
                "error",
                "Terjadi kesalahan pada server RAG. Silakan coba beberapa saat lagi.",
            )

    except Exception as e:
        logger.exception(f"[SUBPROCESS] Unexpected error untuk {task_id}: {e}")
        _finalize_task(
            task_id,
            doc_id,
            "error",
            "Terjadi kesalahan pada server RAG. Silakan coba beberapa saat lagi.",
        )
    finally:
        if acquired:
            _ocr_semaphore.release()
            logger.info(f"[SUBPROCESS] Semaphore released untuk {task_id}")


async def sync_document(
    doc_id: str,
    file_url: str,
    organization_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("[API] ========== DOC-SYNC START ==========")
    logger.info(f"[API] doc_id={doc_id} | org={organization_id} | filename={filename}")
    logger.info(f"[API] file_url={file_url}")

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
            ),
        }

    file_path = file_url
    if file_path.startswith("file://"):
        file_path = file_path.replace("file://", "")

    file_size_mb = 0
    if not file_path.startswith(("http://", "https://")):
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File tidak ditemukan: {file_path}"}
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        logger.info(f"[API] File lokal: {file_path} ({file_size_mb:.2f} MB)")

    task_id = f"{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    update_task_status(task_id, doc_id, "queued", "Dokumen sedang antri untuk diproses.")

    params = {
        "task_id": task_id,
        "doc_id": doc_id,
        "organization_id": organization_id,
        "filename": filename,
        "file_url": file_path,
        "lang": "id",
        "collection_name": config.COLLECTION_DOCUMENT,
    }

    thread = threading.Thread(
        target=run_ocr_subprocess,
        args=(task_id, params),
        daemon=True,
    )
    thread.start()

    logger.info(f"[API] Task queued: {task_id}")

    response = {
        "status": "queued",
        "task_id": task_id,
        "message": "Dokumen sedang diproses. Gunakan GET /internal/sync/status/{task_id} untuk cek status.",
    }
    if file_size_mb:
        response["file_size_mb"] = file_size_mb
    return response
