import os
import sys
import time
import logging
import subprocess
import json
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

logger = logging.getLogger("rag_document.sync")

ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png']

# Task status tracker
task_status: Dict[str, dict] = {}
task_lock = threading.Lock()


def update_task_status(task_id: str, status: str, result: dict = None):
    """Update task status."""
    with task_lock:
        task_status[task_id] = {
            "status": status,
            "result": result,
            "updated_at": datetime.utcnow().isoformat()
        }
        logger.info(f"[TASK] {task_id} -> {status}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status."""
    with task_lock:
        return task_status.get(task_id)


def get_all_tasks() -> Dict[str, dict]:
    """Get all tasks."""
    with task_lock:
        return {"tasks": task_status}


def run_ocr_subprocess(task_id: str, params: dict):
    """Run OCR in a separate subprocess."""
    try:
        logger.info(f"[SUBPROCESS] Starting OCR subprocess for {task_id}")
        update_task_status(task_id, "processing")
        
        # Path to worker script
        worker_script = os.path.join(os.path.dirname(__file__), "worker.py")
        worker_script = os.path.abspath(worker_script)
        
        # Working directory
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(worker_script)))
        process = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            env={**os.environ, "PYTHONPATH": work_dir}
        )
        
        # Send params and wait for result
        input_data = json.dumps(params).encode('utf-8')
        stdout, stderr = process.communicate(input=input_data, timeout=600)
        
        if stderr:
            logger.info(f"[SUBPROCESS] Stderr: {stderr.decode('utf-8', errors='ignore')}")
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
            logger.error(f"[SUBPROCESS] Process failed: {error_msg}")
            update_task_status(task_id, "error", {"message": f"Subprocess error: {error_msg}"})
            return
        
        # Parse result
        try:
            result = json.loads(stdout.decode('utf-8'))
            update_task_status(task_id, "completed", result)
            logger.info(f"[SUBPROCESS] Completed: {result.get('status')}")
        except json.JSONDecodeError as e:
            logger.error(f"[SUBPROCESS] JSON decode error: {e}")
            logger.error(f"[SUBPROCESS] Stdout: {stdout.decode('utf-8', errors='ignore')}")
            update_task_status(task_id, "error", {"message": f"Invalid response: {str(e)}"})
            
    except subprocess.TimeoutExpired:
        logger.error(f"[SUBPROCESS] Timeout for task {task_id}")
        process.kill()
        update_task_status(task_id, "error", {"message": "OCR process timeout (10 minutes)"})
    except Exception as e:
        logger.exception(f"[SUBPROCESS] Error: {e}")
        update_task_status(task_id, "error", {"message": str(e)})


async def sync_document(
    doc_id: str,
    file_url: str,
    opd_name: Optional[str] = None
) -> Dict[str, Any]:
    """Sync document — trigger OCR worker as subprocess."""
    logger.info(f"[API] ========== DOC-SYNC START ==========")
    logger.info(f"[API] doc_id={doc_id}")
    logger.info(f"[API] opd={opd_name}")
    logger.info(f"[API] file_url={file_url}")
    sys.stdout.flush()
    
    # Validate file extension
    file_ext = os.path.splitext(file_url)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return {
            "status": "error",
            "message": f"Tipe file tidak didukung. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }
    
    # Validate file exists
    file_path = file_url
    if file_path.startswith("file://"):
        file_path = file_path.replace("file://", "")
    
    if not file_path.startswith(("http://", "https://")) and not os.path.exists(file_path):
        return {
            "status": "error",
            "message": f"File tidak ditemukan: {file_path}"
        }
    
    file_size_mb = 0
    if os.path.exists(file_path):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"[API] File validated: {file_path} ({file_size_mb:.2f} MB)")
    
    # Generate task ID
    task_id = f"{doc_id}_{int(time.time())}"
    update_task_status(task_id, "queued")
    
    # Prepare params for subprocess
    params = {
        "doc_id": doc_id,
        "opd_name": opd_name,
        "file_url": file_path,
        "lang": "id",
        "collection_name": config.COLLECTION_DOCUMENT
    }
    
    # Start subprocess in background thread
    thread = threading.Thread(
        target=run_ocr_subprocess,
        args=(task_id, params),
        daemon=True
    )
    thread.start()
    
    logger.info(f"[API] Task queued: {task_id}")
    
    return {
        "status": "queued",
        "task_id": task_id,
        "message": "Dokumen sedang diproses. Gunakan GET /api/doc-sync/status/{task_id} untuk cek status.",
        "file_size_mb": round(file_size_mb, 2)
    }
