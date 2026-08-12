#!/usr/bin/env python3
"""
RAG Medan v4 - Bulk Import Script: Qdrant → LightRAG Server

Mengambil data dari Qdrant collections yang sudah ada,
lalu mengirim teks utuh ke LightRAG Server untuk indexing.

LightRAG akan:
  1. Chunking ulang dengan caranya sendiri
  2. Ekstrak entities & relationships → Neo4j
  3. Simpan embeddings → Qdrant (collection lightrag_vdb_*)

Usage:
  python scripts/bulk_import_lightrag.py --source web
  python scripts/bulk_import_lightrag.py --source document
  python scripts/bulk_import_lightrag.py --source text
  python scripts/bulk_import_lightrag.py --source all

Options:
  --source     Sumber data: web | document | text | all (default: web)
  --batch-size Jumlah dokumen per batch (default: 5)
  --dry-run    Hanya tampilkan ringkasan, jangan kirim ke LightRAG
  --qdrant-url URL Qdrant (default: http://localhost:6333)
  --lightrag-url URL LightRAG Server (default: http://127.0.0.1:9621)
  --api-key    LightRAG API key (default: dari env LIGHTRAG_API_KEY)
"""
import argparse
import asyncio
import hashlib
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import httpx

# ============== CONFIGURATION ==============

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_LIGHTRAG_URL = "http://127.0.0.1:9621"
DEFAULT_BATCH_SIZE = 5
INSERT_TIMEOUT = 300.0  # 5 menit per dokumen (LLM extraction bisa lama)
DELAY_BETWEEN_REQUESTS = 2.0  # detik, hindari overload LLM


# ============== QDRANT READER ==============

class QdrantReader:
    """Baca semua points dari Qdrant collection via HTTP."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def scroll_all(
        self,
        collection_name: str,
        scroll_limit: int = 256,
    ) -> List[Dict[str, Any]]:
        """Scroll seluruh points dari collection, return list payload."""
        all_points = []
        offset = None

        while True:
            body: Dict[str, Any] = {
                "limit": scroll_limit,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset

            response = await self.client.post(
                f"{self.base_url}/collections/{collection_name}/points/scroll",
                json=body,
            )
            response.raise_for_status()
            data = response.json().get("result", {})

            points = data.get("points", [])
            all_points.extend(points)

            next_offset = data.get("next_page_offset")
            if not next_offset or not points:
                break
            offset = next_offset

        return all_points


# ============== CONTENT BUILDERS ==============

def build_web_documents(
    points: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Group web chunks per web_bank_id, urutkan by chunk_index,
    gabung jadi satu teks utuh per halaman web.

    Returns:
        List of {"text": ..., "description": ...}
    """
    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for point in points:
        payload = point.get("payload", {})

        # Skip inactive atau deleted
        if payload.get("is_deleted", False):
            continue
        if not payload.get("is_active", True):
            continue

        web_bank_id = payload.get("web_bank_id") or payload.get("link_id")
        if not web_bank_id:
            continue

        grouped[web_bank_id].append(payload)

    documents = []
    for web_bank_id, chunks in grouped.items():
        # Urutkan by chunk_index
        chunks.sort(key=lambda c: c.get("chunk_index", 0))

        # Ambil metadata dari chunk pertama
        first_chunk = chunks[0]
        title = first_chunk.get("title", "") or first_chunk.get("name", "")
        url = first_chunk.get("url", "")

        # Gabung konten
        content_parts = []
        for chunk in chunks:
            text = chunk.get("content", "").strip()
            if text:
                content_parts.append(text)

        full_text = "\n\n".join(content_parts).strip()
        if not full_text or len(full_text) < 50:
            continue  # Skip halaman yang terlalu pendek

        # Bangun teks dengan header metadata
        document_text = f"Judul: {title}\nURL: {url}\n\n{full_text}"
        description = f"web:{web_bank_id}"

        documents.append({
            "text": document_text,
            "description": description,
        })

    return documents


def build_document_documents(
    points: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Group document chunks per mysql_id, urutkan by chunk_index,
    gabung jadi satu teks utuh per dokumen.
    Filter chunks yang terlalu pendek (heading-only).

    Returns:
        List of {"text": ..., "description": ...}
    """
    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for point in points:
        payload = point.get("payload", {})

        if payload.get("is_deleted", False):
            continue
        if not payload.get("is_active", True):
            continue

        mysql_id = payload.get("mysql_id")
        if not mysql_id:
            continue

        grouped[mysql_id].append(payload)

    documents = []
    for mysql_id, chunks in grouped.items():
        chunks.sort(key=lambda c: c.get("chunk_index", 0))

        first_chunk = chunks[0]
        filename = first_chunk.get("filename", "")
        organization_id = first_chunk.get("organization_id") or first_chunk.get("opd", "")

        # Gabung teks, skip chunks terlalu pendek (heading-only < 20 char)
        content_parts = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text and len(text) >= 20:
                content_parts.append(text)

        full_text = "\n\n".join(content_parts).strip()
        if not full_text or len(full_text) < 100:
            continue  # Skip dokumen terlalu pendek

        document_text = (
            f"Dokumen: {filename}\n"
            f"OPD: {organization_id}\n\n"
            f"{full_text}"
        )
        description = f"document:{mysql_id}"

        documents.append({
            "text": document_text,
            "description": description,
        })

    return documents


def build_text_documents(
    points: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Build dokumen dari knowledge_bank (FAQ).
    Catatan: knowledge_bank hanya berisi pertanyaan, tidak ada jawaban.
    Fungsi ini disediakan untuk kelengkapan, tapi hasilnya terbatas.

    Returns:
        List of {"text": ..., "description": ...}
    """
    documents = []
    for point in points:
        payload = point.get("payload", {})
        question = payload.get("question", "") or payload.get("question_rag_name", "")
        if not question:
            continue

        question_id = payload.get("question_id", "")
        category_id = payload.get("category_id", "")

        document_text = f"Pertanyaan: {question}"
        description = f"text:{question_id}"

        documents.append({
            "text": document_text,
            "description": description,
        })

    return documents


# ============== LIGHTRAG INSERTER ==============

class LightRAGInserter:
    """Kirim dokumen ke LightRAG Server untuk indexing."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=INSERT_TIMEOUT,
        )
        self.total_success = 0
        self.total_failed = 0
        self.errors: List[Dict[str, str]] = []

    async def close(self):
        await self.client.aclose()

    async def insert_one(
        self,
        text: str,
        description: str,
    ) -> bool:
        """Insert satu dokumen ke LightRAG. Return True jika berhasil."""
        try:
            response = await self.client.post(
                f"{self.base_url}/documents/text",
                json={"text": text, "description": description},
            )
            response.raise_for_status()
            self.total_success += 1
            return True

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = (exc.response.text or "")[:200]
            print(f"    ✗ HTTP {status}: {body}")
            self.total_failed += 1
            self.errors.append({
                "description": description,
                "error": f"HTTP {status}: {body}",
            })
            return False

        except Exception as exc:
            print(f"    ✗ {type(exc).__name__}: {exc}")
            self.total_failed += 1
            self.errors.append({
                "description": description,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return False


# ============== MAIN ORCHESTRATOR ==============

async def run_import(
    source: str,
    qdrant_url: str,
    lightrag_url: str,
    api_key: str,
    batch_size: int,
    dry_run: bool,
):
    """Jalankan bulk import dari Qdrant ke LightRAG."""

    # ── Tentukan collection dan builder ──
    source_config = {
        "web": {
            "collection": "web_scraping_bank",
            "builder": build_web_documents,
            "label": "Web Scraping",
        },
        "document": {
            "collection": "document_bank",
            "builder": build_document_documents,
            "label": "Dokumen",
        },
        "text": {
            "collection": "knowledge_bank",
            "builder": build_text_documents,
            "label": "Text/FAQ",
        },
    }

    if source not in source_config:
        print(f"Error: source '{source}' tidak dikenali. Pilih: web | document | text | all")
        return

    cfg = source_config[source]
    print(f"\n{'='*60}")
    print(f"  BULK IMPORT: {cfg['label']} → LightRAG")
    print(f"  Qdrant   : {qdrant_url} ({cfg['collection']})")
    print(f"  LightRAG : {lightrag_url}")
    print(f"  Mode     : {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    # ── Step 1: Baca dari Qdrant ──
    print(f"[1/3] Membaca points dari {cfg['collection']}...")
    reader = QdrantReader(qdrant_url)
    try:
        raw_points = await reader.scroll_all(cfg["collection"])
    finally:
        await reader.close()

    print(f"    Total points mentah: {len(raw_points)}")

    # ── Step 2: Build dokumen ──
    print(f"[2/3] Membangun dokumen dari chunks...")
    documents = cfg["builder"](raw_points)
    print(f"    Total dokumen siap kirim: {len(documents)}")

    if not documents:
        print("\n    Tidak ada dokumen untuk di-import. Selesai.")
        return

    # Tampilkan ringkasan ukuran
    total_chars = sum(len(d["text"]) for d in documents)
    avg_chars = total_chars // len(documents)
    print(f"    Total karakter: {total_chars:,}")
    print(f"    Rata-rata per dokumen: {avg_chars:,} karakter")

    if dry_run:
        print(f"\n[DRY RUN] Selesai. Tidak ada data yang dikirim.")
        for i, doc in enumerate(documents[:5], 1):
            preview = doc["text"][:100].replace("\n", " ")
            print(f"    [{i}] {doc['description']} | {preview}...")
        if len(documents) > 5:
            print(f"    ... dan {len(documents) - 5} dokumen lainnya")
        return

    # ── Step 3: Kirim ke LightRAG ──
    print(f"[3/3] Mengirim {len(documents)} dokumen ke LightRAG...")
    print(f"    (timeout={INSERT_TIMEOUT}s, delay={DELAY_BETWEEN_REQUESTS}s)\n")

    inserter = LightRAGInserter(lightrag_url, api_key)
    start_time = time.time()

    try:
        for i, doc in enumerate(documents, 1):
            preview = doc["text"][:80].replace("\n", " ")
            print(f"  [{i}/{len(documents)}] {doc['description']} | {preview}...")

            success = await inserter.insert_one(doc["text"], doc["description"])
            status_icon = "✓" if success else "✗"
            print(f"    {status_icon} {'Berhasil' if success else 'Gagal'}")

            # Delay antar request
            if i < len(documents):
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    finally:
        await inserter.close()

    elapsed = time.time() - start_time

    # ── Ringkasan ──
    print(f"\n{'='*60}")
    print(f"  HASIL IMPORT: {cfg['label']}")
    print(f"{'='*60}")
    print(f"  Total dokumen  : {len(documents)}")
    print(f"  Berhasil       : {inserter.total_success}")
    print(f"  Gagal          : {inserter.total_failed}")
    print(f"  Durasi         : {elapsed:.1f}s")
    print(f"{'='*60}")

    if inserter.errors:
        print(f"\n  Error detail:")
        for err in inserter.errors[:10]:
            print(f"    - {err['description']}: {err['error'][:100]}")
        if len(inserter.errors) > 10:
            print(f"    ... dan {len(inserter.errors) - 10} error lainnya")


async def main():
    parser = argparse.ArgumentParser(
        description="Bulk import data dari Qdrant ke LightRAG Server"
    )
    parser.add_argument(
        "--source",
        default="web",
        help="Sumber data: web | document | text | all (default: web)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Jumlah dokumen per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya tampilkan ringkasan, jangan kirim ke LightRAG",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"URL Qdrant (default: {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--lightrag-url",
        default=DEFAULT_LIGHTRAG_URL,
        help=f"URL LightRAG Server (default: {DEFAULT_LIGHTRAG_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LIGHTRAG_API_KEY", ""),
        help="LightRAG API key (default: dari env LIGHTRAG_API_KEY)",
    )

    args = parser.parse_args()

    if args.source == "all":
        for source_name in ["web", "document", "text"]:
            await run_import(
                source=source_name,
                qdrant_url=args.qdrant_url,
                lightrag_url=args.lightrag_url,
                api_key=args.api_key,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            print()
    else:
        await run_import(
            source=args.source,
            qdrant_url=args.qdrant_url,
            lightrag_url=args.lightrag_url,
            api_key=args.api_key,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    asyncio.run(main())
