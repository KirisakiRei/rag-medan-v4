# RAG Medan v3

Sistem **Retrieval-Augmented Generation** (RAG) untuk knowledge base internal Pemda Medan,
didesain untuk menjawab pertanyaan warga/staf lewat WhatsApp dengan menggabungkan **4 sumber
pengetahuan** (text Q&A, dokumen PDF/Office, konten web scraping, dan usulan warga) ke dalam
satu API unified.

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur](#arsitektur)
- [Komponen Utama](#komponen-utama)
- [Penyimpanan Data](#penyimpanan-data)
- [Flow Utama](#flow-utama)
  - [1. Search Pipeline (Unified)](#1-search-pipeline-unified)
  - [2. Document Ingestion (OCR)](#2-document-ingestion-ocr)
  - [3. Web Scraping Pipeline](#3-web-scraping-pipeline)
  - [4. Text & Usulan Sync](#4-text--usulan-sync)
- [Keamanan](#keamanan)
- [Konfigurasi](#konfigurasi)
- [Deployment](#deployment)
- [Testing](#testing)

---

## Gambaran Umum

Sistem terdiri dari **orchestrator** (1 entry point untuk klien eksternal) dan **5 microservice**
internal yang masing-masing menangani sumber pengetahuan spesifik. Komunikasi antar service dilakukan
via HTTP dengan header `X-API-Key`.

Sumber pengetahuan:

| Sumber | Koleksi Qdrant | Cara Isi | Contoh Data |
|---|---|---|---|
| **RAG Text** | `knowledge_bank` | Sync manual (CRUD Q&A) | FAQ terstruktur |
| **RAG Document** | `document_bank` | Trigger OCR untuk PDF/DOC/XLSX/IMG | Buku, surat, laporan |
| **RAG Web** | `web_scraping_bank` | Scraping periodik URL OPD | Website pemda |
| **RAG Usulan** | `usulan_bank` | Sync usulan warga | Usulan Musrenbang |

Semua sumber di-embed ke vektor yang sama (dimensi 384 untuk small / 1024 untuk large) lalu
di-query paralel dari orchestrator — hasil terbaik dipilih oleh **AI relevance checker**.

---

## Arsitektur

```
                                ┌─────────────────────────┐
                                │   Klien (WA Gateway,    │
                                │   Dashboard, dll.)      │
                                └────────────┬────────────┘
                                             │ X-API-Key
                                             ▼
                    ┌────────────────────────────────────────────────┐
                    │          ORCHESTRATOR (port 5000)              │
                    │   /api/search  /api/doc-sync  /api/web-...     │
                    │                                                │
                    │   pre-filter → parallel fan-out → aggregate    │
                    │   → AI relevance check → answer extraction     │
                    └───────────┬────────────┬───────────┬───────────┘
                                │            │           │
        ┌───────────────────────┤            │           ├───────────────────────┐
        ▼                       ▼            ▼           ▼                       ▼
┌──────────────┐       ┌──────────────┐ ┌────────────┐ ┌──────────────┐   ┌─────────────┐
│  RAG Text    │       │ RAG Document │ │  RAG Web   │ │ RAG Usulan   │   │  Embedding  │
│  :5010       │       │   :5011      │ │  :5012     │ │  :5013       │   │  Service    │
│              │       │              │ │            │ │              │   │   :5014     │
│ - search     │       │ - search     │ │ - search   │ │ - search     │   │             │
│ - sync       │       │ - sync (OCR) │ │ - trigger  │ │ - sync       │   │ /embed      │
│              │       │ - delete     │ │ - update   │ │              │   │ (shared     │
│              │       │              │ │ - sync     │ │              │   │  small/large│
└──────┬───────┘       └──────┬───────┘ │ - delete   │ └──────┬───────┘   │  model)     │
       │                      │         └─────┬──────┘        │          └──────┬──────┘
       ▼                      ▼               ▼               ▼                 ▼
  ┌──────────────┐     ┌──────────────┐  ┌───────────┐  ┌──────────────┐   ┌──────────┐
  │ knowledge_   │     │ document_    │  │ web_      │  │ usulan_      │   │ MySQL    │
  │ bank         │     │ bank         │  │ scraping_ │  │ bank         │   │ (varia-  │
  │ (Qdrant)     │     │ (Qdrant)     │  │ bank      │  │ (Qdrant)     │   │  bles)   │
  └──────────────┘     │              │  │ + state   │  └──────────────┘   └──────────┘
                       │     + worker │  │ (Qdrant)  │
                       │   (subproc)  │  └───────────┘
                       └──────────────┘
```

Port default:
- `5000` — Orchestrator (entry point)
- `5010` — RAG Text
- `5011` — RAG Document
- `5012` — RAG Web
- `5013` — RAG Usulan
- `5014` — Embedding Service

---

## Komponen Utama

### Orchestrator (`orchestrator/`)

Entry point tunggal. Menerima request eksternal di `/api/*`, lalu melakukan:

1. **AI Pre-filter** — cek domain relevance via LLM (opsional, `use_ai_pre_filter`).
   Kalau out-of-domain, return cepat tanpa fan-out.
2. **Parallel Fan-out** — panggil `/internal/search-unified` ke 3 service (text, document, web)
   secara paralel dengan `asyncio.gather`. Ada **adaptive early exit**: kalau salah satu service
   sudah mengembalikan kandidat dengan `dense_score >= EARLY_EXIT_THRESHOLD` (0.92), service
   lain yang belum selesai di-cancel.
3. **Aggregate & Sort** — gabungkan kandidat dari semua service, rerank dengan kombinasi
   dense score, lexical overlap, dan category boost.
4. **AI Relevance Check** — LLM mengevaluasi top kandidat (mode `single` atau `batch`).
5. **Answer Extraction** — untuk sumber `document`/`web`, LLM mengekstrak jawaban spesifik
   dari chunk relevan.
6. **Validation** — jawabannya dicek (reject jika kosong/leakage prompt).
7. **Response** — build payload `success` atau `low_confidence` dengan timing breakdown.

Modul:
- `orchestrator.py` — routing & lifespan
- `search_handler.py` — pipeline search
- `service_client.py` — HTTP wrapper (call internal services, header X-API-Key)
- `aggregation.py` — reranking & merging kandidat
- `answer_validation.py` — validasi struktur jawaban AI
- `models.py` — Pydantic schemas

### RAG Text (`services/rag_text/`)

Knowledge base Q&A terstruktur. Data masuk lewat endpoint `/internal/sync` dengan aksi:
- `bulk_sync` — embed batch Q&A, upsert ke Qdrant + buat fulltext index di `question_rag_name`.
- `add` / `update` — embed 1 item, upsert.
- `delete` — hapus point by ID.

Search (`/internal/search-unified`): embed pertanyaan user (small model via shared embedding),
query cosine similarity di Qdrant, rerank dengan LLM, kembalikan top-k.

### RAG Document (`services/rag_document/`) + Worker

Menangani dokumen terlampir (PDF, DOCX, XLSX, TXT, gambar). Ingestion dilakukan via
subprocess `worker.py` untuk isolasi OCR yang berat.

Pipeline:
1. `sync` endpoint menerima metadata (doc_id, org, file_url, is_active).
2. Spawn subprocess worker → download file, deteksi duplikasi (file_hash + content_hash),
   ekstrak teks, chunk parent-child, embed, upsert.
3. Progress diemit lewat stderr JSON (dibaca parent, dikirim ke dashboard via webhook).

### RAG Web (`services/rag_web/`)

Scraping periodik website OPD (situs resmi). Alur:
1. `trigger`/`update` menerima metadata web bank (URL, CSS selector, interval).
2. **Scraper** — ambil raw HTML (httpx dulu, fallback ke Playwright jika konten < 300 char).
3. **Cleaner** — buang non-konten (nav, footer, ads), ekstrak main content.
4. **FAQ Extractor** — deteksi pola FAQ (accordion, dl, details) → struktur Q&A.
5. **Chunker** — parent-child chunking dengan semantic merge.
6. Embed + upsert ke `web_scraping_bank`.
7. Update state di `web_scraping_state` (last_content_hash, next_scrape_at).

Ada **deduplikasi konten**: kalau hash tidak berubah, scraping di-skip.

### RAG Usulan (`services/rag_usulan/`)

Sama seperti RAG Text, tapi untuk usulan warga (Musrenbang). Data disinkronkan dari MySQL
eksternal. Search & sync mirip text.

### Embedding Service (`services/embedding_service/`)

Service tunggal yang memuat **satu copy** model embedding dan melayani encoding untuk semua
service RAG via HTTP. Menghemat ~3-4 GB RAM (4 service × 1 model → 1 model).

- **Small model** (e5-small, dim=384) — pre-warmed di startup.
- **Large model** (e5-large, dim=1024) — lazy load saat dibutuhkan (document/web search),
  idle-unload setelah `LARGE_MODEL_IDLE_TIMEOUT` (default 1800s).
- Endpoint `POST /embed` menerima `{texts, prefix, model_size}` → return embeddings.
- Encoding dijalankan di `ThreadPoolExecutor` (CPU-bound).

---

## Penyimpanan Data

### Qdrant (Vector DB)

5 koleksi:

| Koleksi | Dimensi | Sumber |
|---|---|---|
| `knowledge_bank` | 384 | RAG Text |
| `document_bank` | 1024 | RAG Document (parent-child) |
| `web_scraping_bank` | 384 | RAG Web (parent-child) |
| `web_scraping_state` | 1 | Status scraping (hash, timestamp) |
| `usulan_bank` | 384 | RAG Usulan |

Payload fields (document/web): `mysql_id`, `chunk_id`, `chunk_level`, `parent_chunk_id`,
`heading_path`, `page_start`, `page_end`, `file_hash`, `content_hash`, `is_active`,
`is_deleted`, `source_kind`.

### MySQL (Variables table)

Tabel `variables` — menyimpan konfigurasi dinamis yang bisa di-override tanpa restart
(prompt LLM, relevance_mode, dll). Dibaca lewat `shared/db.py::get_variable()`.

Contoh:
- `RAG_SYSTEM_PROMPT`, `RAG_DOMAIN_CONTEXT`
- `PROMPT_RELEVANCE_SINGLE`, `PROMPT_RELEVANCE_BATCH`
- `RELEVANCE_MODE` (single / batch)

---

## Flow Utama

### 1. Search Pipeline (Unified)

```
POST /api/search
  │
  ├── [1] AI Pre-filter (opsional)
  │       └── LLM cek: pertanyaan relevan dengan domain?
  │           ├── No → return "out of domain" cepat
  │           └── Yes → lanjut + clean question
  │
  ├── [2] Normalize question
  │       └── clean_location_terms() → normalize_text() → detect_category()
  │
  ├── [3] Parallel Fan-out
  │       ├── rag-text  /internal/search-unified
  │       ├── rag-document /internal/search-unified
  │       └── rag-web   /internal/search-unified
  │
  │       Setiap service:
  │         1. embed pertanyaan (shared embedding)
  │         2. cosine search di Qdrant (top-k)
  │         3. rerank dengan LLM (relevance per kandidat)
  │         4. return {status, data.results[], data.count}
  │
  │       Adaptive early exit: kalau ada kandidat dense_score ≥ 0.92,
  │       service lain yang belum selesai di-cancel.
  │
  ├── [4] Aggregate & Sort
  │       └── merge kandidat semua service, compute final_score
  │           (dense × 0.6 + overlap × 0.3 + boost)
  │
  ├── [5] AI Relevance Check
  │       └── LLM cek top kandidat (batch / single mode)
  │           ├── relevant → pilih
  │           └── not relevant → return low_confidence
  │
  ├── [6] Answer Extraction (untuk document/web)
  │       └── LLM ekstrak jawaban dari chunk terpilih
  │       └── Validasi: kosong / leakage → reject & fallback
  │
  └── [7] Build Response
          ├── success: source, similar_questions, answer_doc, timing
          └── low_confidence: reason + top kandidat untuk debugging
```

### 2. Document Ingestion (OCR)

```
POST /api/doc-sync {doc_id, organization_id, file_url, is_active}
  │
  ▼
POST /internal/sync (rag-document)
  │
  └── spawn subprocess worker.py
        │
        ├── [a] Download file → document_temp/{YYYY-MM-DD}/
        │
        ├── [b] Cek duplikasi by file_hash
        │       ├── match aktif → skip
        │       └── match soft-deleted → reactivate
        │
        ├── [c] Ekstraksi teks
        │       ├── PDF  → extract_pdf_layout (PyMuPDF)
        │       │           └── tiap halaman: tabel + heading + paragraf
        │       │           └── OCR fallback (local / API) untuk halaman gambar
        │       ├── IMG  → PaddleOCR atau LLM OCR
        │       ├── XLSX → openpyxl (per sheet, per baris)
        │       └── TXT  → baca langsung
        │
        ├── [d] Cek duplikasi by content_hash (text)
        │
        ├── [e] Chunking parent-child
        │       ├── parent chunk: ~1100 chars (tidak di-embed)
        │       ├── child  chunk: ~380 chars (di-embed)
        │       ├── semantic merge: gabungkan paragraf mirip (sim > 0.32)
        │       └── heading_path: track konteks section
        │
        ├── [f] Embed via shared embedding service (model_size=large)
        │
        └── [g] Upsert ke Qdrant (parent + child points)
                └── hapus chunk lama untuk doc_id dulu
```

Progress diemit via stderr JSON:
```json
{"stage": "embedding", "message": "Embedding chunk 50/200...", "processed_chunks": 50, ...}
```

### 3. Web Scraping Pipeline

```
POST /api/web-trigger {web_bank_id, url, css_selector, scrape_interval}
  │
  ▼
POST /internal/trigger (rag-web)
  │
  └── reserve_job (cegah duplikat in-flight)
      └── background task: process_url
            │
            ├── [a] Scraping
            │       ├── httpx (default, fast)
            │       └── auto-detect: kalau konten < 300 char → retry dengan Playwright
            │
            ├── [b] Cleaner
            │       ├── buang script/style/nav/footer
            │       └── ekstrak main content + judul
            │
            ├── [c] FAQ Extractor (jika ada pola)
            │       ├── Bootstrap accordion
            │       ├── dl (definition list)
            │       └── details/summary
            │
            ├── [d] Chunker (parent-child)
            │       ├── heading-aware splitting
            │       └── semantic merge
            │
            ├── [e] Embed & upsert ke web_scraping_bank
            │
            ├── [f] Update web_scraping_state
            │       └── last_content_hash, scraped_at, next_scrape_at
            │
            └── [g] Webhook callback ke dashboard (opsional)
                    └── report status, chunk count
```

**Deduplikasi**: kalau `last_content_hash == new_content_hash`, scraping di-skip.
**Rate limiting**: jeda `RATE_LIMIT_DELAY` detik antar request ke domain sama.

### 4. Text & Usulan Sync

Sederhana — CRUD langsung di Qdrant.

```
POST /api/sync {action: "bulk_sync" | "add" | "update" | "delete", content}
  │
  ▼
POST /internal/sync (rag-text / rag-usulan)
  │
  ├── bulk_sync → embed batch + upsert semua
  ├── add/update → embed 1 item + upsert
  └── delete → hapus point
```

---

## Keamanan

Semua endpoint internal (`/internal/*`) dilindungi middleware **`InternalAuthMiddleware`**
(`shared/security.py`):

- Header `X-API-Key` wajib untuk semua request internal (orchestrator ↔ services).
- **Fail-closed**: kalau `INTERNAL_API_KEY` kosong di `.env`, semua request non-allowlist
  ditolak 401.
- Allowlist: `/health`, `/docs`, CORS preflight, `/` (root).
- Endpoint eksternal `/api/*` di orchestrator juga dilindungi dengan header yang sama.

Header dikirim otomatis oleh `service_client.py` (`orchestrator/`) saat memanggil service internal.

---

## Konfigurasi

Konfigurasi terpusat di `config.py`, dibaca dari `.env` via `python-dotenv`.

### Variabel `.env` penting

| Kelompok | Variabel | Fungsi |
|---|---|---|
| **Port** | `ORCHESTRATOR_PORT`, `TEXT_SERVICE_PORT`, `DOCUMENT_SERVICE_PORT`, `WEB_SERVICE_PORT`, `USULAN_SERVICE_PORT`, `EMBEDDING_SERVICE_PORT` | Port tiap service |
| **Model** | `EMB_MODEL_PATH`, `EMB_LARGE_PATH`, `EMBEDDING_DIMENSION`, `EMBEDDING_DIMENSION_LARGE` | Path & dimensi model embedding |
| **LLM** | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT_SEC`, `LLM_PROVIDER` | Gemini / Router API |
| **Qdrant** | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`, `COLLECTION_*` | Vector DB |
| **MySQL** | `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD` | Variabel dinamis |
| **OCR** | `OCR_MODE` (local/api), `ROUTER_API_URL`, `ROUTER_API_KEY`, `OCR_LLM_MODEL` | Mode OCR |
| **Security** | `INTERNAL_API_KEY` | X-API-Key antar service |
| **Tuning** | `MODEL_IDLE_TIMEOUT`, `LARGE_MODEL_IDLE_TIMEOUT`, `EARLY_EXIT_THRESHOLD` | Timeout & threshold |
| **Embedding** | `USE_SHARED_EMBEDDING`, `SHARED_EMBEDDING_URL` | Shared vs local |
| **Scraping** | `SCRAPING_TIMEOUT`, `AUTO_DETECT_MIN_CONTENT`, `PLAYWRIGHT_TIMEOUT`, `RATE_LIMIT_DELAY` | Web scraping |

### Override prompt dari MySQL

Prompt LLM (`PROMPT_*`, `RAG_SYSTEM_PROMPT`, dll) disimpan di tabel `variables`.
Bisa diubah tanpa restart — di-cache `PROMPT_CACHE_TTL` detik (default 300).

---

## Deployment

### PM2 (`ecosystem.config.js`)

Setiap service dijalankan sebagai 1 instance PM2:

```bash
pm2 start ecosystem.config.js      # semua service
pm2 status                          # cek status & memory
pm2 logs orchestrator               # log orchestrator
pm2 restart all                     # restart
```

Optimasi RAM:
- `PYTHONMALLOC=malloc` + `MALLOC_TRIM_THRESHOLD_=100000` di env tiap service.
- `USE_SHARED_EMBEDDING=true` → 4 service RAG tidak memuat model.
- Idle-unload otomatis setelah `MODEL_IDLE_TIMEOUT` (600s) / `LARGE_MODEL_IDLE_TIMEOUT` (1800s).
- `max_memory_restart` per service di PM2 (guard).

### Requirements

```bash
pip install -r requirements.txt
playwright install chromium   # untuk web scraping JS-heavy
```

### Prasyarat

- Python 3.10+
- Qdrant server (lokal atau cloud)
- MySQL server (tabel `variables` + tabel terkait di dashboard)
- (Opsional) Playwright browser untuk JS scraping
- Model embedding lokal di path yang di-set di `EMB_MODEL_PATH` / `EMB_LARGE_PATH`

---

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Test mencakup:
- `test_internal_auth.py` — middleware X-API-Key (fail-closed, allowlist).
- `test_lazy_model_gate.py` — gate USE_SHARED_EMBEDDING di LazyModel.
- `test_batch_relevance_filtering.py` — parser batch AI.
- `test_document_answer_*.py` — pipeline answer extraction & validation.
- `test_document_context_expansion.py` — parent-child context expansion.
- `test_document_parent_child_context.py` — dedup by parent before ranking.
- `test_pdf_layout_extractor.py` — layout-aware PDF extraction.
- `test_ocr_utils_pdf_layout_integration.py` — integration OCR + layout.
- `test_document_worker_pdf_routing.py` — PDF routing di worker.

Verifikasi compile:

```bash
python -m py_compile shared/bootstrap.py shared/db.py \
  services/rag_text/main.py services/rag_document/main.py \
  services/rag_web/main.py services/rag_usulan/main.py \
  services/embedding_service/main.py orchestrator/orchestrator.py
```

---

## Ringkasan Alur

```
User bertanya → Orchestrator → Pre-filter LLM
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
            RAG Text          RAG Document       RAG Web
            (Q&A)             (PDF/DOC/XLSX)     (Web scraping)
                  │                 │                 │
                  └────────────┬────┴─────────────────┘
                               ▼
                       Aggregate + Sort
                               │
                               ▼
                       AI Relevance Check
                               │
                               ▼
                      Answer Extraction
                               │
                               ▼
                     JSON Response ke user
```

Setiap langkah punya **timing breakdown** (`timing.ai_domain_sec`, `ai_relevance_sec`,
`parallel_search_sec`, `total_sec`) untuk observability.
