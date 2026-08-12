# CHANGELOG

Format perubahan mengikuti urutan kronologis. `[Security]` menandai
perubahan yang berdampak pada keamanan dan memerlukan tindakan saat
deployment.

---

## [2026-08-09] - Optimasi P1 & P2 (RAM, Koneksi DB, Health, Dependensi)

### P1: Gate Model Embedding Lokal (Hemat RAM Besar)

- **Baru: `shared/bootstrap.py`** — `LazyModel` holder model embedding:
  - Gate `USE_SHARED_EMBEDDING=True` (default): `get()` mengembalikan
    `None` dan model lokal **tidak pernah dimuat**. Sebelumnya keempat
    service tetap memuat SentenceTransformer (~500MB-1GB) padahal encoding
    sudah didelegasikan ke embedding microservice.
  - Mode `USE_SHARED_EMBEDDING=False`: lazy load + thundering herd
    protection + idle unload (perilaku lama dipertahankan).
  - `on_load` callback untuk wiring instance ke search/sync module.
- **`services/rag_{text,usulan,document,web}/main.py`** — pindah ke
  `LazyModel`, hapus duplikasi `get_model`/`_idle_unload_loop`/`init_qdrant`.

### P1: MySQL Connection Pooling

- **`shared/db.py`** — ganti koneksi baru per-call dengan
  `MySQLConnectionPool` (size 5, `pool_reset_session=False`):
  - `ping(reconnect=True)` saat checkout mencegah "server has gone away"
    pada koneksi reuse.
  - Fallback koneksi langsung jika pool habis/error (tidak ada request
    yang gagal hanya karena pool exhaustion).

### P1: Bootstrap Refactor (Deduplikasi)

- `create_qdrant_client()`, `ensure_payload_index()`, `backfill_is_active()`
  dipindah ke `shared/bootstrap.py` (sebelumnya duplikat identik di
  rag_document & rag_web).

### P2: Health Endpoint Gabungan (Orchestrator)

- **`orchestrator/orchestrator.py`** `/health`:
  - Cek **paralel** via `asyncio.gather` (sebelumnya 4x sequential, 10s
    timeout tiap service → lambat).
  - Tambah komponen `embedding_service`.
  - Threshold: core (text/document/web/usulan) wajib healthy; embedding
    service hanya wajib saat `USE_SHARED_EMBEDDING=True`.
  - Status: `healthy` / `degraded` / `unhealthy` + detail per komponen.

### P2: Rapi Requirements.txt

- **Hapus (tak terpakai di kode aktif):** aiohttp, langchain,
  langchain-text-splitters, nltk, langdetect, Sastrawi, rapidfuzz,
  trafilatura, PyMySQL, lxml, tqdm, click, python-multipart, pdf2image.
- **Tambah (dipakai tapi belum terdaftar):** `PyMuPDF` (fitz, dipakai
  `pdf_layout_extractor`/`ocr_utils`), `openpyxl` (dipakai ekstraksi .xlsx).
- Versi dilonggarkan ke `>=` (floor = versi teruji); paddlepaddle/paddleocr
  tetap di-pin karena kombinasi versi sudah teruji.

### P2: Deduplikasi Scraper/Init

- Duplikasi `init_qdrant` + payload index + backfill antara rag_document
  dan rag_web disatukan lewat `shared/bootstrap.py`.

### Verifikasi

- `py_compile` semua file berubah — OK.
- Test suite: **49 passed** (termasuk `test_lazy_model_gate` baru untuk
  gate shared embedding, dan `test_internal_auth`).
- `import asyncio`/`gc`/`time` yang tak terpakai dibersihkan dari 4 service.

---

## [2026-08-09] - Config: Ramping `.env` (Hapus Key Tuning Duplikat)

### Ringkasan

Sebanyak 16 key tuning yang sebelumnya ditambahkan ke `.env` ternyata
memiliki nilai default identik di `config.py` — dihapus agar `.env`
hanya menyimpan konfigurasi spesifik environment. Tidak ada perubahan
perilaku (nilai fallback sama persis).

### Key yang Dihapus dari `.env`

`RELEVANCE_MODE`, `ENABLE_CITATION`, `OCR_STALL_TIMEOUT`,
`OCR_HARD_TIMEOUT`, `OCR_QUEUE_LOG_INTERVAL`, `OCR_PDF_DPI`,
`OCR_PDF_DPI_RETRY`, `OCR_DOWNLOAD_PROGRESS_MB`, `DOC_CHILD_CHUNK_SIZE`,
`DOC_PARENT_CHUNK_SIZE`, `DOC_CHUNK_OVERLAP`, `WEB_CHILD_CHUNK_SIZE`,
`WEB_PARENT_CHUNK_SIZE`, `ENABLE_SEMANTIC_MERGE`,
`SEMANTIC_MERGE_SIM_THRESHOLD`, `RETRIEVAL_CONTEXT_EXPANSION`.

Catatan: `RELEVANCE_MODE` dan `ENABLE_CITATION` tetap berfungsi — default
`config.py` dipakai saat DB variable (`relevance_mode`, `enable_citation`)
kosong.

### Key Baru yang Dipertahankan di `.env`

| Key | Alasan |
|---|---|
| `COLLECTION_WEB_STATE` | Nama koleksi, spesifik environment |
| `OCR_MODE` | Mode OCR per mesin (local/api) |
| `ROUTER_API_URL`, `ROUTER_API_KEY` | Kredensial, hanya dipakai saat `OCR_MODE=api` |
| `OCR_LLM_MODEL`, `OCR_LLM_MAX_TOKENS` | Konfigurasi model OCR LLM (mode api) |
| `OCR_DELAY`, `OCR_RETRIES` | Perilaku request LLM OCR (mode api) |

### Verifikasi

- `.env` tersisa 65 key, semuanya terpakai di kode.
- Nilai fallback `config.py` identik dengan nilai yang dihapus.

---

## [2026-08-09] - Config: Pembersihan Variabel Mati

### Ringkasan

Audit seluruh key konfigurasi terhadap pemakaian di kode aktif
(orchestrator, services, shared, tests). Hasilnya, 13 atribut di
`config.py` dan 14 entri di `.env` tidak pernah dipakai — dihapus agar
konfigurasi lebih ramping dan tidak menyesatkan.

### Atribut `config.py` yang Dihapus (tidak dipakai di kode mana pun)

- `DEBUG` (tidak pernah dibaca)
- `OCR_ENGINE`, `OCR_LANG` (engine selalu PaddleOCR, lang selalu "id")
- `TEXT_DENSE_THRESHOLD`, `TEXT_OVERLAP_THRESHOLD`
  (nilai ambang di-hardcode di `services/rag_text/search.py`)
- `DOCUMENT_SCORE_THRESHOLD`, `WEB_SCORE_THRESHOLD`,
  `USULAN_SCORE_THRESHOLD` (di-hardcode di search masing-masing service)
- `RERANK_TOP_K`, `RERANK_WEIGHT_TEXT`, `RERANK_WEIGHT_DOC`,
  `RERANK_WEIGHT_WEB` (fungsi `ai_rerank_results` memang tidak dipakai)

### Entri `.env` yang Dihapus

- 13 key di atas
- `CHUNK_SIZE`, `CHUNK_OVERLAP` (legacy, tidak ada di `config.py`)

### File Diubah

| File | Perubahan |
|---|---|
| `config.py` | Hapus 13 atribut mati |
| `.env` | Hapus 14 entri mati (sisa 81 key, semua terpakai) |
| `.env.example` | Hapus key mati yang sama agar template konsisten |

### Prinsip yang Diterapkan

- `.env` berisi hal yang berbeda antar environment: port, URL, path model,
  kredensial, key, mode OCR, CORS, dll.
- `config.py` menyimpan nilai default tuning (timeout, ukuran chunk,
  threshold, delay) — cukup diatur di kode, tidak wajib di `.env`.
- Key yang terpakai lewat alias `app_config.` (mis. `RATE_LIMIT_DELAY`,
  `WEBHOOK_RETRY_ATTEMPTS`, `WEB_CALLBACK_URL`) dipertahankan.

### Verifikasi

- `python -m py_compile config.py` — lolos.
- Audit ulang pemakaian: tidak ada atribut mati tersisa di `config.py`.
- `.env` tidak mengandung key mati (cek otomatis).

---

## [2026-08-09] - Security: Autentikasi Internal X-API-Key

### Ringkasan

Menambahkan autentikasi berbasis header `X-API-Key` untuk seluruh
orchestrator dan service internal. Struktur payload API **tidak berubah**
sama sekali; autentikasi murni berjalan di level HTTP header.

### Keputusan Desain

- Satu key bersama `INTERNAL_API_KEY` untuk semua service (mudah rotasi).
- Proteksi diterapkan di orchestrator (port 5000, `/api/*`) dan seluruh
  service internal (5010-5014, `/internal/*` dan `/embed`).
- Fail-closed: jika `INTERNAL_API_KEY` kosong, semua request non-allowlist
  ditolak HTTP 401. Sistem tidak berjalan "terbuka" secara diam-diam.
- Allowlist tetap terbuka: `/`, `/health`, `/docs`, `/redoc`,
  `/openapi.json`, dan semua metode `OPTIONS` (CORS preflight).
- Perbandingan key memakai `secrets.compare_digest` (constant-time).
- Key dibaca per-request dari `config.INTERNAL_API_KEY`, sehingga rotasi
  cukup mengubah `.env` + restart PM2 (tanpa ubah kode).
- Nilai key tidak pernah ditulis ke log; 401 dicatat dengan client IP.

### File Baru

| File | Isi |
|---|---|
| `shared/security.py` | `InternalAuthMiddleware` (middleware auth) |
| `tests/test_internal_auth.py` | Unit test middleware (fail-closed, key benar/salah, allowlist, OPTIONS) |
| `CHANGELOG.md` | File ini |

### File Diubah

| File | Perubahan |
|---|---|
| `config.py` | Tambah `INTERNAL_API_KEY = _env("INTERNAL_API_KEY", "")` |
| `.env.example` | Tambah blok konfigurasi `INTERNAL_API_KEY` (kini ter-versioning, dihapus dari `.gitignore`) |
| `orchestrator/orchestrator.py` | Pasang `InternalAuthMiddleware` (setelah CORS) |
| `services/rag_text/main.py` | Pasang middleware |
| `services/rag_document/main.py` | Pasang middleware |
| `services/rag_web/main.py` | Pasang middleware |
| `services/rag_usulan/main.py` | Pasang middleware |
| `services/embedding_service/main.py` | Pasang middleware |
| `orchestrator/service_client.py` | Semua panggilan keluar ke service internal membawa header `X-API-Key` (POST/PUT/GET/DELETE) |
| `shared/utils.py` | Client embedding (`encode_texts`) membawa header `X-API-Key` |
| `services/rag_document/worker.py` | Request embedding dari subprocess OCR membawa header `X-API-Key` |
| `.gitignore` | Hapus `tests/` dan `.env.example` dari daftar ignore agar ter-versioning |

### Alur Request Setelah Perubahan

```
WA Manajemen (header: X-API-Key)
   └─► Orchestrator 5000  [auth] /api/*
          └─► Text 5010 / Document 5011 / Web 5012 / Usulan 5013  [auth] /internal/*
                └─► Embedding 5014  [auth] /embed
```

Header outbound ditambahkan otomatis oleh kode, payload JSON tidak berubah.
Respons 401 dari service internal tetap terpropagasi sebagai
`{"status": "error", "message": "Unauthorized"}`.

### Yang TIDAK Diubah

- Semua `models.py` (request/response) — struktur payload mutlak sama.
- Semua endpoint, response builder, logika pencarian, chunking, OCR.
- Callback keluar ke WA manajemen (`X-API-Key`/`X-API-KEY` yang sudah ada).
- `ecosystem.config.js` — sengaja TIDAK menambahkan key di blok `env`,
  karena PM2 env akan menimpa nilai dari `.env`; `config.py` sudah
  memanggil `load_dotenv()`.

### Langkah Deployment (Wajib)

1. Set nilai rahasia kuat pada `INTERNAL_API_KEY` di `.env` server.
2. Restart semua service: `pm2 restart all`.
3. Koordinasi ke tim sistem eksternal (WA manajemen): semua request ke
   orchestrator `/api/*` wajib membawa header `X-API-Key`; payload tidak
   berubah.
4. Rotasi key: ubah nilai `.env`, `pm2 restart all`.

### Verifikasi yang Dilakukan

- `python -m py_compile` pada semua file yang diubah — lolos.
- Grep propagasi `X-API-Key`: seluruh jalur outbound ke service internal
  tercakup (orchestrator, client embedding, worker OCR).
- Review diff: tidak ada perubahan struktur payload.
- Sistem tidak dijalankan (sesuai kesepakatan, dijalankan di server).

### Menjalankan Unit Test (di server)

```bash
pip install pytest
python -m pytest tests/test_internal_auth.py -v
```

`pytest` tidak ditambahkan ke `requirements.txt` karena khusus test.
