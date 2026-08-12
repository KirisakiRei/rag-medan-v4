# Plan Implementasi — LightRAG Engine Integration (RAG Medan v4)

**Document:** `plan-implement-v4.md`  
**Created:** 2025-07-14  
**Status:** Active Implementation Plan  
**Scope:** Phase 0 (Foundation) + Phase 1 (Adapter + Search Provider)

---

## 1. Tujuan

Mengimplementasikan LightRAG sebagai unified retrieval engine untuk RAG Medan v4
tanpa mengubah API eksternal. Semua perubahan bersifat internal.

**Prinsip:**
- Tidak ada perubahan pada endpoint eksternal (`/api/*`)
- Tidak ada perubahan pada request/response contract
- Tidak ada perubahan pada authentication (`X-API-Key`)
- SQL tetap sebagai source of truth
- Legacy search tetap berjalan selama migrasi
- Rollback instan via environment variable

---

## 2. Keputusan Desain Final

| Aspek | Keputusan |
|---|---|
| Vector Storage | Qdrant (reuse existing, collection baru) |
| KV + Graph + DocStatus | PostgreSQL (install baru) |
| Embedding Model | E5-small 384 dim (tetap) |
| LLM | Gemini 2.0 Flash (reuse existing) |
| LightRAG Server | Clone dari GitHub, port 9621 |
| LightRAG Adapter | Service baru, port 5015 |
| Query Mode | `mix` (default) |
| Auth | `X-API-Key` (sama dengan existing) |
| Workspace | `medan-main` |
| Summary Language | Indonesian |
| Entity Extraction | JSON format enabled |

---

## 3. Scope Implementasi

### 3.1 Termasuk (In Scope)

1. Clone LightRAG repository ke dalam project
2. Buat service `lightrag_adapter` (port 5015)
3. Buat `SearchProvider` abstraction di Orchestrator
4. Update `config.py` dengan variable baru
5. Update `ecosystem.config.js` dengan entry baru
6. Update `.env.example` dengan section baru

### 3.2 Tidak Termasuk (Out of Scope)

- Modifikasi pipeline ingestion existing (dilakukan di Phase 2)
- Modifikasi search_handler.py secara langsung (dilakukan saat aktivasi lightrag)
- Setup PostgreSQL server (dilakukan di VPS terpisah)
- Setup LightRAG Server runtime config (dilakukan di VPS)
- Benchmark dan shadow mode (Phase 1-2)
- Reindex pipeline (Phase 2+)
- Unit tests (akan ditambahkan setelah core logic stabil)

---

## 4. Struktur File Baru

```
rag-medan-v4/
├── lightrag/                          # [BARU] Clone dari GitHub HKUDS/LightRAG
│   ├── lightrag/                      # LightRAG Python package
│   ├── docs/
│   ├── examples/
│   ├── env.example                    # Template config LightRAG Server
│   ├── docker-compose.yml
│   └── ...
│
├── services/
│   └── lightrag_adapter/              # [BARU] Adapter service
│       ├── __init__.py
│       ├── main.py                    # FastAPI app + endpoints
│       ├── config.py                  # Adapter-specific config
│       ├── client.py                  # LightRAG HTTP client + circuit breaker
│       ├── models.py                  # Pydantic models
│       ├── search.py                  # Search logic
│       ├── sync.py                    # Sync logic (text/doc/web)
│       ├── source_mapper.py           # Source ID + URI mapping
│       ├── references.py             # Citation mapping
│       ├── fallback.py               # Legacy fallback routing
│       ├── errors.py                 # Custom exceptions
│       └── health.py                 # Health check logic
│
├── orchestrator/
│   └── search_provider.py            # [BARU] SearchProvider abstraction
│
├── config.py                         # [UPDATE] Tambah LIGHTRAG_* vars
├── ecosystem.config.js               # [UPDATE] Tambah lightrag-adapter entry
└── .env.example                      # [UPDATE] Tambah section LightRAG
```

---

## 5. Urutan Implementasi

### Step 1: Clone LightRAG Repository
- Clone `https://github.com/HKUDS/LightRAG.git` ke `lightrag/`
- Hanya source code, tidak jalankan server

### Step 2: Buat Foundation Files (lightrag_adapter)
- `__init__.py` — Package marker
- `config.py` — Adapter configuration
- `errors.py` — Custom exception classes
- `models.py` — Pydantic request/response models

### Step 3: Buat Core Logic Files
- `source_mapper.py` — Deterministic source ID + URI generation
- `references.py` — Citation mapping dari LightRAG ke canonical format
- `client.py` — HTTP client dengan circuit breaker + retry

### Step 4: Buat Business Logic Files
- `search.py` — Unified search via LightRAG
- `sync.py` — Sync handlers untuk text/document/web
- `fallback.py` — Legacy fallback routing
- `health.py` — Health check

### Step 5: Buat FastAPI Application
- `main.py` — FastAPI app, lifespan, endpoints

### Step 6: Update Existing Files
- `config.py` — Tambah LIGHTRAG_* configuration
- `orchestrator/search_provider.py` — SearchProvider abstraction (baru)
- `ecosystem.config.js` — Tambah lightrag-adapter PM2 entry
- `.env.example` — Tambah LightRAG section
- `services/__init__.py` — Update services list

---

## 6. Dependencies

### Python Dependencies yang Perlu Ditambahkan ke `requirements.txt`:
- `lightrag-hku` (atau install from source via pip install -e)
- `asyncpg` (PostgreSQL async driver untuk LightRAG)
- `psycopg2-binary` (PostgreSQL driver, jika belum ada)

### Infrastructure Dependencies (setup di VPS):
- PostgreSQL server (untuk LightRAG KV/Graph/DocStatus)
- LightRAG Server running di port 9621

---

## 7. Konvensi Kode

Mengikuti conventions existing project:

- **Logging:** `setup_logging(service_name)` dari `shared.logging_config`
- **Security:** `InternalAuthMiddleware` dari `shared.security`
- **Config:** Import `config` dari `config.py`
- **HTTP Client:** `httpx.AsyncClient` pattern (sama seperti `service_client.py`)
- **Models:** Pydantic BaseModel
- **FastAPI lifespan:** `@asynccontextmanager` pattern
- **Error handling:** Return dict `{"status": "error", "error": "..."}` pattern
- **Import path:** `sys.path.insert(0, ...)` untuk root project access

---

## 8. Todo Tasks

```
[x] 1. Buat plan-implement-v4.md
[ ] 2. Clone LightRAG repository ke project
[ ] 3. Buat services/lightrag_adapter/__init__.py
[ ] 4. Buat services/lightrag_adapter/config.py
[ ] 5. Buat services/lightrag_adapter/errors.py
[ ] 6. Buat services/lightrag_adapter/models.py
[ ] 7. Buat services/lightrag_adapter/source_mapper.py
[ ] 8. Buat services/lightrag_adapter/references.py
[ ] 9. Buat services/lightrag_adapter/client.py
[ ] 10. Buat services/lightrag_adapter/search.py
[ ] 11. Buat services/lightrag_adapter/sync.py
[ ] 12. Buat services/lightrag_adapter/fallback.py
[ ] 13. Buat services/lightrag_adapter/health.py
[ ] 14. Buat services/lightrag_adapter/main.py
[ ] 15. Update config.py — tambah LIGHTRAG_* vars
[ ] 16. Buat orchestrator/search_provider.py
[ ] 17. Update ecosystem.config.js
[ ] 18. Update .env.example
[ ] 19. Update services/__init__.py
[ ] 20. Final review dan validasi
```

---

## 9. Testing Strategy (di VPS nanti)

### Smoke Test
1. Start LightRAG Server dengan config yang benar
2. Start LightRAG Adapter (`pm2 start ecosystem.config.js --only lightrag-adapter`)
3. `curl http://localhost:5015/health` → harus return healthy
4. `curl -H "X-API-Key: ..." http://localhost:5000/health` → harus include lightrag_adapter status

### Search Test (Shadow Mode)
1. Set `RAG_SEARCH_ENGINE=shadow` di `.env`
2. Restart Orchestrator
3. `POST /api/search` → harus return legacy result + log LightRAG comparison

### Sync Test
1. `POST /internal/sync/text` ke adapter dengan sample data
2. Verifikasi LightRAG Server menerima dan mengindex data

---

**End of Plan**
