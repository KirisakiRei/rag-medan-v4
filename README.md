# RAG Medan v3 - Unified RAG System

Sistem RAG (Retrieval-Augmented Generation) terintegrasi dengan arsitektur **Orchestrator Pattern** yang menggabungkan:

1. **RAG Text** - Pencarian di knowledge bank (Q&A)
2. **RAG Document** - Pencarian di dokumen PDF (OCR-based)
3. **RAG Web Scraping** - Pencarian di konten web yang sudah di-scrape (NEW)
4. **RAG Usulan** - Pencarian usulan masyarakat

## 🏗️ Architecture

```
                              ┌─────────────────┐
                              │     Client      │
                              │  (WhatsApp/API) │
                              └────────┬────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │     ORCHESTRATOR       │
                          │      (Port 5001)       │
                          │                        │
                          │  API Endpoints:        │
                          │  • /api/search         │
                          │  • /api/sync           │
                          │  • /api/doc-search     │
                          │  • /api/doc-sync       │
                          │  • /api/sync-usulan    │
                          │  • /api/search-usulan  │
                          └───────────┬────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│RAG TEXT SERVICE │        │RAG DOCUMENT SVC │        │RAG USULAN SVC   │
│   (Port 5010)   │        │   (Port 5011)   │        │   (Port 5013)   │
│                 │        │                 │        │                 │
│ /internal/search│        │ /internal/search│        │ /internal/search│
│ /internal/sync  │        │ /internal/sync  │        │ /internal/sync  │
└────────┬────────┘        │ /internal/delete│        └────────┬────────┘
         │                 └────────┬────────┘                  │
         ▼                          ▼                           ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ knowledge_bank  │        │ document_bank   │        │  usulan_bank    │
│  (Qdrant 384d)  │        │ (Qdrant 1024d)  │        │  (Qdrant 384d)  │
└─────────────────┘        └─────────────────┘        └─────────────────┘

                          ┌─────────────────┐
                          │RAG WEB SERVICE  │  (NEW in v3)
                          │   (Port 5012)   │
                          │                 │
                          │ /internal/search│
                          │ /internal/sync  │
                          │ /internal/trigger│
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │web_scraping_bank│
                          │  (Qdrant 384d)  │
                          └─────────────────┘
```

## 📁 Project Structure

```
rag-medan-v3/
├── app.py                      # Main entry point
├── config.py                   # Unified configuration
├── ecosystem.config.js         # PM2 configuration
├── requirements.txt            # Python dependencies
├── README.md
│
├── orchestrator/
│   ├── __init__.py
│   └── orchestrator.py         # Main orchestrator
│
├── services/
│   ├── __init__.py
│   ├── rag_text_service.py     # RAG Text service
│   ├── rag_document_service.py # RAG Document service
│   ├── rag_web_service.py      # RAG Web Scraping service
│   ├── rag_usulan_service.py   # RAG Usulan service
│   └── document_worker.py      # OCR subprocess worker
│
├── shared/
│   ├── __init__.py
│   ├── db.py                   # MySQL connection
│   ├── filtering.py            # AI filtering functions
│   ├── logging_config.py       # Logging setup
│   ├── ocr_utils.py            # OCR utilities
│   ├── prompts.py              # AI prompts
│   ├── schemas.py              # Pydantic models
│   ├── summarizer_utils.py     # Summarization utilities
│   └── utils.py                # General utilities
│
├── logs/                       # Service logs
└── document_temp/              # Temporary files for OCR
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Qdrant (running on port 6333)
- MySQL (for variables/prompts)
- PM2 (for production deployment)

### Installation

```bash
# Clone/create project directory
cd rag-medan-v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Configuration

Create `.env` file:

```env
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USERNAME=root
DB_PASSWORD=
DB_DATABASE=rag_medan

# Gemini API
LLM_API_KEY=your_api_key_here

# Embedding Models Path
EMB_MODEL_PATH=/path/to/small/model
EMB_LARGE_PATH=/path/to/large/model

# Service Ports (optional, defaults shown)
ORCHESTRATOR_PORT=5001
TEXT_SERVICE_PORT=5010
DOCUMENT_SERVICE_PORT=5011
WEB_SERVICE_PORT=5012
USULAN_SERVICE_PORT=5013
```

### Running Services

**Development (Single Service):**

```bash
python app.py text        # Start RAG text service only
python app.py document    # Start RAG document service only
python app.py web         # Start RAG web service only
python app.py usulan      # Start RAG usulan service only
python app.py orchestrator # Start orchestrator only
```

**Production (PM2):**

```bash
pm2 start ecosystem.config.js

# View logs
pm2 logs

# View status
pm2 status

# Stop all
pm2 stop all
```

## 📡 API Endpoints (V2 Compatible)

Semua endpoint **SAMA PERSIS** dengan v2 untuk backward compatibility.

### Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | Search text + fallback document |
| POST | `/api/doc-search` | Search dokumen only |
| POST | `/api/search-usulan` | Search usulan |

### Sync Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sync` | Sync knowledge_bank |
| POST | `/api/doc-sync` | Sync document (OCR) |
| GET | `/api/doc-sync/status/{task_id}` | Document sync status |
| DELETE | `/api/doc-delete` | Delete document |
| POST | `/api/sync-usulan` | Sync usulan_bank |

### Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root/info |
| GET | `/health` | Health check all services |

### Web Scraping Endpoints (NEW in v3)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/web-trigger` | Trigger web scraping |
| POST | `/api/web-sync` | Sync web content |
| DELETE | `/api/web-delete` | Delete web content |
| POST | `/api/web-search` | Search web content |

## 🔍 Search Request/Response (V2 Compatible)

### /api/search Request

```json
{
    "question": "Bagaimana cara mengurus KTP?",
    "wa_number": "+62812345678"
}
```

### /api/search Response

```json
{
    "status": "success",
    "message": "Hasil ditemukan",
    "source": "text",
    "data": {
        "similar_questions": [
            {
                "question": "Cara mengurus KTP baru",
                "question_rag_name": "Cara mengurus KTP...",
                "answer_id": 123,
                "answer_doc": "",
                "category_id": "1",
                "dense_score": 0.92,
                "overlap_score": 0.35,
                "final_score": 0.753,
                "note": "auto_accepted_by_dense"
            }
        ],
        "metadata": {
            "wa_number": "+62812345678",
            "original_question": "Bagaimana cara mengurus KTP?",
            "final_question": "cara mengurus ktp",
            "category": "Kependudukan",
            "ai_reason": "-",
            "ai_reformulated": "-",
            "final_score_top": 0.753
        }
    },
    "timing": {
        "ai_domain_sec": 0.5,
        "ai_relevance_sec": 0.3,
        "embedding_sec": 0.1,
        "qdrant_sec": 0.2,
        "total_sec": 1.1
    }
}
```

## 🔧 Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Orchestrator | 5001 | Main API endpoint |
| RAG Text Service | 5010 | RAG Text Q&A |
| RAG Document Service | 5011 | RAG Document PDF |
| RAG Web Service | 5012 | RAG Web Scraping |
| RAG Usulan Service | 5013 | RAG Usulan |
| Qdrant | 6333 | Vector database |

## 📊 Qdrant Collections

| Collection | Dimension | Model | Description |
|------------|-----------|-------|-------------|
| knowledge_bank | 384 | small model | Q&A text pairs |
| document_bank | 1024 | large model | PDF document chunks |
| web_scraping_bank | 384 | small model | Web content chunks |
| usulan_bank | 384 | small model | Usulan content |

## 📦 Migration from v2

### Perubahan Arsitektur

| Aspek | v2 | v3 |
|-------|----|----|
| Pattern | Fallback (single process) | Orchestrator (multi-service) |
| Services | All in one | Separated per function |
| Scaling | Vertical | Horizontal per service |
| Deploy | Single uvicorn | PM2 multi-process |

### Endpoint Compatibility

✅ **100% Backward Compatible** - Semua endpoint v2 tetap sama:
- `/api/search`
- `/api/sync`
- `/api/doc-search`
- `/api/doc-sync`
- `/api/doc-sync/status/{task_id}`
- `/api/doc-delete`
- `/api/sync-usulan`
- `/api/search-usulan`

### Fitur Baru v3

- ✨ Web Scraping RAG
- ✨ Orchestrator pattern
- ✨ Independent service scaling
- ✨ Better fault isolation

## 🐛 Troubleshooting

### Service not starting
- Check if port is already in use
- Verify Qdrant is running
- Check embedding model paths in config

### Search returns no results
- Check Qdrant collections exist
- Verify data has been synced
- Check score thresholds in config

### OCR timeout
- Increase timeout in document_worker.py
- Check PDF file is valid
- Ensure PaddleOCR is properly installed

## 📄 License

Internal - Pemkot Medan
