# 📚 RAG Medan v3 - API Testing Guide

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Masuk ke folder project
cd C:\Kominfo\rag-medan-v3

# Buat virtual environment
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy dan edit env file
copy .env.example .env
# Edit .env sesuai konfigurasi lokal
```

### 2. Download Embedding Models

```bash
# Buat folder models (sesuaikan path di .env)
mkdir D:\models

# Download via Python
python -c "
from sentence_transformers import SentenceTransformer
# Small model (384 dim) - untuk text, usulan, web
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('D:/models/all-MiniLM-L6-v2')

# Large model (1024 dim) - untuk document
model_large = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model_large.save('D:/models/paraphrase-multilingual-mpnet-base-v2')
"
```

### 3. Start Qdrant

```bash
# Via Docker
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Atau download binary dari https://qdrant.tech/documentation/quick-start/
```

### 4. Start Services

```bash
# Terminal 1: Orchestrator (port 5001)
.venv\Scripts\activate; python -m uvicorn orchestrator.orchestrator:app --host 0.0.0.0 --port 5001 --reload

# Terminal 2: RAG Text Service (port 5010)
.venv\Scripts\activate; python -m uvicorn services.rag_text.main:app --host 0.0.0.0 --port 5010 --reload

# Terminal 3: RAG Document Service (port 5011)
.venv\Scripts\activate; python -m uvicorn services.rag_document.main:app --host 0.0.0.0 --port 5011 --reload

# Terminal 4: RAG Usulan Service (port 5013)
.venv\Scripts\activate; python -m uvicorn services.rag_usulan.main:app --host 0.0.0.0 --port 5013 --reload

# Terminal 5: RAG Web Service (port 5012)
.venv\Scripts\activate; python -m uvicorn services.rag_web.main:app --host 0.0.0.0 --port 5012 --reload
```

Atau gunakan PM2:
```bash
pm2 start ecosystem.config.js
```

---

## 📋 API Endpoints

### Base URL: `http://localhost:5001/api`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Search di RAG Text (knowledge_bank) |
| `/api/sync` | POST | Sync data ke knowledge_bank |
| `/api/doc-search` | POST | Search di RAG Document |
| `/api/doc-sync` | POST | Sync dokumen (trigger OCR) |
| `/api/doc-sync/status/{task_id}` | GET | Cek status task OCR |
| `/api/doc-delete` | DELETE | Soft delete dokumen |
| `/api/search-usulan` | POST | Search di RAG Usulan |
| `/api/sync-usulan` | POST | Sync data usulan |
| `/api/web-search` | POST | Search di RAG Web |
| `/api/web-trigger` | POST | Trigger web scraping |
| `/api/web-sync` | POST | Sync edited content |
| `/api/web-delete` | DELETE | Delete web content |
| `/health` | GET | Health check orchestrator |

---

## 🧪 CURL Testing Examples

### Health Check

```bash
# Check Orchestrator
curl -X GET http://localhost:5001/health

# Response:
{
  "status": "healthy",
  "service": "orchestrator",
  "version": "3.0.0",
  "services": {
    "text": "healthy",
    "document": "healthy",
    "web": "healthy",
    "usulan": "healthy"
  }
}
```

---

## 📝 RAG TEXT (knowledge_bank)

### 1. SYNC - Bulk Sync Data

Sync semua data dari web manajemen ke knowledge_bank.

```bash
curl -X POST http://localhost:5001/api/sync \
  -H "Content-Type: application/json" \
  -d '{
    "action": "bulk_sync",
    "content": [
      {
        "question_rag_id": "550e8400-e29b-41d4-a716-446655440001",
        "question_id": "550e8400-e29b-41d4-a716-000000000001",
        "answer_id": "550e8400-e29b-41d4-a716-aaa000000001",
        "category_id": "550e8400-e29b-41d4-a716-ccc000000001",
        "question": "Bagaimana cara mengurus KTP?",
        "question_rag_name": "Cara mengurus KTP baru di Kota Medan"
      },
      {
        "question_rag_id": "550e8400-e29b-41d4-a716-446655440002",
        "question_id": "550e8400-e29b-41d4-a716-000000000002",
        "answer_id": "550e8400-e29b-41d4-a716-aaa000000002",
        "category_id": "550e8400-e29b-41d4-a716-ccc000000001",
        "question": "Syarat membuat KK baru?",
        "question_rag_name": "Persyaratan dan prosedur pembuatan Kartu Keluarga baru"
      },
      {
        "question_rag_id": "550e8400-e29b-41d4-a716-446655440003",
        "question_id": "550e8400-e29b-41d4-a716-000000000003",
        "answer_id": "550e8400-e29b-41d4-a716-aaa000000003",
        "category_id": "550e8400-e29b-41d4-a716-ccc000000002",
        "question": "Jam operasional kantor Disdukcapil?",
        "question_rag_name": "Jam buka dan tutup pelayanan Disdukcapil Kota Medan"
      }
    ]
  }'

# Response:
{
  "status": "success",
  "message": "Sinkronisasi 3 data berhasil",
  "total_synced": 3
}
```

### 2. SYNC - Add Single Data

```bash
curl -X POST http://localhost:5001/api/sync \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add",
    "content": {
      "question_rag_id": "550e8400-e29b-41d4-a716-446655440004",
      "question_id": "550e8400-e29b-41d4-a716-000000000004",
      "answer_id": "550e8400-e29b-41d4-a716-aaa000000004",
      "category_id": "550e8400-e29b-41d4-a716-ccc000000001",
      "question": "Cara membuat akta kelahiran?",
      "question_rag_name": "Prosedur dan syarat pembuatan akta kelahiran anak"
    }
  }'

# Response:
{
  "status": "success",
  "message": "Data berhasil ditambahkan",
  "id": "550e8400-e29b-41d4-a716-446655440004"
}
```

### 3. SYNC - Update Data

```bash
curl -X POST http://localhost:5001/api/sync \
  -H "Content-Type: application/json" \
  -d '{
    "action": "update",
    "content": {
      "question_rag_id": "550e8400-e29b-41d4-a716-446655440001",
      "question_id": "550e8400-e29b-41d4-a716-000000000001",
      "answer_id": "550e8400-e29b-41d4-a716-aaa000000001",
      "category_id": "550e8400-e29b-41d4-a716-ccc000000001",
      "question": "Bagaimana cara mengurus KTP elektronik?",
      "question_rag_name": "Cara mengurus KTP elektronik (e-KTP) di Kota Medan"
    }
  }'

# Response:
{
  "status": "success",
  "message": "Data berhasil diperbarui"
}
```

### 4. SYNC - Delete Data

```bash
curl -X POST http://localhost:5001/api/sync \
  -H "Content-Type: application/json" \
  -d '{
    "action": "delete",
    "content": {
      "question_rag_id": "550e8400-e29b-41d4-a716-446655440004"
    }
  }'

# Response:
{
  "status": "success",
  "message": "Data berhasil dihapus"
}
```

### 5. SEARCH - Cari Jawaban

```bash
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "gimana cara bikin KTP?",
    "wa_number": "6281234567890"
  }'

# Response SUCCESS:
{
  "status": "success",
  "message": "Hasil ditemukan",
  "source": "text",
  "data": {
    "similar_questions": [
      {
        "question": "Bagaimana cara mengurus KTP elektronik?",
        "question_rag_name": "Cara mengurus KTP elektronik (e-KTP) di Kota Medan",
        "answer_id": "550e8400-e29b-41d4-a716-aaa000000001",
        "answer_doc": "",
        "category_id": "550e8400-e29b-41d4-a716-ccc000000001",
        "dense_score": 0.912,
        "overlap_score": 0.35,
        "final_score": 0.715,
        "note": "auto_accepted_by_dense"
      }
    ],
    "metadata": {
      "wa_number": "6281234567890",
      "original_question": "gimana cara bikin KTP?",
      "final_question": "cara bikin ktp",
      "category": "Kependudukan",
      "ai_reason": "-",
      "ai_reformulated": "-",
      "final_score_top": 0.715
    }
  },
  "timing": {
    "ai_domain_sec": 0.523,
    "ai_relevance_sec": 0.412,
    "embedding_sec": 0.045,
    "qdrant_sec": 0.012,
    "total_sec": 1.002
  }
}

# Response LOW_CONFIDENCE (tidak ada hasil relevan):
{
  "status": "low_confidence",
  "message": "Tidak ada hasil cukup relevan",
  "source": "text",
  "data": {
    "similar_questions": [...],
    "metadata": {...}
  },
  "timing": {...}
}
```

---

## 📄 RAG DOCUMENT (document_bank)

### 1. DOC-SYNC - Upload & Process Dokumen

```bash
curl -X POST http://localhost:5001/api/doc-sync \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
    "opd_name": "Dinas Pendidikan",
    "file_url": "D:/documents/surat_edaran.pdf"
  }'

# Response:
{
  "status": "queued",
  "task_id": "d290f1ee-6c54-4b01-90e6-d701748f0851_1704520800",
  "message": "Dokumen sedang diproses. Gunakan GET /api/doc-sync/status/{task_id} untuk cek status.",
  "file_size_mb": 2.45
}
```

### 2. DOC-SYNC STATUS - Cek Progress OCR

```bash
curl -X GET "http://localhost:5001/api/doc-sync/status/d290f1ee-6c54-4b01-90e6-d701748f0851_1704520800"

# Response (processing):
{
  "status": "processing",
  "result": null,
  "updated_at": "2026-01-06T10:15:30.123456"
}

# Response (completed):
{
  "status": "completed",
  "result": {
    "status": "success",
    "chunks_indexed": 15,
    "total_pages": 5
  },
  "updated_at": "2026-01-06T10:16:45.789012"
}

# Response (error):
{
  "status": "error",
  "result": {
    "message": "OCR process timeout (10 minutes)"
  },
  "updated_at": "2026-01-06T10:25:30.123456"
}
```

### 3. DOC-SEARCH - Cari di Dokumen

```bash
curl -X POST http://localhost:5001/api/doc-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "peraturan tentang jam sekolah",
    "limit": 5
  }'

# Response SUCCESS:
{
  "status": "success",
  "mode": "direct",
  "query": "peraturan tentang jam sekolah",
  "results": [
    {
      "doc_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
      "opd": "Dinas Pendidikan",
      "filename": "surat_edaran.pdf",
      "page_number": 2,
      "chunk_index": 3,
      "section": "BAB II",
      "summary": "Ketentuan jam belajar siswa...",
      "text": "Berdasarkan Peraturan Walikota Medan nomor 15 tahun 2025, jam operasional sekolah dimulai pukul 07.00 WIB dan berakhir pukul 14.00 WIB untuk jenjang SD dan SMP...",
      "score": 0.823
    }
  ]
}

# Response EMPTY:
{
  "status": "empty",
  "results": []
}
```

### 4. DOC-DELETE - Hapus Dokumen

```bash
curl -X DELETE http://localhost:5001/api/doc-delete \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "d290f1ee-6c54-4b01-90e6-d701748f0851"
  }'

# Response:
{
  "status": "deleted",
  "deleted": 15
}

# Response NOT FOUND:
{
  "status": "not_found",
  "deleted": 0
}
```

---

## 📊 RAG USULAN (usulan_bank)

### 1. SYNC-USULAN - Bulk Sync

```bash
curl -X POST http://localhost:5001/api/sync-usulan \
  -H "Content-Type: application/json" \
  -d '{
    "action": "bulk_sync",
    "content": [
      {
        "request_rag_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "request_id": "a1b2c3d4-0000-0000-0000-000000000101",
        "organization_id": "a1b2c3d4-0000-0000-0000-org000000005",
        "request_name": "Perbaikan jalan berlubang",
        "request_rag_name": "Usulan perbaikan jalan berlubang di Jalan Gatot Subroto Medan"
      },
      {
        "request_rag_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
        "request_id": "b2c3d4e5-0000-0000-0000-000000000102",
        "organization_id": "b2c3d4e5-0000-0000-0000-org000000003",
        "request_name": "Lampu jalan mati",
        "request_rag_name": "Permohonan perbaikan lampu penerangan jalan umum yang mati"
      }
    ]
  }'

# Response:
{
  "status": "success",
  "message": "2 data berhasil disinkronkan ke usulan_bank"
}
```

### 2. SYNC-USULAN - Add Single

```bash
curl -X POST http://localhost:5001/api/sync-usulan \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add",
    "content": {
      "request_rag_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
      "request_id": "c3d4e5f6-0000-0000-0000-000000000103",
      "organization_id": "c3d4e5f6-0000-0000-0000-org000000007",
      "request_name": "Banjir di kelurahan",
      "request_rag_name": "Laporan banjir dan permintaan penanganan drainase di Kelurahan Sunggal"
    }
  }'

# Response:
{
  "status": "success",
  "message": "Data add berhasil"
}
```

### 3. SYNC-USULAN - Update

```bash
curl -X POST http://localhost:5001/api/sync-usulan \
  -H "Content-Type: application/json" \
  -d '{
    "action": "update",
    "content": {
      "request_rag_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "request_id": "a1b2c3d4-0000-0000-0000-000000000101",
      "organization_id": "a1b2c3d4-0000-0000-0000-org000000005",
      "request_name": "Perbaikan jalan berlubang (URGENT)",
      "request_rag_name": "Usulan URGENT perbaikan jalan berlubang di Jalan Gatot Subroto Medan"
    }
  }'

# Response:
{
  "status": "success",
  "message": "Data update berhasil"
}
```

### 4. SYNC-USULAN - Delete

```bash
curl -X POST http://localhost:5001/api/sync-usulan \
  -H "Content-Type: application/json" \
  -d '{
    "action": "delete",
    "content": {
      "request_rag_id": "c3d4e5f6-a7b8-9012-cdef-123456789012"
    }
  }'

# Response:
{
  "status": "success",
  "message": "Data berhasil dihapus"
}
```

### 5. SEARCH-USULAN - Cari Usulan

```bash
curl -X POST http://localhost:5001/api/search-usulan \
  -H "Content-Type: application/json" \
  -d '{
    "question": "jalan rusak di gatot subroto",
    "wa_number": "6281234567890"
  }'

# Response SUCCESS:
{
  "status": "success",
  "message": "Hasil ditemukan",
  "data": {
    "similar_questions": [
      {
        "request_id": "a1b2c3d4-0000-0000-0000-000000000101",
        "organization_id": "a1b2c3d4-0000-0000-0000-org000000005",
        "request_name": "Perbaikan jalan berlubang (URGENT)",
        "request_rag_name": "Usulan URGENT perbaikan jalan berlubang di Jalan Gatot Subroto Medan",
        "dense_score": 0.892,
        "final_score": 0.892,
        "note": "Data yang Relevan Ditemukan"
      }
    ],
    "metadata": {
      "wa_number": "6281234567890",
      "user_question": "jalan rusak di gatot subroto",
      "final_score_top": 0.892
    }
  },
  "timing": {
    "reform_sec": 0.312,
    "embedding_sec": 0.042,
    "qdrant_sec": 0.015,
    "total_sec": 0.389
  }
}

# Response LOW_CONFIDENCE:
{
  "status": "low_confidence",
  "message": "Tidak ada hasil cukup relevan",
  "data": {
    "similar_questions": [...],
    "metadata": {...}
  },
  "timing": {...}
}
```

---

## 🌐 RAG WEB (web_scraping_bank)

### 1. WEB-TRIGGER - Trigger Scraping URL

```bash
curl -X POST http://localhost:5001/api/web-trigger \
  -H "Content-Type: application/json" \
  -d '{
    "link_id": "f47ac10b-58cc-4372-a567-web000000001",
    "url": "https://pemkomedan.go.id/berita/pengumuman-libur-nasional",
    "metadata": {
      "category": "pengumuman",
      "source": "pemkomedan"
    }
  }'

# Response:
{
  "status": "processing",
  "message": "Scraping job started",
  "link_id": "f47ac10b-58cc-4372-a567-web000000001",
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

### 2. WEB-SYNC - Sync Edited Content

```bash
curl -X POST http://localhost:5001/api/web-sync \
  -H "Content-Type: application/json" \
  -d '{
    "link_id": "f47ac10b-58cc-4372-a567-web000000001",
    "edited_content": "Pengumuman resmi dari Pemerintah Kota Medan tentang libur nasional dan cuti bersama tahun 2026. Tanggal libur: 1 Januari (Tahun Baru), 29 Januari (Imlek), dst..."
  }'

# Response:
{
  "status": "success",
  "message": "Content synced successfully",
  "link_id": "f47ac10b-58cc-4372-a567-web000000001",
  "chunks_count": 3
}
```

### 3. WEB-SEARCH - Cari di Web Content

```bash
curl -X POST http://localhost:5001/api/web-search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "kapan libur tahun baru 2026?",
    "wa_number": "6281234567890"
  }'

# Response SUCCESS:
{
  "status": "success",
  "message": "Hasil ditemukan dari web scraping",
  "source": "web_scraping",
  "data": {
    "similar_questions": [
      {
        "question": "-",
        "question_rag_name": "-",
        "answer_id": null,
        "answer_doc": "Pengumuman resmi dari Pemerintah Kota Medan tentang libur nasional dan cuti bersama tahun 2026. Tanggal libur: 1 Januari (Tahun Baru)...",
        "category_id": null,
        "dense_score": 0.845,
        "overlap_score": 0.0,
        "final_score": 0.845,
        "note": "from_web_scraping"
      }
    ],
    "metadata": {
      "wa_number": "6281234567890",
      "original_question": "kapan libur tahun baru 2026?",
      "final_question": "kapan libur tahun baru 2026",
      "category": "Web Scraping",
      "ai_reason": "",
      "ai_reformulated": "",
      "final_score_top": 0.845,
      "web_info": {
        "url": "https://pemkomedan.go.id/berita/pengumuman-libur-nasional",
        "title": "Pengumuman Libur Nasional 2026",
        "link_id": "f47ac10b-58cc-4372-a567-web000000001"
      }
    }
  },
  "timing": {
    "embedding_sec": 0.038,
    "qdrant_sec": 0.011,
    "total_sec": 0.052
  }
}
```

### 4. WEB-DELETE - Hapus Web Content

```bash
curl -X DELETE http://localhost:5001/api/web-delete \
  -H "Content-Type: application/json" \
  -d '{
    "link_id": "f47ac10b-58cc-4372-a567-web000000001"
  }'

# Response:
{
  "status": "success",
  "message": "Content deleted",
  "link_id": "f47ac10b-58cc-4372-a567-web000000001",
  "deleted_chunks": 3
}
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Service tidak bisa connect ke Qdrant**
```bash
# Cek Qdrant running
curl http://localhost:6333/collections

# Pastikan port tidak dipakai
netstat -ano | findstr :6333
```

**2. Embedding model tidak ditemukan**
```bash
# Cek path di .env
# Pastikan folder model ada dan berisi file config.json, model.safetensors, dll
```

**3. OCR timeout**
- Cek apakah PaddleOCR terinstall dengan benar
- Cek apakah file PDF tidak corrupt
- Kurangi ukuran file atau jumlah halaman

**4. LLM API Error**
```bash
# Test Gemini API
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
```

---

## 📌 Important Notes

1. **ID Format**: Semua `*_id` adalah **UUID v4** dari web manajemen
2. **Status Values**:
   - `success` = hasil ditemukan dengan confidence tinggi
   - `low_confidence` = tidak ada hasil yang cukup relevan
   - `error` = terjadi kesalahan sistem
   - `empty` = tidak ada data (khusus doc-search)
3. **Scoring**:
   - Text: `0.65 * dense_score + 0.35 * overlap_score`
   - Document: `score` langsung dari Qdrant
   - Usulan: `dense_score >= 0.85` untuk accepted
   - Web: `score` langsung dari Qdrant
