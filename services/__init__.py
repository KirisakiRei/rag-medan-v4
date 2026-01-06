"""
RAG Medan v3 - Services Package (Modular Architecture)

Available services:
- rag_text/: RAG Text Q&A (port 5010) - search, sync
- rag_document/: RAG Document OCR (port 5011) - search, sync, delete, worker
- rag_web/: RAG Web Scraping (port 5012) - search, sync, scraper, cleaner, chunker
- rag_usulan/: RAG Usulan (port 5013) - search, sync

Each service is modular with separate files:
- main.py: FastAPI app entry point
- search.py: Search functionality
- sync.py: Sync functionality  
- models.py: Pydantic models
- (additional files per service)"""