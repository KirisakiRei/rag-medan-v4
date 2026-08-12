"""
RAG Medan v4 - LightRAG Adapter Service (port 5015)

Adapter layer antara Orchestrator dan LightRAG Server.
Menyediakan unified search dan sync API untuk knowledge indexing
dan retrieval melalui LightRAG knowledge graph engine.

Modules:
- main.py: FastAPI app + endpoints
- config.py: Adapter configuration
- client.py: LightRAG HTTP client + circuit breaker
- models.py: Pydantic request/response models
- search.py: Unified search logic
- sync.py: Knowledge sync (text/document/web)
- source_mapper.py: Deterministic source ID + URI generation
- references.py: Citation/reference mapping
- fallback.py: Legacy fallback routing
- errors.py: Custom exceptions
- health.py: Health check logic
"""
