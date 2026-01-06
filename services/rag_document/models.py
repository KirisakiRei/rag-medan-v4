"""
RAG Document Service - Pydantic Models
Request/Response models untuk rag_document service
PERSIS SEPERTI V2!
"""
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ============== REQUEST MODELS (PERSIS V2) ==============

class SearchRequest(BaseModel):
    """Request dari orchestrator untuk search. (PERSIS V2: DocSearchRequest)"""
    query: str
    limit: int = 5


class SyncRequest(BaseModel):
    """Request untuk sync document. (PERSIS V2: DocSyncRequest)"""
    doc_id: str
    opd_name: Optional[str] = None  # PERSIS V2
    file_url: str


class DeleteRequest(BaseModel):
    """Request untuk delete document. (PERSIS V2: DocDeleteRequest)"""
    doc_id: str


# ============== RESPONSE MODELS (PERSIS V2) ==============

# Note: V2 doc-search returns format ini:
# {
#     "status": "success" | "empty",
#     "mode": "direct" | "post-summary",
#     "query": "...",
#     "results": [
#         {
#             "doc_id": ...,      -> mysql_id di V2
#             "opd": ...,
#             "filename": ...,
#             "page_number": ...,
#             "chunk_index": ...,
#             "section": ...,
#             "summary": ...,
#             "text": ...,
#             "score": ...
#         }
#     ],
#     # jika post-summary mode:
#     "summary": "..."
# }
