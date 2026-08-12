"""
RAG Medan v4 - LightRAG Adapter — Citation & Reference Mapping.

Menerjemahkan raw context dari LightRAG ke format canonical yang
dipahami oleh Orchestrator dan downstream consumers.

Alur:
  LightRAG raw context (doc_id, content, score)
      ↓
  parse_document_id()  →  (kb_id, source_type, source_id)
      ↓
  make_source_uri()    →  logical URI
      ↓
  Canonical ContextItem (source_type, source_id, source_uri, reference_id)
"""
import logging
from typing import List, Dict, Any

from services.lightrag_adapter.source_mapper import (
    parse_document_id,
    make_source_uri,
)

logger = logging.getLogger("lightrag_adapter.references")


def _parse_source_descriptor(doc_descriptor: str) -> tuple[str, str]:
    """
    Parse source descriptor ke (source_type, source_id).

    Mendukung dua format:
    - Lengkap: "kb:medan-main:web:019d3e2c-..." (dari make_document_id)
    - Sederhana: "web:019d3e2c-..." (dari file_source ingest)

    Returns:
        (source_type, source_id); keduanya "" jika tidak bisa di-parse.
    """
    if not doc_descriptor:
        return "", ""

    parts = doc_descriptor.split(":")
    if len(parts) >= 4 and parts[0] == "kb":
        source_type = parts[2]
        source_id = ":".join(parts[3:])
        return source_type, source_id

    if len(parts) >= 2 and parts[0] in ("text", "document", "web"):
        source_type = parts[0]
        source_id = ":".join(parts[1:])
        return source_type, source_id

    return "", ""


def map_lightrag_context_to_canonical(
    lightrag_contexts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Transform LightRAG raw context items ke format canonical.

    Input (LightRAG raw):
        [{"content": "...", "doc_id": "kb:medan-main:text:123", "score": 0.85}, ...]

    Output (canonical):
        [{
            "content": "...",
            "source_type": "text",
            "source_id": "123",
            "title": "...",
            "source_uri": "sql://rag_text/123",
            "reference_id": "1",
            "score": 0.85
        }, ...]

    Args:
        lightrag_contexts: List of raw context dicts dari LightRAG response.

    Returns:
        List of canonical context dicts.
    """
    canonical = []

    for idx, ctx in enumerate(lightrag_contexts):
        doc_id = ctx.get("doc_id", "") or ctx.get("document_id", "")
        content = ctx.get("content", "") or ctx.get("text", "")
        score = ctx.get("score")

        # Parse document ID untuk dapatkan source_type dan source_id
        try:
            kb_id, source_type, source_id = parse_document_id(doc_id)
        except Exception:
            # Fallback: coba parse descriptor sederhana ("web:<id>")
            source_type, source_id = _parse_source_descriptor(doc_id)
            if not source_type:
                logger.warning(
                    f"Cannot parse doc_id '{doc_id}', "
                    f"skipping reference mapping"
                )
                source_type = "unknown"
                source_id = doc_id

        # Build logical source URI
        source_uri = make_source_uri(source_type, source_id)

        canonical.append({
            "content": content,
            "source_type": source_type,
            "source_id": source_id,
            "title": ctx.get("title", ""),
            "source_uri": source_uri,
            "reference_id": str(idx + 1),
            "score": score,
        })

    return canonical


def build_references(
    canonical_contexts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build unique reference list dari canonical contexts.

    Deduplicates by (source_type, source_id) — satu source hanya
    muncul sekali di references meskipun ada beberapa chunks.

    Args:
        canonical_contexts: List dari map_lightrag_context_to_canonical().

    Returns:
        Deduplicated list of reference dicts.
    """
    seen = set()
    references = []

    for ctx in canonical_contexts:
        key = (ctx["source_type"], ctx["source_id"])
        if key in seen:
            continue
        seen.add(key)

        ref = {
            "source_type": ctx["source_type"],
            "source_id": ctx["source_id"],
            "title": ctx.get("title", ""),
            "source_uri": ctx.get("source_uri", ""),
            "reference_id": ctx.get("reference_id", ""),
        }

        # Tambah URL khusus untuk web sources
        if ctx["source_type"] == "web":
            ref["url"] = ctx["source_uri"]

        references.append(ref)

    return references
