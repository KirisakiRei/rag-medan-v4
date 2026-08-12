"""
RAG Medan v4 - LightRAG Adapter — Source ID & URI Mapping.

Menyediakan deterministic logical ID untuk setiap source yang di-index
ke LightRAG, serta logical URI untuk citation traceability.

Format:
  Logical ID : kb:{knowledge_base_id}:{source_type}:{source_id}
  Source URI : sql://rag_text/123 | document://456/sop.pdf | https://...

Contoh:
  kb:medan-main:text:123
  kb:medan-main:document:456
  kb:medan-main:web:789
"""
import logging
from typing import Tuple, Optional

from services.lightrag_adapter.errors import SourceMappingError

logger = logging.getLogger("lightrag_adapter.source_mapper")


# ============== DOCUMENT ID ==============

def make_document_id(
    knowledge_base_id: str,
    source_type: str,
    source_id: str,
) -> str:
    """
    Generate deterministic LightRAG document ID.

    Args:
        knowledge_base_id: Workspace identifier (e.g. "medan-main")
        source_type: "text" | "document" | "web"
        source_id: Application-level primary key

    Returns:
        Logical ID string, e.g. "kb:medan-main:text:123"

    Raises:
        SourceMappingError: Jika parameter kosong.
    """
    if not source_type or not source_id:
        raise SourceMappingError(
            f"source_type and source_id required, got: "
            f"type={source_type!r}, id={source_id!r}"
        )
    return f"kb:{knowledge_base_id}:{source_type}:{source_id}"


def parse_document_id(document_id: str) -> Tuple[str, str, str]:
    """
    Parse LightRAG document ID kembali ke komponen asalnya.

    Args:
        document_id: Logical ID, e.g. "kb:medan-main:text:123"

    Returns:
        (knowledge_base_id, source_type, source_id)

    Raises:
        SourceMappingError: Jika format tidak valid.
    """
    parts = document_id.split(":")
    if len(parts) < 4 or parts[0] != "kb":
        raise SourceMappingError(f"Cannot parse document ID: {document_id!r}")
    kb_id = parts[1]
    source_type = parts[2]
    # source_id bisa mengandung ':' (misal URL), jadi join sisa parts
    source_id = ":".join(parts[3:])
    return kb_id, source_type, source_id


# ============== SOURCE URI ==============

def make_source_uri(source_type: str, source_id: str, **kwargs) -> str:
    """
    Generate logical source URI untuk citation traceability.

    Text    : sql://rag_text/123
    Document: document://456/sop-pelayanan.pdf
    Web     : https://example.go.id/page

    Args:
        source_type: "text" | "document" | "web"
        source_id: Application-level primary key
        **kwargs: Optional file_name (document), url (web)

    Returns:
        Logical URI string.
    """
    if source_type == "text":
        return f"sql://rag_text/{source_id}"
    elif source_type == "document":
        file_name = kwargs.get("file_name", "")
        if file_name:
            return f"document://{source_id}/{file_name}"
        return f"document://{source_id}"
    elif source_type == "web":
        url = kwargs.get("url", "")
        return url if url else f"web://{source_id}"
    else:
        return f"{source_type}://{source_id}"


# ============== CONTENT NORMALIZATION ==============

def normalize_text_content(
    title: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    category: Optional[str] = None,
    raw_content: Optional[str] = None,
) -> str:
    """
    Normalize text/FAQ content sebelum dikirim ke LightRAG.

    Jika raw_content disediakan, gunakan langsung.
    Jika tidak, bangun dari komponen Q&A + metadata.

    Returns:
        Normalized text string.
    """
    if raw_content:
        return raw_content

    parts = [f"Title: {title}"]
    if category:
        parts.append(f"Category: {category}")
    if question:
        parts.append(f"\nQuestion:\n{question}")
    if answer:
        parts.append(f"\nAnswer:\n{answer}")
    return "\n".join(parts)


def normalize_document_content(
    title: str,
    normalized_content: str,
    organization_id: Optional[str] = None,
    file_name: Optional[str] = None,
) -> str:
    """
    Normalize document content sebelum dikirim ke LightRAG.

    Header berisi metadata (title, organization, file),
    diikuti konten dokumen yang sudah diekstrak oleh Document Worker.

    Returns:
        Normalized text string.
    """
    header_parts = [f"Title: {title}"]
    if organization_id:
        header_parts.append(f"Organization: {organization_id}")
    if file_name:
        header_parts.append(f"File: {file_name}")
    return "\n".join(header_parts) + "\n\n" + normalized_content


def normalize_web_content(
    title: str,
    url: str,
    clean_content: str,
) -> str:
    """
    Normalize web page content sebelum dikirim ke LightRAG.

    Returns:
        Normalized text string.
    """
    return f"Title: {title}\nURL: {url}\n\n{clean_content}"
