"""
Document Chunker - RAG Medan v3
Hybrid structure-aware chunking for narrative, OCR, and tabular sources.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


@dataclass
class ChunkBlock:
    """Logical text block extracted from a source document."""

    text: str
    page_number: int = 1
    block_type: str = "paragraph"
    heading_level: Optional[int] = None
    heading_text: str = ""
    heading_path: List[str] = field(default_factory=list)
    source_kind: str = "narrative"
    sheet_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    block_order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkItem:
    """Chunk ready to be embedded/indexed."""

    chunk_id: str
    text: str
    chunk_level: str
    chunk_kind: str
    source_kind: str
    page_start: int
    page_end: int
    section_title: str
    heading_path: str
    block_order: int
    parent_chunk_id: Optional[str] = None
    window_prev_id: Optional[str] = None
    window_next_id: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(len(text.split()), len(text) // 4)


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _adjacent_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left).lower()
    right_norm = _normalize_text(right).lower()
    if not left_norm or not right_norm:
        return 0.0

    token_left = _token_set(left_norm)
    token_right = _token_set(right_norm)
    jaccard = (
        len(token_left & token_right) / len(token_left | token_right)
        if token_left and token_right else 0.0
    )
    sequence = SequenceMatcher(None, left_norm[:500], right_norm[:500]).ratio()
    return max(jaccard, sequence * 0.55)


def _block_to_chunk_block(block: Dict[str, Any], block_order: int) -> ChunkBlock:
    heading_path = block.get("heading_path") or []
    if isinstance(heading_path, str):
        heading_path = [part.strip() for part in heading_path.split(" > ") if part.strip()]

    return ChunkBlock(
        text=_normalize_text(block.get("text", "")),
        page_number=int(block.get("page_number", 1) or 1),
        block_type=block.get("block_type", "paragraph"),
        heading_level=block.get("heading_level"),
        heading_text=block.get("heading_text", "") or "",
        heading_path=list(heading_path),
        source_kind=block.get("source_kind", "narrative"),
        sheet_name=block.get("sheet_name"),
        row_start=block.get("row_start"),
        row_end=block.get("row_end"),
        block_order=int(block.get("block_order", block_order)),
        metadata=dict(block.get("metadata") or {}),
    )


def semantic_chunk(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150
) -> List[str]:
    """
    Backward-compatible natural-break splitter.

    Kept for compatibility, but not the main chunking path anymore.
    """
    oversized = split_oversize_block(
        ChunkBlock(text=text),
        max_tokens=chunk_size,
        overlap=overlap,
    )
    return [block.text for block in oversized if block.text.strip()]


def split_oversize_block(
    block: ChunkBlock,
    max_tokens: int,
    overlap: int = 0,
) -> List[ChunkBlock]:
    """Split an oversized block while preserving section context."""
    if _estimate_tokens(block.text) <= max_tokens:
        return [block]

    prefix_parts = []
    if block.heading_path:
        prefix_parts.append(" > ".join(block.heading_path))
    elif block.heading_text:
        prefix_parts.append(block.heading_text)
    prefix = "\n".join(prefix_parts).strip()
    prefix_tokens = _estimate_tokens(prefix)
    available_tokens = max(80, max_tokens - prefix_tokens)

    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(block.text) if part.strip()]
    if not paragraphs:
        paragraphs = [block.text]

    split_blocks: List[ChunkBlock] = []
    current_parts: List[str] = []

    def flush_current() -> None:
        if not current_parts:
            return
        body = "\n\n".join(current_parts).strip()
        text = f"{prefix}\n{body}".strip() if prefix else body
        split_blocks.append(
            ChunkBlock(
                text=text,
                page_number=block.page_number,
                block_type=block.block_type,
                heading_level=block.heading_level,
                heading_text=block.heading_text,
                heading_path=list(block.heading_path),
                source_kind=block.source_kind,
                sheet_name=block.sheet_name,
                row_start=block.row_start,
                row_end=block.row_end,
                block_order=block.block_order,
                metadata=dict(block.metadata),
            )
        )

    for paragraph in paragraphs:
        para = _normalize_text(paragraph)
        if not para:
            continue

        if _estimate_tokens(para) > available_tokens:
            sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
            sentence_parts: List[str] = []
            for sentence in sentences:
                candidate = " ".join(sentence_parts + [sentence]).strip()
                if sentence_parts and _estimate_tokens(candidate) > available_tokens:
                    paragraph_piece = " ".join(sentence_parts).strip()
                    combined = "\n\n".join(current_parts + [paragraph_piece]).strip()
                    current_parts = [combined] if combined else []
                    flush_current()
                    current_parts = []
                    sentence_parts = [sentence]
                else:
                    sentence_parts.append(sentence)
            if sentence_parts:
                candidate_piece = " ".join(sentence_parts).strip()
                candidate_full = "\n\n".join(current_parts + [candidate_piece]).strip()
                if current_parts and _estimate_tokens(candidate_full) > available_tokens:
                    flush_current()
                    current_parts = [candidate_piece]
                else:
                    current_parts.append(candidate_piece)
            continue

        candidate = "\n\n".join(current_parts + [para]).strip()
        if current_parts and _estimate_tokens(candidate) > available_tokens:
            flush_current()
            if overlap > 0 and split_blocks:
                tail_words = split_blocks[-1].text.split()[-overlap:]
                tail_text = " ".join(tail_words).strip()
                current_parts = [tail_text, para] if tail_text else [para]
            else:
                current_parts = [para]
        else:
            current_parts.append(para)

    flush_current()
    return split_blocks or [block]


def semantic_merge_blocks(
    blocks: List[ChunkBlock],
    *,
    max_tokens: int,
    similarity_threshold: float,
    enabled: bool = True,
) -> List[ChunkBlock]:
    """Merge adjacent narrative blocks when they still form one topic."""
    if not enabled or len(blocks) < 2:
        return blocks

    merged: List[ChunkBlock] = []
    current = blocks[0]

    for nxt in blocks[1:]:
        same_context = (
            # Izinkan merge untuk narrative (docx/txt) dan ocr (PDF-OCR, gambar).
            # "table" dan sumber lain tetap tidak dimerge.
            current.source_kind == nxt.source_kind
            and current.source_kind in {"narrative", "ocr"}
            and current.block_type not in {"heading", "table_row", "sheet_header"}
            and nxt.block_type not in {"heading", "table_row", "sheet_header"}
            and current.heading_path == nxt.heading_path
            and abs(current.page_number - nxt.page_number) <= 1
        )

        if not same_context:
            merged.append(current)
            current = nxt
            continue

        combined_text = f"{current.text}\n\n{nxt.text}".strip()
        combined_tokens = _estimate_tokens(combined_text)
        similarity = _adjacent_similarity(current.text, nxt.text)
        should_merge = (
            combined_tokens <= max_tokens and (
                similarity >= similarity_threshold
                or _estimate_tokens(current.text) < 140
                or _estimate_tokens(nxt.text) < 140
            )
        )

        if should_merge:
            current = ChunkBlock(
                text=combined_text,
                page_number=min(current.page_number, nxt.page_number),
                block_type="paragraph",
                heading_level=current.heading_level or nxt.heading_level,
                heading_text=current.heading_text or nxt.heading_text,
                heading_path=list(current.heading_path or nxt.heading_path),
                source_kind=current.source_kind,
                sheet_name=current.sheet_name or nxt.sheet_name,
                row_start=current.row_start,
                row_end=nxt.row_end,
                block_order=min(current.block_order, nxt.block_order),
                metadata={**current.metadata, **nxt.metadata},
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def _section_identity(block: ChunkBlock) -> tuple:
    heading_identity = tuple(block.heading_path) if block.heading_path else (
        block.heading_text or f"page-{block.page_number}"
    )
    if block.source_kind in {"ocr", "table"}:
        return (block.source_kind, block.page_number, heading_identity, block.sheet_name or "")
    return (block.source_kind, heading_identity, block.sheet_name or "")


def _section_title(block: ChunkBlock) -> str:
    if block.heading_path:
        return block.heading_path[-1]
    if block.heading_text:
        return block.heading_text
    if block.sheet_name:
        return f"Sheet {block.sheet_name}"
    return f"Page {block.page_number}"


def _build_chunk_text(blocks: List[ChunkBlock], *, include_heading: bool = True) -> str:
    if not blocks:
        return ""

    heading_path = blocks[0].heading_path
    heading_text = blocks[0].heading_text
    prefix = " > ".join(heading_path).strip() if heading_path else heading_text.strip()

    body_parts = []
    for block in blocks:
        text = _normalize_text(block.text)
        if text:
            body_parts.append(text)

    body = "\n\n".join(body_parts).strip()
    if include_heading and prefix and not body.lower().startswith(prefix.lower()):
        return f"{prefix}\n{body}".strip()
    return body


def _make_chunk_item(
    *,
    chunk_level: str,
    chunk_kind: str,
    source_kind: str,
    blocks: List[ChunkBlock],
    parent_chunk_id: Optional[str] = None,
) -> ChunkItem:
    text = _build_chunk_text(blocks, include_heading=True)
    section_title = _section_title(blocks[0])
    heading_path = " > ".join(blocks[0].heading_path) if blocks[0].heading_path else blocks[0].heading_text
    metadata = {
        "sheet_name": blocks[0].sheet_name,
        "block_types": [block.block_type for block in blocks],
        "content_preview": text[:280],
        "normalized_text": _normalize_text(text.lower()),
        "row_start": blocks[0].row_start,
        "row_end": blocks[-1].row_end,
    }

    return ChunkItem(
        chunk_id=str(uuid.uuid4()),
        text=text,
        chunk_level=chunk_level,
        chunk_kind=chunk_kind,
        source_kind=source_kind,
        page_start=min(block.page_number for block in blocks),
        page_end=max(block.page_number for block in blocks),
        section_title=section_title,
        heading_path=heading_path,
        block_order=min(block.block_order for block in blocks),
        parent_chunk_id=parent_chunk_id,
        token_count=_estimate_tokens(text),
        metadata={k: v for k, v in metadata.items() if v not in ("", None, [])},
    )


def _split_section_into_children(
    blocks: List[ChunkBlock],
    *,
    child_chunk_size: int,
    overlap: int,
    parent_chunk_id: str,
) -> List[ChunkItem]:
    child_items: List[ChunkItem] = []
    current_blocks: List[ChunkBlock] = []

    def flush_current() -> None:
        if not current_blocks:
            return
        child_items.append(
            _make_chunk_item(
                chunk_level="child",
                chunk_kind=current_blocks[0].block_type,
                source_kind=current_blocks[0].source_kind,
                blocks=list(current_blocks),
                parent_chunk_id=parent_chunk_id,
            )
        )

    for block in blocks:
        split_blocks = split_oversize_block(block, max_tokens=child_chunk_size, overlap=overlap)
        for piece in split_blocks:
            piece_tokens = _estimate_tokens(piece.text)
            if not current_blocks:
                current_blocks = [piece]
                continue

            combined_text = _build_chunk_text(current_blocks + [piece])
            if _estimate_tokens(combined_text) <= child_chunk_size:
                current_blocks.append(piece)
                continue

            flush_current()
            if overlap > 0 and current_blocks:
                tail_words = current_blocks[-1].text.split()[-overlap:]
                tail_text = " ".join(tail_words).strip()
                if tail_text:
                    current_blocks = [
                        ChunkBlock(
                            text=tail_text,
                            page_number=current_blocks[-1].page_number,
                            block_type=current_blocks[-1].block_type,
                            heading_level=current_blocks[-1].heading_level,
                            heading_text=current_blocks[-1].heading_text,
                            heading_path=list(current_blocks[-1].heading_path),
                            source_kind=current_blocks[-1].source_kind,
                            sheet_name=current_blocks[-1].sheet_name,
                            row_start=current_blocks[-1].row_start,
                            row_end=current_blocks[-1].row_end,
                            block_order=current_blocks[-1].block_order,
                            metadata=dict(current_blocks[-1].metadata),
                        )
                    ]
                else:
                    current_blocks = []
            else:
                current_blocks = []

            if piece_tokens > child_chunk_size:
                current_blocks = split_oversize_block(piece, max_tokens=child_chunk_size, overlap=overlap)
                flush_current()
                current_blocks = []
            else:
                current_blocks.append(piece)

    flush_current()

    for index, item in enumerate(child_items):
        item.window_prev_id = child_items[index - 1].chunk_id if index > 0 else None
        item.window_next_id = child_items[index + 1].chunk_id if index < len(child_items) - 1 else None

    return child_items


def structure_chunk_document(
    blocks: List[Dict[str, Any]],
    *,
    child_chunk_size: int,
    parent_chunk_size: int,
    overlap: int,
    enable_semantic_merge: bool,
    similarity_threshold: float,
) -> List[ChunkItem]:
    """Create parent-child chunks from structured blocks."""
    if not blocks:
        return []

    normalized_blocks = [
        _block_to_chunk_block(block, order)
        for order, block in enumerate(blocks)
        if _normalize_text(block.get("text", ""))
    ]
    if not normalized_blocks:
        return []

    merged_blocks = semantic_merge_blocks(
        normalized_blocks,
        max_tokens=child_chunk_size,
        similarity_threshold=similarity_threshold,
        enabled=enable_semantic_merge,
    )

    all_items: List[ChunkItem] = []
    section_blocks: List[ChunkBlock] = []
    current_section_id = _section_identity(merged_blocks[0])
    current_section_tokens = 0

    def flush_section() -> None:
        nonlocal section_blocks, current_section_tokens
        if not section_blocks:
            return

        # Split large sections into parent groups while keeping section identity.
        parent_groups: List[List[ChunkBlock]] = []
        current_parent: List[ChunkBlock] = []
        for block in section_blocks:
            block_tokens = _estimate_tokens(block.text)
            if current_parent and current_section_tokens and (
                _estimate_tokens(_build_chunk_text(current_parent + [block])) > parent_chunk_size
            ):
                parent_groups.append(current_parent)
                current_parent = [block]
            else:
                current_parent.append(block)
        if current_parent:
            parent_groups.append(current_parent)

        for parent_group in parent_groups:
            parent_item = _make_chunk_item(
                chunk_level="parent",
                chunk_kind="section",
                source_kind=parent_group[0].source_kind,
                blocks=parent_group,
            )
            all_items.append(parent_item)
            all_items.extend(
                _split_section_into_children(
                    parent_group,
                    child_chunk_size=child_chunk_size,
                    overlap=overlap,
                    parent_chunk_id=parent_item.chunk_id,
                )
            )

        section_blocks = []
        current_section_tokens = 0

    for block in merged_blocks:
        block_section_id = _section_identity(block)
        projected_tokens = _estimate_tokens(_build_chunk_text(section_blocks + [block])) if section_blocks else _estimate_tokens(block.text)
        if section_blocks and (block_section_id != current_section_id or projected_tokens > parent_chunk_size * 1.35):
            flush_section()
            current_section_id = block_section_id

        section_blocks.append(block)
        current_section_tokens = projected_tokens

    flush_section()
    return all_items


def chunk_xlsx(
    rows: List[List[str]],
    rows_per_chunk: int = 30,
    max_size: int = 1200
) -> List[str]:
    """Backward-compatible XLSX row chunking helper."""
    if not rows:
        return []

    blocks: List[Dict[str, Any]] = []
    header = [str(cell) for cell in rows[0]] if rows else []
    header_text = " | ".join(cell for cell in header if cell is not None).strip()
    if header_text:
        blocks.append({
            "page_number": 1,
            "text": header_text,
            "block_type": "sheet_header",
            "heading_text": header_text,
            "heading_path": [header_text],
            "source_kind": "table",
            "block_order": 0,
        })

    for row_index, row in enumerate(rows[1:], start=1):
        row_text = " | ".join(str(cell) if cell is not None else "" for cell in row).strip()
        if not row_text:
            continue
        blocks.append({
            "page_number": 1,
            "text": f"{header_text}\n{row_text}" if header_text else row_text,
            "block_type": "table_row",
            "heading_text": header_text,
            "heading_path": [header_text] if header_text else [],
            "source_kind": "table",
            "row_start": row_index,
            "row_end": row_index,
            "block_order": row_index,
        })

    chunk_items = structure_chunk_document(
        blocks,
        child_chunk_size=max_size,
        parent_chunk_size=max_size * 2,
        overlap=0,
        enable_semantic_merge=False,
        similarity_threshold=0.0,
    )
    return [item.text for item in chunk_items if item.chunk_level == "child"]
