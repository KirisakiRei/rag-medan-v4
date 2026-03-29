"""
RAG Web Service - Chunker Module
Structure-aware chunking for HTML pages and FAQ content.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from config import config
from services.rag_document.chunker import ChunkItem, structure_chunk_document

logger = logging.getLogger("rag_web.chunker")

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCK_TAGS = {"p", "li", "blockquote", "pre", "td", "th"}
_SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas", "form", "button"}
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _extract_accordion_faq_blocks(root: Tag) -> List[Dict[str, Any]]:
    """Extract FAQ-style blocks from Bootstrap accordion markup."""
    blocks: List[Dict[str, Any]] = []
    accordion_items = root.select(".accordion-item")
    if not accordion_items:
        return blocks

    block_order = 0
    for item in accordion_items:
        question_node = item.select_one(".accordion-button, .accordion-header")
        answer_node = item.select_one(".accordion-body")
        question = _normalize_text(question_node.get_text(" ", strip=True)) if question_node else ""
        answer = _normalize_text(answer_node.get_text(" ", strip=True)) if answer_node else ""
        if not question or not answer:
            continue

        text = f"Pertanyaan: {question}\nJawaban: {answer}"
        blocks.append({
            "page_number": 1,
            "text": text,
            "block_type": "faq_item",
            "heading_level": None,
            "heading_text": "FAQ",
            "heading_path": ["FAQ"],
            "source_kind": "faq",
            "block_order": block_order,
            "metadata": {"faq_question": question},
        })
        block_order += 1

    return blocks


def _clean_soup(raw_html: str, css_selector: Optional[str] = None) -> Tag:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for tag_name in _SKIP_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    if css_selector:
        try:
            selected = soup.select(css_selector)
            if selected:
                wrapper = soup.new_tag("div")
                for element in selected:
                    wrapper.append(element)
                return wrapper
        except Exception as exc:
            logger.warning(f"[WEB-CHUNKER] Selector '{css_selector}' gagal untuk block extraction: {exc}")

    return soup.find("main") or soup.find("article") or soup.body or soup


def extract_html_blocks(
    raw_html: str,
    *,
    css_selector: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract structure-aware blocks from HTML."""
    root = _clean_soup(raw_html, css_selector=css_selector)
    faq_blocks = _extract_accordion_faq_blocks(root)
    if faq_blocks:
        return faq_blocks

    blocks: List[Dict[str, Any]] = []
    heading_path: List[str] = []
    order = 0

    def visit(node: Tag) -> None:
        nonlocal order, heading_path
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue

            tag_name = child.name.lower()
            if tag_name in _HEADING_TAGS:
                text = _normalize_text(child.get_text(" ", strip=True))
                if text:
                    level = _HEADING_TAGS[tag_name]
                    heading_path = heading_path[:level - 1] + [text]
                    blocks.append({
                        "page_number": 1,
                        "text": text,
                        "block_type": "heading",
                        "heading_level": level,
                        "heading_text": text,
                        "heading_path": list(heading_path),
                        "source_kind": "narrative",
                        "block_order": order,
                        "metadata": {},
                    })
                    order += 1
                continue

            if tag_name in {"table"}:
                rows = child.find_all("tr")
                header_cells: List[str] = []
                for row_index, row in enumerate(rows, start=1):
                    cells = [_normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                    cells = [cell for cell in cells if cell]
                    if not cells:
                        continue
                    if not header_cells:
                        header_cells = cells
                        blocks.append({
                            "page_number": 1,
                            "text": " | ".join(header_cells),
                            "block_type": "sheet_header",
                            "heading_level": None,
                            "heading_text": heading_path[-1] if heading_path else "",
                            "heading_path": list(heading_path),
                            "source_kind": "table",
                            "row_start": row_index,
                            "row_end": row_index,
                            "block_order": order,
                            "metadata": {},
                        })
                        order += 1
                        continue
                    row_text = " | ".join(header_cells) + "\n" + " | ".join(cells)
                    blocks.append({
                        "page_number": 1,
                        "text": row_text,
                        "block_type": "table_row",
                        "heading_level": None,
                        "heading_text": heading_path[-1] if heading_path else "",
                        "heading_path": list(heading_path),
                        "source_kind": "table",
                        "row_start": row_index,
                        "row_end": row_index,
                        "block_order": order,
                        "metadata": {},
                    })
                    order += 1
                continue

            if tag_name in _BLOCK_TAGS:
                text = _normalize_text(child.get_text(" ", strip=True))
                if text:
                    block_type = "list_item" if tag_name == "li" else "paragraph"
                    blocks.append({
                        "page_number": 1,
                        "text": text,
                        "block_type": block_type,
                        "heading_level": None,
                        "heading_text": heading_path[-1] if heading_path else "",
                        "heading_path": list(heading_path),
                        "source_kind": "narrative",
                        "block_order": order,
                        "metadata": {"tag_name": tag_name},
                    })
                    order += 1
                continue

            # Fallback for content containers such as div.accordion-body
            if tag_name == "div":
                nested_block_exists = child.find(list(_HEADING_TAGS) + list(_BLOCK_TAGS) + ["table"]) is not None
                text = _normalize_text(child.get_text(" ", strip=True))
                if text and not nested_block_exists and len(text) > 20:
                    blocks.append({
                        "page_number": 1,
                        "text": text,
                        "block_type": "paragraph",
                        "heading_level": None,
                        "heading_text": heading_path[-1] if heading_path else "",
                        "heading_path": list(heading_path),
                        "source_kind": "narrative",
                        "block_order": order,
                        "metadata": {"tag_name": tag_name},
                    })
                    order += 1
                    continue

            visit(child)

    visit(root)
    return blocks


def chunk_html(
    raw_html: str,
    *,
    css_selector: Optional[str] = None,
) -> List[ChunkItem]:
    """Create parent-child chunks from HTML blocks."""
    blocks = extract_html_blocks(raw_html, css_selector=css_selector)
    chunks = structure_chunk_document(
        blocks,
        child_chunk_size=config.WEB_CHILD_CHUNK_SIZE,
        parent_chunk_size=config.WEB_PARENT_CHUNK_SIZE,
        overlap=max(20, config.DOC_CHUNK_OVERLAP // 2),
        enable_semantic_merge=config.ENABLE_SEMANTIC_MERGE,
        similarity_threshold=config.SEMANTIC_MERGE_SIM_THRESHOLD,
    )
    logger.info(f"[WEB-CHUNKER] HTML blocks={len(blocks)} -> chunks={len(chunks)}")
    return chunks


def chunk_text(plain_text: str) -> List[ChunkItem]:
    """Create structure-aware chunks from plain text content."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", plain_text or "") if part.strip()]
    blocks: List[Dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs):
        blocks.append({
            "page_number": 1,
            "text": paragraph,
            "block_type": "paragraph",
            "heading_text": "",
            "heading_path": [],
            "source_kind": "narrative",
            "block_order": index,
            "metadata": {},
        })

    return structure_chunk_document(
        blocks,
        child_chunk_size=config.WEB_CHILD_CHUNK_SIZE,
        parent_chunk_size=config.WEB_PARENT_CHUNK_SIZE,
        overlap=max(20, config.DOC_CHUNK_OVERLAP // 2),
        enable_semantic_merge=config.ENABLE_SEMANTIC_MERGE,
        similarity_threshold=config.SEMANTIC_MERGE_SIM_THRESHOLD,
    )


def chunk_faq(
    pairs: list,
    *,
    heading_path: Optional[List[str]] = None,
) -> List[ChunkItem]:
    """Chunk FAQ pairs while preserving question and optional page heading path."""
    blocks: List[Dict[str, Any]] = []
    block_order = 0
    active_heading_path = list(heading_path or [])

    for question, answer in pairs:
        q_text = _normalize_text(question)
        a_text = _normalize_text(answer)
        if not q_text or not a_text:
            continue
        text = f"Pertanyaan: {q_text}\nJawaban: {a_text}"
        blocks.append({
            "page_number": 1,
            "text": text,
            "block_type": "faq_item",
            "heading_level": None,
            "heading_text": active_heading_path[-1] if active_heading_path else "",
            "heading_path": list(active_heading_path),
            "source_kind": "faq",
            "block_order": block_order,
            "metadata": {"faq_question": q_text},
        })
        block_order += 1

    return structure_chunk_document(
        blocks,
        child_chunk_size=config.WEB_CHILD_CHUNK_SIZE,
        parent_chunk_size=config.WEB_PARENT_CHUNK_SIZE,
        overlap=0,
        enable_semantic_merge=False,
        similarity_threshold=1.0,
    )
