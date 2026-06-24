"""Layout-aware PDF extraction helpers.

This module keeps the public document pipeline text-based while improving how
PDF text layers are converted into page text and structured blocks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional

import fitz


@dataclass
class PdfLayoutResult:
    pages: Dict[int, str]
    blocks: List[dict]


@dataclass
class _RawBlock:
    page_number: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    is_bold: bool
    page_width: float
    page_height: float


_WHITESPACE_RE = re.compile(r"[ \t]+")
_LIST_HINT_RE = re.compile(r"^(\d+[\.\)]|[-*]|[a-zA-Z][\.\)])\s+")
_OFFICIAL_HEADING_RE = re.compile(
    r"^(bab|bagian|section|pasal|lampiran|judul|ketentuan|persyaratan|prosedur|alur|tahapan)\b",
    re.IGNORECASE,
)

_logger = __import__("logging").getLogger("pdf_layout_extractor")


def _extract_table_blocks_from_page(
    page,
    page_number: int,
) -> tuple[list[dict], list[tuple]]:
    """Ekstrak blok tabel terstruktur dari halaman PDF menggunakan find_tables().

    Mengembalikan (table_blocks, table_bboxes).
    - table_blocks: list blok dict siap masuk ke pipeline (source_kind="table")
    - table_bboxes: list (x0, y0, x1, y1) area tabel untuk filter raw block
    Fallback ke ([], []) jika find_tables tidak tersedia atau tabel tidak ditemukan.
    """
    table_blocks: list[dict] = []
    table_bboxes: list[tuple] = []
    try:
        finder = page.find_tables()
        if not finder or not getattr(finder, "tables", None):
            return table_blocks, table_bboxes

        for table in finder.tables:
            bbox = tuple(float(v) for v in table.bbox)
            rows = table.extract()
            if not rows:
                continue

            # Baris pertama sebagai header
            header = [str(c).strip() if c is not None else "" for c in rows[0]]
            header_text = " | ".join(header)
            data_rows = rows[1:] if len(rows) > 1 else []

            if not data_rows:
                # Tabel hanya header (mis. judul kolom tanpa data)
                if any(h.strip() for h in header):
                    table_blocks.append({
                        "page_number": page_number,
                        "text": header_text,
                        "block_type": "table_row",
                        "heading_level": None,
                        "heading_text": "",
                        "heading_path": [],
                        "source_kind": "table",
                        "block_order": 0,  # Akan di-reorder di extract_pdf_layout
                        "metadata": {"is_header": True},
                    })
                    table_bboxes.append(bbox)
            else:
                table_bboxes.append(bbox)
                for row_idx, row in enumerate(data_rows, start=1):
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    if not any(c for c in cells):
                        continue
                    row_text = " | ".join(cells)
                    combined = f"{header_text}\n{row_text}" if header_text else row_text
                    table_blocks.append({
                        "page_number": page_number,
                        "text": combined,
                        "block_type": "table_row",
                        "heading_level": None,
                        "heading_text": "",
                        "heading_path": [],
                        "source_kind": "table",
                        "block_order": 0,
                        "row_start": row_idx,
                        "row_end": row_idx,
                        "metadata": {"header_row": header},
                    })
    except Exception as exc:
        _logger.debug(f"[PDF-TABLE] Halaman {page_number}: find_tables gagal atau tidak tersedia — {exc}")

    return table_blocks, table_bboxes


def _raw_block_in_table(block: "_RawBlock", table_bboxes: list[tuple]) -> bool:
    """Kembalikan True jika raw block berada ≥30% overlap dengan area tabel.

    Digunakan untuk mencegah duplikasi teks tabel di antara raw blocks dan table_blocks.
    """
    bw = max(1.0, block.x1 - block.x0)
    bh = max(1.0, block.y1 - block.y0)
    block_area = bw * bh
    for tx0, ty0, tx1, ty1 in table_bboxes:
        ox = max(0.0, min(block.x1, tx1) - max(block.x0, tx0))
        oy = max(0.0, min(block.y1, ty1) - max(block.y0, ty0))
        if ox > 0 and oy > 0 and (ox * oy) / block_area >= 0.30:
            return True
    return False


def _normalize_line(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _normalize_text(text: str) -> str:
    lines = [_normalize_line(line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _flat_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalized_noise_key(text: str) -> str:
    lowered = _flat_text(text).lower()
    lowered = re.sub(r"\d+", "#", lowered)
    return lowered


def should_ocr_page(page_text: str, text_blocks: List[dict]) -> bool:
    """Return True when a PDF text layer is likely unusable for extraction."""
    normalized = _flat_text(page_text)
    if len(normalized) < 20:
        return True

    alpha_count = sum(1 for char in normalized if char.isalpha())
    if alpha_count < 12:
        return True

    block_texts = [_flat_text(block.get("text", "")) for block in text_blocks if _flat_text(block.get("text", ""))]
    if not block_texts:
        return True

    short_blocks = sum(1 for text in block_texts if len(text) <= 3)
    if len(block_texts) >= 3 and short_blocks / len(block_texts) >= 0.6:
        return True

    avg_words = sum(len(text.split()) for text in block_texts) / max(1, len(block_texts))
    if len(block_texts) >= 6 and avg_words < 2:
        return True

    return False


def _extract_raw_blocks(page, page_number: int) -> List[_RawBlock]:
    page_dict = page.get_text("dict") or {}
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    raw_blocks: List[_RawBlock] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            span_groups: List[List[dict]] = []
            current_group: List[dict] = []
            previous_x1: Optional[float] = None
            for span in sorted(line.get("spans", []), key=lambda item: item.get("bbox", (0, 0, 0, 0))[0]):
                span_text = _normalize_line(span.get("text", ""))
                if not span_text:
                    continue
                x0, _y0, _x1, _y1 = [float(value) for value in span.get("bbox", (0, 0, 0, 0))]
                if current_group and previous_x1 is not None and x0 - previous_x1 > 45:
                    span_groups.append(current_group)
                    current_group = []
                current_group.append(span)
                previous_x1 = _x1
            if current_group:
                span_groups.append(current_group)

            for group in span_groups:
                line_parts: List[str] = []
                font_sizes: List[float] = []
                bold_votes = 0
                span_count = 0
                bboxes = []
                for span in group:
                    span_text = _normalize_line(span.get("text", ""))
                    if not span_text:
                        continue
                    line_parts.append(span_text)
                    font_sizes.append(float(span.get("size", 0) or 0))
                    bboxes.append(tuple(float(value) for value in span.get("bbox", (0, 0, 0, 0))))
                    font_name = str(span.get("font", "")).lower()
                    if "bold" in font_name:
                        bold_votes += 1
                    span_count += 1

                text = _normalize_line(" ".join(line_parts))
                if not text or not bboxes:
                    continue

                x0 = min(bbox[0] for bbox in bboxes)
                y0 = min(bbox[1] for bbox in bboxes)
                x1 = max(bbox[2] for bbox in bboxes)
                y1 = max(bbox[3] for bbox in bboxes)
                raw_blocks.append(
                    _RawBlock(
                        page_number=page_number,
                        text=text,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        font_size=max(font_sizes) if font_sizes else 0.0,
                        is_bold=bold_votes > 0 and bold_votes >= span_count / 2,
                        page_width=page_width,
                        page_height=page_height,
                    )
                )

    return raw_blocks


def _find_repeated_marginal_noise(blocks_by_page: Dict[int, List[_RawBlock]]) -> set[str]:
    page_count = len(blocks_by_page)
    if page_count < 2:
        return set()

    candidates: Dict[str, set[int]] = {}
    for page_number, blocks in blocks_by_page.items():
        for block in blocks:
            in_margin = block.y0 <= block.page_height * 0.08 or block.y1 >= block.page_height * 0.94
            if not in_margin:
                continue
            key = _normalized_noise_key(block.text)
            if len(key) < 4:
                continue
            candidates.setdefault(key, set()).add(page_number)

    threshold = max(2, int(page_count * 0.5))
    return {key for key, pages in candidates.items() if len(pages) >= threshold}


def _has_multicolumn_layout(blocks: List[_RawBlock]) -> bool:
    if len(blocks) < 4:
        return False
    page_width = blocks[0].page_width
    left = [block for block in blocks if block.x0 < page_width * 0.42]
    right = [block for block in blocks if block.x0 > page_width * 0.45]
    if len(left) < 2 or len(right) < 2:
        return False

    left_right_edge = median([block.x1 for block in left])
    right_left_edge = median([block.x0 for block in right])
    if right_left_edge - left_right_edge < 35:
        return False

    for left_block in left:
        for right_block in right:
            overlaps_vertically = left_block.y0 <= right_block.y1 and right_block.y0 <= left_block.y1
            if overlaps_vertically or abs(left_block.y0 - right_block.y0) < 18:
                return True
    return False


def _order_blocks(blocks: List[_RawBlock]) -> List[_RawBlock]:
    if not blocks:
        return []
    if not _has_multicolumn_layout(blocks):
        return sorted(blocks, key=lambda block: (round(block.y0 / 4) * 4, block.x0))

    page_width = blocks[0].page_width

    def column_index(block: _RawBlock) -> int:
        return 0 if (block.x0 + block.x1) / 2 < page_width / 2 else 1

    return sorted(blocks, key=lambda block: (column_index(block), round(block.y0 / 4) * 4, block.x0))


def _is_upper_heading(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return False
    upper_letters = sum(1 for char in letters if char.isupper())
    return upper_letters / len(letters) >= 0.72


def _is_heading_candidate(block: _RawBlock, page_font_median: float) -> bool:
    text = _flat_text(block.text)
    if not text or len(text) > 140:
        return False
    if _OFFICIAL_HEADING_RE.match(text):
        return True
    if text.endswith((".", "?", "!", ";")):
        return False

    words = text.split()
    if block.font_size >= max(14.0, page_font_median * 1.35) and len(words) <= 14:
        return True
    if block.is_bold and len(words) <= 12 and not _LIST_HINT_RE.match(text):
        return True
    return False


def _heading_level(block: _RawBlock) -> int:
    lowered = _flat_text(block.text).lower()
    if lowered.startswith("bab"):
        return 1
    if lowered.startswith("bagian"):
        return 2
    if lowered.startswith("pasal"):
        return 3
    return 1 if block.font_size >= 18 else 2


def _legacy_block(
    *,
    page_number: int,
    text: str,
    block_type: str,
    heading_level: Optional[int],
    heading_text: str,
    heading_path: List[str],
    source_kind: str,
    block_order: int,
) -> dict:
    return {
        "page_number": page_number,
        "text": _flat_text(text),
        "block_type": block_type,
        "heading_level": heading_level,
        "heading_text": heading_text,
        "heading_path": list(heading_path),
        "source_kind": source_kind,
        "block_order": block_order,
        "metadata": {},
    }


def _blocks_from_plain_text(
    text: str,
    *,
    page_number: int,
    source_kind: str,
    block_order_start: int,
) -> List[dict]:
    blocks: List[dict] = []
    heading_path: List[str] = []
    block_order = block_order_start
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text or "") if part.strip()]
    for paragraph in paragraphs:
        flat = _flat_text(paragraph)
        if not flat:
            continue
        block_type = "list_item" if _LIST_HINT_RE.match(flat) else "paragraph"
        heading_level = None
        heading_text = heading_path[-1] if heading_path else ""
        if _OFFICIAL_HEADING_RE.match(flat) or (_is_upper_heading(flat) and len(flat.split()) <= 12):
            block_type = "heading"
            heading_level = 1
            heading_path = [flat]
            heading_text = flat
        blocks.append(
            _legacy_block(
                page_number=page_number,
                text=flat,
                block_type=block_type,
                heading_level=heading_level,
                heading_text=heading_text,
                heading_path=heading_path,
                source_kind=source_kind,
                block_order=block_order,
            )
        )
        block_order += 1
    return blocks


def _build_legacy_blocks(
    ordered_blocks_by_page: Dict[int, List[_RawBlock]],
    *,
    source_kind: str,
) -> List[dict]:
    blocks: List[dict] = []
    heading_path: List[str] = []
    block_order = 0

    for page_number in sorted(ordered_blocks_by_page):
        page_blocks = ordered_blocks_by_page[page_number]
        font_sizes = [block.font_size for block in page_blocks if block.font_size > 0]
        page_font_median = median(font_sizes) if font_sizes else 11.0

        for block in page_blocks:
            flat = _flat_text(block.text)
            if not flat:
                continue

            block_type = "list_item" if _LIST_HINT_RE.match(flat) else "paragraph"
            heading_level = None
            heading_text = heading_path[-1] if heading_path else ""

            if _is_heading_candidate(block, page_font_median):
                block_type = "heading"
                heading_level = _heading_level(block)
                heading_path = heading_path[: heading_level - 1] + [flat]
                heading_text = flat

            blocks.append(
                _legacy_block(
                    page_number=page_number,
                    text=flat,
                    block_type=block_type,
                    heading_level=heading_level,
                    heading_text=heading_text,
                    heading_path=heading_path,
                    source_kind=source_kind,
                    block_order=block_order,
                )
            )
            block_order += 1

    return blocks


def _page_text_from_blocks(blocks: Iterable[_RawBlock]) -> str:
    return "\n\n".join(_flat_text(block.text) for block in blocks if _flat_text(block.text)).strip()


def _page_text_from_legacy_blocks(blocks: Iterable[dict]) -> str:
    return "\n\n".join(_flat_text(block.get("text", "")) for block in blocks if _flat_text(block.get("text", ""))).strip()


def extract_pdf_layout(
    pdf_path: str,
    *,
    source_kind: str = "ocr",
    ocr_page_callback: Optional[Callable[[int, object], str]] = None,
) -> PdfLayoutResult:
    """Extract page text and legacy structured blocks from a PDF."""
    document = fitz.open(pdf_path)
    try:
        raw_blocks_by_page: Dict[int, List[_RawBlock]] = {}
        # Kumpulkan tabel terstruktur dan bbox-nya di pass pertama
        table_blocks_by_page: Dict[int, list[dict]] = {}
        table_bboxes_by_page: Dict[int, list[tuple]] = {}

        for page_index in range(len(document)):
            page_number = page_index + 1
            page = document[page_index]
            raw_blocks_by_page[page_number] = _extract_raw_blocks(page, page_number)
            tbl_blocks, tbl_bboxes = _extract_table_blocks_from_page(page, page_number)
            table_blocks_by_page[page_number] = tbl_blocks
            table_bboxes_by_page[page_number] = tbl_bboxes

        repeated_noise = _find_repeated_marginal_noise(raw_blocks_by_page)
        ordered_blocks_by_page: Dict[int, List[_RawBlock]] = {}
        pages: Dict[int, str] = {}
        all_blocks: List[dict] = []
        # Halaman dengan text layer yang punya tabel terstruktur
        pages_with_tables: set[int] = set()
        block_order = 0

        for page_index in range(len(document)):
            page_number = page_index + 1
            tbl_bboxes = table_bboxes_by_page.get(page_number, [])

            # Keluarkan raw blocks yang berada di dalam area tabel
            # untuk mencegah duplikasi antara text layer dan table_blocks
            raw_blocks = [
                block for block in raw_blocks_by_page.get(page_number, [])
                if _normalized_noise_key(block.text) not in repeated_noise
                and not _raw_block_in_table(block, tbl_bboxes)
            ]
            ordered_raw_blocks = _order_blocks(raw_blocks)
            provisional_text = _page_text_from_blocks(ordered_raw_blocks)
            provisional_blocks = [{"text": block.text} for block in ordered_raw_blocks]

            if should_ocr_page(provisional_text, provisional_blocks) and ocr_page_callback:
                # Halaman scan: gunakan OCR, skip table injection
                # (struktur tabel hilang saat discan — tidak bisa diandalkan)
                ocr_text = ocr_page_callback(page_number, document[page_index])

                # Pilih parser sesuai OCR_MODE:
                # - 'api'  : LLM mengeluarkan Markdown → gunakan _blocks_from_markdown
                # - 'local': PaddleOCR mengeluarkan plain text → gunakan _blocks_from_plain_text
                try:
                    from config import config as _cfg
                    _use_markdown = _cfg.OCR_MODE == "api"
                except Exception:
                    _use_markdown = False

                if _use_markdown:
                    from shared.ocr_utils import _blocks_from_markdown
                    ocr_blocks = _blocks_from_markdown(
                        ocr_text,
                        page_number=page_number,
                        source_kind=source_kind,
                        block_order_start=block_order,
                    )
                else:
                    ocr_blocks = _blocks_from_plain_text(
                        ocr_text,
                        page_number=page_number,
                        source_kind=source_kind,
                        block_order_start=block_order,
                    )

                if ocr_blocks:
                    pages[page_number] = _page_text_from_legacy_blocks(ocr_blocks)
                    all_blocks.extend(ocr_blocks)
                    block_order += len(ocr_blocks)
                    ordered_blocks_by_page[page_number] = []
                    continue

            ordered_blocks_by_page[page_number] = ordered_raw_blocks
            if table_blocks_by_page.get(page_number):
                pages_with_tables.add(page_number)

        text_layer_blocks = _build_legacy_blocks(ordered_blocks_by_page, source_kind=source_kind)

        # Inject table_blocks dari halaman text-layer ke hasil akhir
        if pages_with_tables:
            extra_table_blocks: List[dict] = []
            for page_number in pages_with_tables:
                extra_table_blocks.extend(table_blocks_by_page[page_number])
            if extra_table_blocks:
                text_layer_blocks = sorted(
                    all_blocks + text_layer_blocks + extra_table_blocks,
                    key=lambda block: (block["page_number"], block["block_order"]),
                )
                for index, block in enumerate(text_layer_blocks):
                    block["block_order"] = index
                all_blocks = []  # Sudah digabung di atas

        if all_blocks:
            text_layer_blocks = sorted(
                all_blocks + text_layer_blocks,
                key=lambda block: (block["page_number"], block["block_order"]),
            )
            for index, block in enumerate(text_layer_blocks):
                block["block_order"] = index

        for page_number in sorted(ordered_blocks_by_page):
            if page_number not in pages:
                page_blocks = [block for block in text_layer_blocks if block["page_number"] == page_number]
                pages[page_number] = _page_text_from_legacy_blocks(page_blocks)

        return PdfLayoutResult(pages=dict(sorted(pages.items())), blocks=text_layer_blocks)
    finally:
        document.close()

