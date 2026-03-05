"""
Document Chunker — RAG Medan v3
Semantic chunking untuk dokumen teks dan XLSX table chunking.

Ported from rag-medan-v2 core/chunker.py dengan adaptasi untuk arsitektur v3.
"""
import re
from typing import List, Optional


def semantic_chunk(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150
) -> List[str]:
    """
    Semantic sliding-window chunker dengan deteksi batas natural.

    Urutan prioritas break point:
      1. Paragraph break (\\n\\n)
      2. Sentence break ('. ')
      3. Hard cut pada chunk_size

    Args:
        text: Teks input yang sudah di-extract
        chunk_size: Target ukuran maksimum chunk (karakter)
        overlap: Jumlah karakter overlap antar chunk

    Returns:
        List[str]: Daftar chunk teks yang sudah dibersihkan
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            # Coba paragraph break (\n\n) dulu
            break_point = text.rfind('\n\n', start, end)
            if break_point != -1 and break_point > start:
                end = break_point + 2  # include \n\n
            else:
                # Fallback ke sentence break
                break_point = text.rfind('. ', start, end)
                if break_point != -1 and break_point > start:
                    end = break_point + 1  # include titik
                # Else: hard cut pada end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Geser start dengan overlap
        if end >= text_len:
            break
        start = max(start + 1, end - overlap)

    return chunks


def chunk_xlsx(
    rows: List[List[str]],
    rows_per_chunk: int = 30,
    max_size: int = 1200
) -> List[str]:
    """
    Chunker khusus untuk data tabel XLSX.

    Setiap chunk berisi:
      - Baris header (row pertama) diulang di setiap chunk
      - rows_per_chunk baris data
      - Jika ukuran chunk melebihi max_size, chunk dipecah lebih kecil

    Args:
        rows: List baris tabel. rows[0] = header, rows[1:] = data.
        rows_per_chunk: Jumlah baris data maksimum per chunk
        max_size: Ukuran karakter maksimum per chunk

    Returns:
        List[str]: Daftar chunk dalam format teks tabel
    """
    if not rows:
        return []

    header = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []

    if not data_rows:
        # Tidak ada data — kembalikan header saja jika ada
        header_text = " | ".join(str(c) for c in header if c is not None)
        return [header_text] if header_text.strip() else []

    header_text = " | ".join(str(c) for c in header if c is not None)
    chunks: List[str] = []

    batch_size = rows_per_chunk
    i = 0

    while i < len(data_rows):
        batch = data_rows[i:i + batch_size]
        batch_lines = []

        for row in batch:
            row_text = " | ".join(str(c) for c in row if c is not None)
            if row_text.strip():
                batch_lines.append(row_text)

        if not batch_lines:
            i += batch_size
            continue

        # Gabungkan header + batch
        chunk_text = (header_text + "\n" + "\n".join(batch_lines)).strip()

        # Jika chunk terlalu besar, potong lagi setengah
        if len(chunk_text) > max_size and len(batch_lines) > 1:
            half = max(1, len(batch) // 2)
            # Pecah rekursif: proses setengah pertama saja, sisanya diproses di iterasi berikutnya
            first_half_lines = []
            for row in batch[:half]:
                row_text = " | ".join(str(c) for c in row if c is not None)
                if row_text.strip():
                    first_half_lines.append(row_text)
            if first_half_lines:
                chunk_text = (header_text + "\n" + "\n".join(first_half_lines)).strip()
                chunks.append(chunk_text)

            # Proses sisa batch dengan batch_size yang lebih kecil
            remaining = batch[half:]
            for row in remaining:
                row_text = " | ".join(str(c) for c in row if c is not None)
                if row_text.strip():
                    small_chunk = (header_text + "\n" + row_text).strip()
                    if small_chunk.strip():
                        chunks.append(small_chunk)
        else:
            if chunk_text.strip():
                chunks.append(chunk_text)

        i += batch_size

    return chunks
