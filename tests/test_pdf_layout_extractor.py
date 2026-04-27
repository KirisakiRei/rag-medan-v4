import tempfile
import unittest
from pathlib import Path

import fitz

from shared.pdf_layout_extractor import (
    extract_pdf_layout,
    should_ocr_page,
)


def _make_pdf(pages):
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(temp_dir.name) / "sample.pdf"
    doc = fitz.open()
    for page_items in pages:
        page = doc.new_page(width=595, height=842)
        for item in page_items:
            page.insert_text(
                item["point"],
                item["text"],
                fontsize=item.get("size", 11),
                fontname=item.get("font", "helv"),
            )
    doc.save(path)
    doc.close()
    return temp_dir, path


class PdfLayoutExtractorTests(unittest.TestCase):
    def test_detects_heading_path_from_larger_text(self):
        temp_dir, pdf_path = _make_pdf([
            [
                {"point": (72, 72), "text": "TJONG A FIE MANSION", "size": 24},
                {"point": (72, 112), "text": "HERITAGE SEJARAH/ARSITEKTUR", "size": 11},
                {"point": (72, 132), "text": "Jl. Jendral Ahmad Yani No.105 Kesawan Barat", "size": 11},
            ]
        ])
        self.addCleanup(temp_dir.cleanup)

        result = extract_pdf_layout(str(pdf_path))

        self.assertEqual(result.pages[1], "TJONG A FIE MANSION\n\nHERITAGE SEJARAH/ARSITEKTUR\n\nJl. Jendral Ahmad Yani No.105 Kesawan Barat")
        self.assertEqual(result.blocks[0]["block_type"], "heading")
        self.assertEqual(result.blocks[0]["heading_path"], ["TJONG A FIE MANSION"])
        self.assertEqual(result.blocks[1]["heading_path"], ["TJONG A FIE MANSION"])

    def test_orders_multicolumn_blocks_by_column_before_row(self):
        temp_dir, pdf_path = _make_pdf([
            [
                {"point": (72, 72), "text": "LEFT ONE", "size": 11},
                {"point": (320, 72), "text": "RIGHT ONE", "size": 11},
                {"point": (72, 112), "text": "LEFT TWO", "size": 11},
                {"point": (320, 112), "text": "RIGHT TWO", "size": 11},
            ]
        ])
        self.addCleanup(temp_dir.cleanup)

        result = extract_pdf_layout(str(pdf_path))
        texts = [block["text"] for block in result.blocks]

        self.assertEqual(texts, ["LEFT ONE", "LEFT TWO", "RIGHT ONE", "RIGHT TWO"])

    def test_removes_repeated_header_and_footer(self):
        temp_dir, pdf_path = _make_pdf([
            [
                {"point": (72, 30), "text": "PEMERINTAH KOTA MEDAN", "size": 9},
                {"point": (72, 100), "text": "Isi halaman pertama", "size": 11},
                {"point": (72, 810), "text": "www.pemkomedan.go.id", "size": 9},
            ],
            [
                {"point": (72, 30), "text": "PEMERINTAH KOTA MEDAN", "size": 9},
                {"point": (72, 100), "text": "Isi halaman kedua", "size": 11},
                {"point": (72, 810), "text": "www.pemkomedan.go.id", "size": 9},
            ],
        ])
        self.addCleanup(temp_dir.cleanup)

        result = extract_pdf_layout(str(pdf_path))
        combined_text = "\n".join(result.pages.values())

        self.assertNotIn("PEMERINTAH KOTA MEDAN", combined_text)
        self.assertNotIn("www.pemkomedan.go.id", combined_text)
        self.assertIn("Isi halaman pertama", combined_text)
        self.assertIn("Isi halaman kedua", combined_text)

    def test_quality_gate_requests_ocr_for_empty_or_fragmented_pages(self):
        self.assertTrue(should_ocr_page("", []))
        self.assertTrue(should_ocr_page("A B C D E F", [{"text": "A"}, {"text": "B"}, {"text": "C"}]))
        self.assertFalse(should_ocr_page("Ini adalah paragraf normal yang cukup panjang untuk text layer PDF.", [{"text": "Ini adalah paragraf normal yang cukup panjang untuk text layer PDF."}]))

    def test_blocks_keep_legacy_shape_without_layout_metadata(self):
        temp_dir, pdf_path = _make_pdf([
            [{"point": (72, 72), "text": "Paragraf dokumen biasa", "size": 11}]
        ])
        self.addCleanup(temp_dir.cleanup)

        result = extract_pdf_layout(str(pdf_path))
        block = result.blocks[0]

        self.assertEqual(
            set(block.keys()),
            {
                "page_number",
                "text",
                "block_type",
                "heading_level",
                "heading_text",
                "heading_path",
                "source_kind",
                "block_order",
                "metadata",
            },
        )
        self.assertEqual(block["metadata"], {})


if __name__ == "__main__":
    unittest.main()
