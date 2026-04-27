import tempfile
import unittest
from pathlib import Path

import fitz

from shared.ocr_utils import extract_blocks_from_file, extract_text_from_file


def _make_pdf(pages):
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(temp_dir.name) / "integration.pdf"
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


class OcrUtilsPdfLayoutIntegrationTests(unittest.TestCase):
    def test_extract_text_from_file_keeps_pdf_page_contract_with_layout_cleanup(self):
        temp_dir, pdf_path = _make_pdf([
            [
                {"point": (72, 30), "text": "PEMERINTAH KOTA MEDAN", "size": 9},
                {"point": (72, 100), "text": "Isi halaman pertama", "size": 11},
            ],
            [
                {"point": (72, 30), "text": "PEMERINTAH KOTA MEDAN", "size": 9},
                {"point": (72, 100), "text": "Isi halaman kedua", "size": 11},
            ],
        ])
        self.addCleanup(temp_dir.cleanup)

        pages = extract_text_from_file(str(pdf_path), return_pages=True)

        self.assertEqual(set(pages.keys()), {1, 2})
        self.assertNotIn("PEMERINTAH KOTA MEDAN", "\n".join(pages.values()))
        self.assertEqual(pages[1], "Isi halaman pertama")
        self.assertEqual(pages[2], "Isi halaman kedua")

    def test_extract_blocks_from_file_uses_layout_heading_and_legacy_shape(self):
        temp_dir, pdf_path = _make_pdf([
            [
                {"point": (72, 72), "text": "TJONG A FIE MANSION", "size": 24},
                {"point": (72, 112), "text": "HERITAGE SEJARAH/ARSITEKTUR", "size": 11},
                {"point": (72, 132), "text": "Jl. Jendral Ahmad Yani No.105 Kesawan Barat", "size": 11},
            ]
        ])
        self.addCleanup(temp_dir.cleanup)

        blocks = extract_blocks_from_file(str(pdf_path))

        self.assertEqual(blocks[0]["block_type"], "heading")
        self.assertEqual(blocks[0]["heading_path"], ["TJONG A FIE MANSION"])
        self.assertEqual(blocks[1]["heading_path"], ["TJONG A FIE MANSION"])
        self.assertEqual(blocks[1]["metadata"], {})
        self.assertEqual(blocks[2]["metadata"], {})


if __name__ == "__main__":
    unittest.main()
