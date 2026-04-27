import unittest
from pathlib import Path


class DocumentWorkerPdfRoutingTests(unittest.TestCase):
    def test_pdf_chunking_uses_layout_aware_block_extractor(self):
        source = Path("services/rag_document/worker.py").read_text(encoding="utf-8")

        self.assertIn('if file_ext == ".pdf":', source)
        pdf_branch_start = source.index('if file_ext == ".pdf":')
        image_branch_start = source.index('if file_ext in [".txt", ".jpg", ".jpeg", ".png"]:', pdf_branch_start)
        pdf_branch = source[pdf_branch_start:image_branch_start]

        self.assertIn("extract_blocks_from_file(", pdf_branch)
        self.assertNotIn("build_blocks_from_extracted_pages(", pdf_branch)


if __name__ == "__main__":
    unittest.main()
