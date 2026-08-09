import importlib.util
import sys
import unittest
from pathlib import Path


def _load_chunker_module():
    source = Path("services/rag_document/chunker.py")
    spec = importlib.util.spec_from_file_location("document_chunker_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DocumentParentChildContextTests(unittest.TestCase):
    def test_sibling_window_does_not_cross_parent_sections(self):
        chunker = _load_chunker_module()
        chunks = chunker.structure_chunk_document(
            [
                {
                    "text": "Informasi hotel pada section pertama.",
                    "heading_text": "Hotel A",
                    "heading_path": ["Hotel A"],
                    "block_order": 0,
                },
                {
                    "text": "Informasi museum pada section kedua.",
                    "heading_text": "Museum B",
                    "heading_path": ["Museum B"],
                    "block_order": 1,
                },
            ],
            child_chunk_size=80,
            parent_chunk_size=200,
            overlap=0,
            enable_semantic_merge=False,
            similarity_threshold=1.0,
        )

        children = [chunk for chunk in chunks if chunk.chunk_level == "child"]

        self.assertEqual(len(children), 2)
        self.assertNotEqual(children[0].parent_chunk_id, children[1].parent_chunk_id)
        self.assertIsNone(children[0].window_next_id)
        self.assertIsNone(children[1].window_prev_id)


if __name__ == "__main__":
    unittest.main()
