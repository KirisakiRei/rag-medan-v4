import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_search_module():
    shared_package = types.ModuleType("shared")
    shared_package.__path__ = []
    shared_summarizer = types.ModuleType("shared.summarizer_utils")
    shared_summarizer.summarize_text = lambda text, max_sentences=5: text
    shared_utils = types.ModuleType("shared.utils")
    shared_utils.format_for_display = lambda text: text
    shared_utils.encode_texts = None

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = object
    qdrant_http = types.ModuleType("qdrant_client.http")
    qdrant_http.models = types.SimpleNamespace()

    sys.modules.update({
        "shared": shared_package,
        "shared.summarizer_utils": shared_summarizer,
        "shared.utils": shared_utils,
        "sentence_transformers": sentence_transformers,
        "qdrant_client": qdrant_client,
        "qdrant_client.http": qdrant_http,
    })

    source = Path("services/rag_document/search.py")
    spec = importlib.util.spec_from_file_location("document_search_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DocumentContextExpansionTests(unittest.TestCase):
    def test_parent_is_preferred_over_sibling_window_when_available(self):
        search = _load_search_module()
        search.config.RETRIEVAL_CONTEXT_EXPANSION = True

        async def related_payloads(_point_ids):
            return {
                "parent-1": {"text": "Konteks parent lengkap."},
                "prev-1": {"text": "Sibling sebelum."},
                "next-1": {"text": "Sibling sesudah."},
            }

        search._retrieve_payloads_by_ids = related_payloads
        result = asyncio.run(search._expand_document_context({
            "text": "Child yang cocok tetapi terlalu pendek untuk menjadi jawaban mandiri.",
            "parent_chunk_id": "parent-1",
            "window_prev_id": "prev-1",
            "window_next_id": "next-1",
        }))

        self.assertEqual(result, "Konteks parent lengkap.")

    def test_sibling_window_is_used_when_parent_is_missing(self):
        search = _load_search_module()
        search.config.RETRIEVAL_CONTEXT_EXPANSION = True

        async def related_payloads(_point_ids):
            return {
                "prev-1": {"text": "Sibling sebelum.", "parent_chunk_id": "parent-missing"},
                "next-1": {"text": "Sibling sesudah.", "parent_chunk_id": "parent-missing"},
            }

        search._retrieve_payloads_by_ids = related_payloads
        result = asyncio.run(search._expand_document_context({
            "text": "Child terpilih.",
            "parent_chunk_id": "parent-missing",
            "window_prev_id": "prev-1",
            "window_next_id": "next-1",
        }))

        self.assertEqual(result, "Sibling sebelum.\n\nChild terpilih.\n\nSibling sesudah.")

    def test_foreign_parent_siblings_are_ignored_when_parent_is_missing(self):
        search = _load_search_module()
        search.config.RETRIEVAL_CONTEXT_EXPANSION = True

        async def related_payloads(_point_ids):
            return {
                "prev-1": {"text": "Section lain.", "parent_chunk_id": "parent-other"},
                "next-1": {"text": "Section lain lagi.", "parent_chunk_id": "parent-other"},
            }

        search._retrieve_payloads_by_ids = related_payloads
        result = asyncio.run(search._expand_document_context({
            "text": "Child terpilih.",
            "parent_chunk_id": "parent-missing",
            "window_prev_id": "prev-1",
            "window_next_id": "next-1",
        }))

        self.assertEqual(result, "Child terpilih.")

    def test_child_hits_are_deduplicated_by_parent_before_ranking(self):
        search = _load_search_module()

        self.assertTrue(hasattr(search, "_deduplicate_document_hits"))
        unique_hits = search._deduplicate_document_hits([
            {"payload": {"parent_chunk_id": "parent-1"}, "score": 0.91},
            {"payload": {"parent_chunk_id": "parent-1"}, "score": 0.82},
            {"payload": {"parent_chunk_id": "parent-2"}, "score": 0.80},
        ])

        self.assertEqual(len(unique_hits), 2)
        self.assertEqual(unique_hits[0]["score"], 0.91)
        self.assertEqual(unique_hits[1]["score"], 0.80)


if __name__ == "__main__":
    unittest.main()
