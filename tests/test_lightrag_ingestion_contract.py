import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from orchestrator.search_handler import _infer_source_type
from services.lightrag_adapter.client import LightRAGClient
from services.lightrag_adapter.references import map_lightrag_context_to_canonical
from services.lightrag_adapter import sync as adapter_sync
from services.rag_text import sync as text_sync
from shared.lightrag_sync import _validate_adapter_response


class LightRAGClientContractTests(unittest.TestCase):
    def test_insert_text_uses_file_source(self):
        client = LightRAGClient()
        client._request = AsyncMock(return_value={"status": "success", "track_id": "t-1"})

        asyncio.run(client.insert_text("question", "text:q-1"))

        client._request.assert_awaited_once_with(
            "POST",
            "/documents/text",
            json_data={"text": "question", "file_source": "text:q-1"},
        )

    def test_delete_uses_supported_endpoint_and_body(self):
        client = LightRAGClient()
        client._request = AsyncMock(return_value={"status": "deletion_started"})

        asyncio.run(client.delete_document("doc-abc"))

        client._request.assert_awaited_once_with(
            "DELETE",
            "/documents/delete_document",
            json_data={
                "doc_ids": ["doc-abc"],
                "delete_file": False,
                "delete_llm_cache": False,
            },
        )

    def test_delete_does_not_fallback_to_legacy(self):
        from services.lightrag_adapter.errors import LightRAGSearchError

        client = LightRAGClient()
        client._request = AsyncMock(side_effect=LightRAGSearchError(
            'LightRAG returned 405: {"detail":"Method Not Allowed"}'
        ))

        with self.assertRaises(LightRAGSearchError):
            asyncio.run(client.delete_document("doc-abc"))

        self.assertEqual(client._request.await_count, 1)

    def test_paginated_listing_uses_modern_post_contract(self):
        client = LightRAGClient()
        client._request = AsyncMock(return_value={"documents": []})

        asyncio.run(client.get_documents_paginated(page=2, page_size=100))

        client._request.assert_awaited_once_with(
            "POST",
            "/documents/paginated",
            json_data={
                "page": 2,
                "page_size": 100,
                "sort_field": "updated_at",
                "sort_direction": "desc",
            },
        )

    def test_startup_contract_requires_modern_delete_endpoint(self):
        client = LightRAGClient()
        request = httpx.Request("GET", "http://lightrag/openapi.json")
        client._client = AsyncMock()
        client._client.get = AsyncMock(return_value=httpx.Response(
            200,
            request=request,
            json={
                "info": {"version": "legacy"},
                "paths": {
                    "/documents/text": {"post": {}},
                    "/documents/track_status/{track_id}": {"get": {}},
                    "/documents/paginated": {"post": {}},
                    "/query": {"post": {}},
                },
            },
        ))

        with self.assertRaisesRegex(RuntimeError, "DELETE /documents/delete_document"):
            asyncio.run(client.verify_modern_api_contract())

    def test_startup_contract_accepts_required_modern_endpoints(self):
        client = LightRAGClient()
        request = httpx.Request("GET", "http://lightrag/openapi.json")
        client._client = AsyncMock()
        client._client.get = AsyncMock(return_value=httpx.Response(
            200,
            request=request,
            json={
                "info": {"version": "0329"},
                "paths": {
                    "/documents/text": {"post": {}},
                    "/documents/delete_document": {"delete": {}},
                    "/documents/track_status/{track_id}": {"get": {}},
                    "/documents/paginated": {"post": {}},
                    "/query": {"post": {}},
                },
            },
        ))

        result = asyncio.run(client.verify_modern_api_contract())

        self.assertTrue(result["compatible"])
        self.assertEqual(result["api_version"], "0329")


class ConfirmedIndexTests(unittest.TestCase):
    def test_index_waits_for_processed_track(self):
        insert = AsyncMock(return_value={"status": "success", "track_id": "insert-1"})
        wait = AsyncMock(return_value={
            "id": "doc-real",
            "status": "PROCESSED",
            "file_path": "document:d-1",
        })
        find = AsyncMock(return_value=None)

        with patch.object(adapter_sync.lightrag_client, "insert_text", insert), \
             patch.object(adapter_sync, "_wait_until_indexed", wait), \
             patch.object(adapter_sync, "_find_document_by_source", find):
            result = asyncio.run(adapter_sync._index_document(
                "kb:medan-main:document:d-1",
                "body",
                "document",
                "d-1",
            ))

        insert.assert_awaited_once_with(
            text="Source-ID: document:d-1\nbody",
            file_source="document:d-1",
        )
        wait.assert_awaited_once_with("insert-1", "document:d-1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["lightrag_document_id"], "doc-real")

    def test_identical_business_content_is_unique_per_source(self):
        insert = AsyncMock(return_value={"status": "success", "track_id": "insert-1"})
        wait = AsyncMock(return_value={
            "id": "doc-real",
            "status": "PROCESSED",
            "file_path": "text:q-2",
        })

        with patch.object(adapter_sync.lightrag_client, "insert_text", insert), \
             patch.object(adapter_sync, "_wait_until_indexed", wait), \
             patch.object(
                 adapter_sync,
                 "_find_document_by_source",
                 AsyncMock(return_value=None),
             ):
            result = asyncio.run(adapter_sync._index_document(
                "kb:medan-main:text:q-2",
                "Title: Pertanyaan sama\n\nQuestion:\nPertanyaan sama",
                "text",
                "q-2",
            ))

        indexed_text = insert.await_args.kwargs["text"]
        self.assertTrue(indexed_text.startswith("Source-ID: text:q-2\n"))
        self.assertNotIn("Answer:", indexed_text)
        self.assertEqual(result["source_id"], "q-2")

    def test_missing_track_id_fails_closed(self):
        with patch.object(
            adapter_sync.lightrag_client,
            "insert_text",
            AsyncMock(return_value={"status": "success"}),
        ), patch.object(
            adapter_sync,
            "_find_document_by_source",
            AsyncMock(return_value=None),
        ):
            result = asyncio.run(adapter_sync._index_document(
                "logical", "body", "text", "q-1"
            ))

        self.assertEqual(result["status"], "error")

    def test_adapter_body_error_is_not_success(self):
        request = httpx.Request("POST", "http://adapter/internal/sync/text")
        response = httpx.Response(
            200,
            request=request,
            json={"status": "error", "message": "index failed"},
        )

        with self.assertRaises(RuntimeError):
            _validate_adapter_response(response, "text:q-1")


class SourceMappingTests(unittest.TestCase):
    def test_text_answer_ids_and_category_recovered_from_ingestion_header(self):
        contexts = map_lightrag_context_to_canonical([{
            "doc_id": "text:q-1",
            "content": (
                "Source-ID: text:q-1\n"
                "Category-ID: layanan\n"
                "Answer-ID: a-1,a-2\n"
                "Title: Bagaimana mengurus KTP?\n\n"
                "Question:\nBagaimana mengurus KTP?"
            ),
            "reference_id": "1",
        }])

        self.assertEqual(contexts[0]["source_type"], "text")
        self.assertEqual(contexts[0]["answer_id"], ["a-1", "a-2"])
        self.assertEqual(contexts[0]["category_id"], "layanan")

    def test_document_metadata_is_recovered_from_ingestion_header(self):
        contexts = map_lightrag_context_to_canonical([{
            "doc_id": "document:d-1",
            "content": (
                "Title: Pedoman Statistik\n"
                "Organization: opd-1\n"
                "File: pedoman.pdf\n\n"
                "Isi dengan tautan https://portal.medan.go.id"
            ),
            "reference_id": "7",
        }])

        self.assertEqual(contexts[0]["source_type"], "document")
        self.assertEqual(contexts[0]["title"], "Pedoman Statistik")
        self.assertEqual(contexts[0]["source_uri"], "document://d-1/pedoman.pdf")
        self.assertEqual(contexts[0]["reference_id"], "7")

    def test_document_body_url_does_not_infer_web(self):
        inferred = _infer_source_type({
            "source_type": "unknown",
            "source_id": "legacy.pdf",
            "title": "legacy.pdf",
            "content": "Baca juga https://portal.medan.go.id",
        })
        self.assertEqual(inferred, "document")

    def test_invalid_full_descriptor_remains_unknown(self):
        contexts = map_lightrag_context_to_canonical([{
            "doc_id": "kb:medan-main:website:123",
            "content": "body",
        }])
        self.assertEqual(contexts[0]["source_type"], "unknown")


class TextSyncTests(unittest.TestCase):
    def test_add_waits_for_confirmed_question_only_ingestion(self):
        confirmed = AsyncMock(return_value={"status": "success"})
        payload = {
            "question_rag_id": "q-1",
            "question": "Bagaimana mengurus KTP?",
            "category_id": "layanan",
            "answer_id": ["a-1"],
        }

        with patch.object(text_sync, "sync_lightrag_text", confirmed):
            result = asyncio.run(text_sync.sync_data("add", payload))

        self.assertEqual(result["status"], "success")
        kwargs = confirmed.await_args.kwargs
        self.assertEqual(kwargs["source_id"], "q-1")
        self.assertEqual(kwargs["question"], payload["question"])
        self.assertEqual(kwargs["category"], "layanan")
        self.assertEqual(kwargs["answer_id"], ["a-1"])
        self.assertIsNone(kwargs["answer"])
        self.assertTrue(kwargs["content_hash"])


class AdapterTextSyncMetadataTests(unittest.TestCase):
    def test_text_sync_sends_category_and_answer_id_metadata(self):
        adapter_sync.lightrag_client.insert_text = AsyncMock(
            return_value={"status": "success", "track_id": "track-1"}
        )
        adapter_sync._find_document_by_source = AsyncMock(return_value=None)
        adapter_sync._wait_until_indexed = AsyncMock(return_value={"id": "doc-1"})

        result = asyncio.run(adapter_sync.sync_text(
            source_id="q-1",
            knowledge_base_id="medan-main",
            title="Bagaimana mengurus KTP?",
            content="",
            content_hash="hash-1",
            category="layanan",
            answer_id=["a-1"],
            question="Bagaimana mengurus KTP?",
            answer=None,
        ))

        self.assertEqual(result["status"], "success")
        adapter_sync.lightrag_client.insert_text.assert_awaited_once()
        kwargs = adapter_sync.lightrag_client.insert_text.await_args.kwargs
        self.assertEqual(kwargs["file_source"], "text:q-1")
        self.assertEqual(kwargs["metadata"], {
            "category_id": "layanan",
            "answer_id": ["a-1"],
        })


class UsulanIntegrationTests(unittest.TestCase):
    def test_parse_document_id_accepts_usulan(self):
        from services.lightrag_adapter.source_mapper import make_document_id, parse_document_id
        doc_id = make_document_id("usulan-main", "usulan", "u-1")
        kb_id, source_type, source_id = parse_document_id(doc_id)
        self.assertEqual(kb_id, "usulan-main")
        self.assertEqual(source_type, "usulan")
        self.assertEqual(source_id, "u-1")

    def test_usulan_context_metadata_recovered_from_ingestion_header(self):
        contexts = map_lightrag_context_to_canonical([{
            "doc_id": "usulan:u-1",
            "content": (
                "Source-ID: usulan:u-1\n"
                "Title: Perbaikan Jalan Rusak\n"
                "Organization: opd-1\n"
                "Request-ID: req-77\n"
                "Request-Name: Perbaikan jalan di Kec. Medan Baru\n\n"
                "Question:\nPerbaikan Jalan Rusak"
            ),
            "reference_id": "3",
        }])

        self.assertEqual(contexts[0]["source_type"], "usulan")
        self.assertEqual(contexts[0]["source_id"], "u-1")
        self.assertEqual(contexts[0]["request_id"], "req-77")
        self.assertEqual(contexts[0]["request_name"], "Perbaikan jalan di Kec. Medan Baru")
        self.assertEqual(contexts[0]["organization_id"], "opd-1")
        self.assertEqual(contexts[0]["source_uri"], "usulan://u-1")

    def test_adapter_sync_usulan_sends_question_only_content_and_metadata(self):
        adapter_sync.lightrag_client.insert_text = AsyncMock(
            return_value={"status": "success", "track_id": "track-1"}
        )
        adapter_sync._find_document_by_source = AsyncMock(return_value=None)
        adapter_sync._wait_until_indexed = AsyncMock(return_value={"id": "doc-1"})

        result = asyncio.run(adapter_sync.sync_usulan(
            source_id="u-1",
            knowledge_base_id="usulan-main",
            title="Perbaikan Jalan Rusak",
            content="",
            content_hash="hash-u",
            organization_id="opd-1",
            request_id="req-77",
            request_name="Perbaikan jalan di Kec. Medan Baru",
            question="Perbaikan Jalan Rusak",
        ))

        self.assertEqual(result["status"], "success")
        adapter_sync.lightrag_client.insert_text.assert_awaited_once()
        kwargs = adapter_sync.lightrag_client.insert_text.await_args.kwargs
        self.assertEqual(kwargs["file_source"], "usulan:u-1")
        self.assertEqual(kwargs["metadata"], {
            "category_id": "opd-1",
            "request_id": "req-77",
            "request_name": "Perbaikan jalan di Kec. Medan Baru",
        })
        # Question-only: tidak ada Answer di konten yang dikirim
        self.assertNotIn("Answer:", kwargs["text"])
        self.assertIn("Question:", kwargs["text"])
        self.assertIn("Request-ID: req-77", kwargs["text"])


if __name__ == "__main__":
    unittest.main()
