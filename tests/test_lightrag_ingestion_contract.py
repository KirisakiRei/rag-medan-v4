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
                    "/documents/paginated": {"get": {}},
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
                    "/documents/paginated": {"get": {}},
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

        insert.assert_awaited_once_with(text="body", file_source="document:d-1")
        wait.assert_awaited_once_with("insert-1", "document:d-1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["lightrag_document_id"], "doc-real")

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
        }

        with patch.object(text_sync, "sync_lightrag_text", confirmed):
            result = asyncio.run(text_sync.sync_data("add", payload))

        self.assertEqual(result["status"], "success")
        kwargs = confirmed.await_args.kwargs
        self.assertEqual(kwargs["source_id"], "q-1")
        self.assertEqual(kwargs["question"], payload["question"])
        self.assertIsNone(kwargs["answer"])
        self.assertTrue(kwargs["content_hash"])


if __name__ == "__main__":
    unittest.main()
