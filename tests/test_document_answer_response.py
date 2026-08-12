import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_search_handler():
    shared_package = types.ModuleType("shared")
    shared_package.__path__ = []
    filtering = types.ModuleType("shared.filtering")
    filtering.ai_pre_filter = None
    filtering.ai_check_relevance = None
    filtering.ai_check_batch_relevance = None
    filtering.ai_extract_answer = None
    filtering.get_relevance_mode = lambda: "single"
    utils = types.ModuleType("shared.utils")
    utils.normalize_text = lambda value: value
    utils.clean_location_terms = lambda value: value
    utils.detect_category = lambda value: None
    service_client = types.ModuleType("orchestrator.service_client")
    service_client.call_service = None
    service_client.call_service_safe = None
    aggregation = types.ModuleType("orchestrator.aggregation")
    aggregation.aggregate_and_sort_candidates = None

    sys.modules.update({
        "shared": shared_package,
        "shared.filtering": filtering,
        "shared.utils": utils,
        "orchestrator.service_client": service_client,
        "orchestrator.aggregation": aggregation,
    })

    source = Path("orchestrator/search_handler.py")
    spec = importlib.util.spec_from_file_location("search_handler_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate(answer_doc="Konteks dokumen"):
    return {
        "source": "document",
        "question": "-",
        "answer_id": None,
        "answer_doc": answer_doc,
        "category_id": None,
        "dense_score": 0.85,
        "overlap_score": 0.0,
        "final_score": 0.85,
        "note": "good_score",
        "document_info": {"filename": "data.pdf", "page_number": 1},
    }


def _payload_shape(value):
    if isinstance(value, dict):
        return {key: _payload_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_payload_shape(value[0])] if value else []
    return type(value).__name__


class DocumentAnswerResponseTests(unittest.TestCase):
    def test_success_payload_keeps_existing_schema_without_extracted_answer(self):
        handler = _load_search_handler()
        response = handler.build_success_response(
            _candidate("Jawaban akhir"),
            "Pertanyaan",
            "Pertanyaan",
            "6281",
            None,
            "relevan",
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        result = response["data"]["similar_questions"][0]
        self.assertEqual(result["answer_doc"], "Jawaban akhir")
        self.assertNotIn("extracted_answer", result)

    def test_invalid_ai_extraction_returns_low_confidence_with_existing_schema(self):
        handler = _load_search_handler()
        candidate = _candidate()

        async def parallel_search_services(*_args):
            return {}, 0.0, 3

        async def check_relevance_with_ai(*_args, **_kwargs):
            return candidate, "candidate relevan", 1, 0.0

        async def ai_extract_answer(*_args, **_kwargs):
            return "Rule 2: If not found, reply exactly: Tidak ditemukan"

        handler.parallel_search_services = parallel_search_services
        handler.aggregate_and_sort_candidates = lambda *_args: [candidate]
        handler.check_relevance_with_ai = check_relevance_with_ai
        handler.ai_extract_answer = ai_extract_answer

        response = asyncio.run(handler.unified_search("hotel di kota medan ada apa aja?"))

        result = response["data"]["similar_questions"][0]
        self.assertEqual(response["status"], "low_confidence")
        self.assertEqual(result["answer_doc"], "Tidak ditemukan")
        self.assertNotIn("extracted_answer", result)

    def test_valid_ai_extraction_becomes_success_answer_doc(self):
        handler = _load_search_handler()
        candidate = _candidate()

        async def parallel_search_services(*_args):
            return {}, 0.0, 3

        async def check_relevance_with_ai(*_args, **_kwargs):
            return candidate, "candidate relevan", 1, 0.0

        async def ai_extract_answer(*_args, **_kwargs):
            return "Berikut daftar hotel yang ditemukan dalam dokumen."

        handler.parallel_search_services = parallel_search_services
        handler.aggregate_and_sort_candidates = lambda *_args: [candidate]
        handler.check_relevance_with_ai = check_relevance_with_ai
        handler.ai_extract_answer = ai_extract_answer

        response = asyncio.run(handler.unified_search("hotel di kota medan ada apa aja?"))

        result = response["data"]["similar_questions"][0]
        self.assertEqual(response["status"], "success")
        self.assertEqual(result["answer_doc"], "Berikut daftar hotel yang ditemukan dalam dokumen.")
        self.assertNotIn("extracted_answer", result)

    def test_empty_ai_extraction_returns_low_confidence(self):
        handler = _load_search_handler()
        candidate = _candidate()

        async def parallel_search_services(*_args):
            return {}, 0.0, 3

        async def check_relevance_with_ai(*_args, **_kwargs):
            return candidate, "candidate relevan", 1, 0.0

        async def ai_extract_answer(*_args, **_kwargs):
            return ""

        handler.parallel_search_services = parallel_search_services
        handler.aggregate_and_sort_candidates = lambda *_args: [candidate]
        handler.check_relevance_with_ai = check_relevance_with_ai
        handler.ai_extract_answer = ai_extract_answer

        response = asyncio.run(handler.unified_search("hotel di kota medan ada apa aja?"))

        self.assertEqual(response["status"], "low_confidence")
        self.assertEqual(response["data"]["similar_questions"][0]["answer_doc"], "Tidak ditemukan")

    def test_batch_selects_candidate_from_valid_rank(self):
        handler = _load_search_handler()
        candidates = [
            _candidate("Kandidat pertama"),
            {**_candidate("Kandidat kedua"), "source": "text", "answer_id": 42},
        ]
        candidates[0]["final_score"] = 0.85
        candidates[1]["final_score"] = 0.80

        async def ai_check_batch_relevance(*_args, **_kwargs):
            return {
                "relevant": True,
                "selected_rank": 2,
                "confidence": 0.91,
                "answer": "",
                "reason": "Kandidat kedua relevan.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": False, "confidence": 0.3, "answer": "", "reason": "Tidak relevan."},
                    {"rank": 2, "relevant": True, "confidence": 0.91, "answer": "", "reason": "Kandidat kedua relevan."},
                ],
            }

        handler.ai_check_batch_relevance = ai_check_batch_relevance
        self.assertTrue(hasattr(handler, "check_relevance_batch_with_ai"))

        selected, reason, checked, _duration = asyncio.run(
            handler.check_relevance_batch_with_ai(candidates, "pertanyaan", max_check=5)
        )

        self.assertIs(selected, candidates[1])
        self.assertEqual(selected["answer_id"], 42)
        self.assertIn("Kandidat kedua relevan.", reason)
        self.assertEqual(selected["judge_confidence"], 0.91)
        self.assertEqual(checked, 2)

    def test_batch_does_not_bypass_judge_for_high_score(self):
        handler = _load_search_handler()
        candidate = _candidate()
        candidate["final_score"] = 0.91
        calls = 0

        async def ai_check_batch_relevance(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return None

        handler.ai_check_batch_relevance = ai_check_batch_relevance
        self.assertTrue(hasattr(handler, "check_relevance_batch_with_ai"))

        selected, _reason, checked, _duration = asyncio.run(
            handler.check_relevance_batch_with_ai([candidate], "pertanyaan", max_check=5)
        )

        self.assertIsNone(selected)
        self.assertEqual(checked, 1)
        self.assertEqual(calls, 1)

    def test_source_priority_text_beats_higher_confidence_document_and_web(self):
        handler = _load_search_handler()
        candidates = [
            {**_candidate("Jawaban web"), "source": "web"},
            {**_candidate("Jawaban dokumen"), "source": "document"},
            {**_candidate("Pertanyaan FAQ"), "source": "text", "answer_id": "faq-1"},
        ]

        async def judge(*_args, **_kwargs):
            return {
                "relevant": True,
                "selected_rank": 1,
                "confidence": 0.99,
                "answer": "Jawaban web",
                "reason": "Web sangat relevan.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": True, "confidence": 0.99, "answer": "Jawaban web", "reason": "Web relevan."},
                    {"rank": 2, "relevant": True, "confidence": 0.96, "answer": "Jawaban dokumen", "reason": "Dokumen relevan."},
                    {"rank": 3, "relevant": True, "confidence": 0.85, "answer": "", "reason": "FAQ semakna."},
                ],
            }

        handler.ai_check_batch_relevance = judge
        selected, _reason, _checked, _duration = asyncio.run(
            handler.check_relevance_batch_with_ai(candidates, "pertanyaan", max_check=5)
        )

        self.assertIs(selected, candidates[2])
        self.assertEqual(selected["source"], "text")
        self.assertEqual(selected["judge_confidence"], 0.85)

    def test_source_priority_document_wins_when_text_below_threshold(self):
        handler = _load_search_handler()
        candidates = [
            {**_candidate("Pertanyaan FAQ"), "source": "text", "answer_id": "faq-1"},
            {**_candidate("Jawaban web"), "source": "web"},
            {**_candidate("Jawaban dokumen"), "source": "document"},
        ]

        async def judge(*_args, **_kwargs):
            return {
                "relevant": True,
                "selected_rank": 2,
                "confidence": 0.99,
                "answer": "Jawaban web",
                "reason": "Pilihan LLM sengaja melanggar prioritas.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": True, "confidence": 0.84, "answer": "", "reason": "FAQ di bawah threshold."},
                    {"rank": 2, "relevant": True, "confidence": 0.99, "answer": "Jawaban web", "reason": "Web relevan."},
                    {"rank": 3, "relevant": True, "confidence": 0.86, "answer": "Jawaban dokumen", "reason": "Dokumen relevan."},
                ],
            }

        handler.ai_check_batch_relevance = judge
        selected, _reason, _checked, _duration = asyncio.run(
            handler.check_relevance_batch_with_ai(candidates, "pertanyaan", max_check=5)
        )

        self.assertIs(selected, candidates[2])
        self.assertEqual(selected["source"], "document")
        self.assertEqual(selected["answer_doc"], "Jawaban dokumen")

    def test_invalid_batch_response_fails_closed(self):
        handler = _load_search_handler()
        candidate = _candidate()
        candidate["final_score"] = 0.85
        fallback_calls = 0

        async def ai_check_batch_relevance(*_args, **_kwargs):
            return None

        async def check_relevance_with_ai(*_args, **_kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            return candidate, "fallback single", 1, 0.01

        handler.ai_check_batch_relevance = ai_check_batch_relevance
        handler.check_relevance_with_ai = check_relevance_with_ai
        self.assertTrue(hasattr(handler, "check_relevance_batch_with_ai"))

        selected, reason, checked, _duration = asyncio.run(
            handler.check_relevance_batch_with_ai([candidate], "pertanyaan", max_check=5)
        )

        self.assertIsNone(selected)
        self.assertIn("Combined AI judge gagal", reason)
        self.assertEqual(checked, 1)
        self.assertEqual(fallback_calls, 0)

    def test_relevance_dispatcher_uses_configured_mode(self):
        handler = _load_search_handler()
        candidate = _candidate()
        calls = []

        async def batch(*_args, **_kwargs):
            calls.append("batch")
            return candidate, "batch", 1, 0.01

        async def single(*_args, **_kwargs):
            calls.append("single")
            return candidate, "single", 1, 0.01

        handler.get_relevance_mode = lambda: "batch"
        handler.check_relevance_batch_with_ai = batch
        handler.check_relevance_with_ai = single
        self.assertTrue(hasattr(handler, "check_relevance_by_mode"))

        selected, reason, _checked, _duration = asyncio.run(
            handler.check_relevance_by_mode([candidate], "pertanyaan", max_check=5)
        )

        self.assertIs(selected, candidate)
        self.assertEqual(reason, "batch")
        self.assertEqual(calls, ["batch"])

    def test_relevance_dispatcher_keeps_single_mode_flow(self):
        handler = _load_search_handler()
        candidate = _candidate()
        calls = []

        async def batch(*_args, **_kwargs):
            calls.append("batch")
            return candidate, "batch", 1, 0.01

        async def single(*_args, **_kwargs):
            calls.append("single")
            return candidate, "single", 1, 0.01

        handler.get_relevance_mode = lambda: "single"
        handler.check_relevance_batch_with_ai = batch
        handler.check_relevance_with_ai = single

        selected, reason, _checked, _duration = asyncio.run(
            handler.check_relevance_by_mode([candidate], "pertanyaan", max_check=5)
        )

        self.assertIs(selected, candidate)
        self.assertEqual(reason, "single")
        self.assertEqual(calls, ["single"])

    def test_batch_no_relevant_candidate_returns_low_confidence(self):
        handler = _load_search_handler()
        candidate = _candidate()
        candidate["final_score"] = 0.85

        async def parallel_search_services(*_args):
            return {}, 0.0, 3

        async def ai_check_batch_relevance(*_args, **_kwargs):
            return {
                "relevant": False,
                "selected_rank": None,
                "confidence": 0.0,
                "answer": "",
                "reason": "Tidak ada kandidat yang menjawab.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": False, "confidence": 0.2, "answer": "", "reason": "Tidak menjawab."},
                ],
            }

        handler.parallel_search_services = parallel_search_services
        handler.aggregate_and_sort_candidates = lambda *_args: [candidate]
        handler.get_relevance_mode = lambda: "batch"
        handler.ai_check_batch_relevance = ai_check_batch_relevance

        response = asyncio.run(handler.unified_search("hotel di kota medan ada apa aja?"))

        self.assertEqual(response["status"], "low_confidence")
        self.assertEqual(
            response["data"]["similar_questions"][0]["answer_doc"],
            "Konteks dokumen",
        )
        self.assertNotIn(
            "extracted_answer",
            response["data"]["similar_questions"][0],
        )

    def test_batch_selected_document_uses_combined_answer_without_second_call(self):
        handler = _load_search_handler()
        candidate = _candidate("Context parent dokumen")
        candidate["final_score"] = 0.85

        async def parallel_search_services(*_args):
            return {}, 0.0, 3

        async def ai_check_batch_relevance(*_args, **_kwargs):
            return {
                "relevant": True,
                "selected_rank": 1,
                "confidence": 0.93,
                "answer": "Jawaban akhir dari combined judge.",
                "reason": "Dokumen memuat jawaban.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": True, "confidence": 0.93, "answer": "Jawaban akhir dari combined judge.", "reason": "Dokumen memuat jawaban."},
                ],
            }

        async def ai_extract_answer(*_args, **_kwargs):
            self.fail("Second extraction call must not run in combined mode")

        handler.parallel_search_services = parallel_search_services
        handler.aggregate_and_sort_candidates = lambda *_args: [candidate]
        handler.get_relevance_mode = lambda: "batch"
        handler.ai_check_batch_relevance = ai_check_batch_relevance
        handler.ai_extract_answer = ai_extract_answer

        response = asyncio.run(handler.unified_search("informasi dokumen"))

        result = response["data"]["similar_questions"][0]
        self.assertEqual(response["status"], "success")
        self.assertEqual(result["answer_doc"], "Jawaban akhir dari combined judge.")
        self.assertNotIn("extracted_answer", result)

    def test_single_and_batch_success_payload_shapes_are_identical(self):
        async def run_search(mode):
            handler = _load_search_handler()
            candidate = {
                **_candidate(""),
                "source": "text",
                "answer_id": 42,
                "question": "Apa saja hotel di Kota Medan?",
                "final_score": 0.85,
                "content_for_check": "Apa saja hotel di Kota Medan?",
            }

            async def parallel_search_services(*_args):
                return {}, 0.0, 3

            async def ai_check_relevance(*_args, **_kwargs):
                return {"relevant": True, "reason": "single relevant"}

            async def ai_check_batch_relevance(*_args, **_kwargs):
                return {
                    "relevant": True,
                    "selected_rank": 1,
                    "confidence": 0.9,
                    "answer": "",
                    "reason": "batch relevant",
                    "reformulated_question": "",
                    "candidate_assessments": [
                        {"rank": 1, "relevant": True, "confidence": 0.9, "answer": "", "reason": "batch relevant"},
                    ],
                }

            handler.parallel_search_services = parallel_search_services
            handler.aggregate_and_sort_candidates = lambda *_args: [candidate]
            handler.get_relevance_mode = lambda: mode
            handler.ai_check_relevance = ai_check_relevance
            handler.ai_check_batch_relevance = ai_check_batch_relevance
            return await handler.unified_search("hotel di kota medan ada apa aja?")

        single_response = asyncio.run(run_search("single"))
        batch_response = asyncio.run(run_search("batch"))

        self.assertEqual(_payload_shape(single_response), _payload_shape(batch_response))
        self.assertEqual(
            batch_response["data"]["similar_questions"][0]["answer_id"],
            42,
        )
        self.assertEqual(
            batch_response["data"]["similar_questions"][0]["answer_doc"],
            "",
        )


if __name__ == "__main__":
    unittest.main()
