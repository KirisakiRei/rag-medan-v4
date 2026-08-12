import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


def _load_filtering_module():
    shared_package = types.ModuleType("shared")
    shared_package.__path__ = []
    shared_db = types.ModuleType("shared.db")
    shared_db.get_variable = lambda _key: None
    shared_utils = types.ModuleType("shared.utils")
    shared_utils.hard_filter_local = lambda question: {
        "valid": True,
        "clean_question": question,
        "reason": "",
    }
    shared_prompts = types.ModuleType("shared.prompts")
    shared_prompts.PROMPT_PRE_FILTER_RAG = "PRE FILTER"
    shared_prompts.PROMPT_PRE_FILTER_USULAN = "PRE FILTER USULAN"
    shared_prompts.PROMPT_RELEVANCE_RAG = "SINGLE RELEVANCE"
    shared_prompts.PROMPT_RELEVANCE_USULAN = "USULAN RELEVANCE"
    shared_prompts.PROMPT_RERANK = "RERANK"
    shared_prompts.PROMPT_AI_BATCH_RELEVANCE = "DEFAULT BATCH RELEVANCE"

    sys.modules.update({
        "shared": shared_package,
        "shared.db": shared_db,
        "shared.utils": shared_utils,
        "shared.prompts": shared_prompts,
    })

    source = Path("shared/filtering.py")
    spec = importlib.util.spec_from_file_location("batch_filtering_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidates():
    return [
        {
            "source": "text",
            "final_score": 0.84,
            "content_for_check": "Apa saja hotel di Kota Medan?",
        },
        {
            "source": "document",
            "final_score": 0.80,
            "content_for_check": "Daftar hotel: Hotel A dan Hotel B.",
        },
    ]


def _load_default_batch_prompt():
    source = Path("shared/prompts.py")
    spec = importlib.util.spec_from_file_location("batch_prompts_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PROMPT_AI_BATCH_RELEVANCE


class BatchRelevanceFilteringTests(unittest.TestCase):
    def test_default_prompt_uses_valid_json_examples(self):
        prompt = _load_default_batch_prompt()

        self.assertNotIn('"selected_rank": 1 atau null', prompt)
        self.assertIn('"candidate_assessments"', prompt)
        self.assertIn('"selected_rank":1', prompt)
        self.assertIn('"selected_rank":null', prompt)
        self.assertIn('text > document > web', prompt)
        self.assertIn('confidence minimal 0.85', prompt)

    def test_database_mode_overrides_config_default(self):
        filtering = _load_filtering_module()
        self.assertTrue(hasattr(filtering, "get_relevance_mode"))
        filtering.config.RELEVANCE_MODE = "single"
        filtering.get_cached_variable = lambda key: "batch" if key == "relevance_mode" else None

        self.assertEqual(filtering.get_relevance_mode(), "batch")

    def test_invalid_mode_falls_back_to_single(self):
        filtering = _load_filtering_module()
        self.assertTrue(hasattr(filtering, "get_relevance_mode"))
        filtering.config.RELEVANCE_MODE = "batch"
        filtering.get_cached_variable = lambda key: "invalid-mode" if key == "relevance_mode" else None

        self.assertEqual(filtering.get_relevance_mode(), "single")

    def test_config_mode_is_used_when_database_variable_is_empty(self):
        filtering = _load_filtering_module()
        filtering.config.RELEVANCE_MODE = "batch"
        filtering.get_cached_variable = lambda _key: None

        self.assertEqual(filtering.get_relevance_mode(), "batch")

    def test_config_mode_is_used_when_database_variable_is_whitespace(self):
        filtering = _load_filtering_module()
        filtering.config.RELEVANCE_MODE = "batch"
        filtering.get_cached_variable = lambda key: "   " if key == "relevance_mode" else None

        self.assertEqual(filtering.get_relevance_mode(), "batch")

    def test_batch_uses_prompt_override_and_serialized_candidates(self):
        filtering = _load_filtering_module()
        self.assertTrue(hasattr(filtering, "ai_check_batch_relevance"))
        captured = {}

        async def call_filter_llm(**kwargs):
            captured.update(kwargs)
            return json.dumps({
                "relevant": True,
                "selected_rank": 2,
                "confidence": 0.91,
                "answer": "Hotel A dan Hotel B.",
                "reason": "Kandidat kedua memuat daftar hotel.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": False, "confidence": 0.3, "answer": "", "reason": "Tidak menjawab."},
                    {"rank": 2, "relevant": True, "confidence": 0.91, "answer": "Hotel A dan Hotel B.", "reason": "Memuat daftar."},
                ],
            })

        valid_override = "CUSTOM candidate_assessments text > document > web threshold 0.85"
        filtering.get_cached_variable = (
            lambda key: valid_override if key == "prompt_ai_combined_judge" else None
        )
        filtering.call_filter_llm = call_filter_llm

        result = asyncio.run(filtering.ai_check_batch_relevance(
            "hotel di medan apa saja?",
            _candidates(),
        ))

        self.assertEqual(captured["system_prompt"], valid_override)
        self.assertIn('"rank": 1', captured["user_message"])
        self.assertIn('"source": "text"', captured["user_message"])
        self.assertIn('"content": "Daftar hotel: Hotel A dan Hotel B."', captured["user_message"])
        self.assertEqual(result["selected_rank"], 2)

    def test_batch_rejects_invalid_rank_values(self):
        filtering = _load_filtering_module()
        self.assertTrue(hasattr(filtering, "ai_check_batch_relevance"))

        for invalid_rank in [0, -1, 3, "2", True]:
            with self.subTest(selected_rank=invalid_rank):
                async def call_filter_llm(**_kwargs):
                    return json.dumps({
                        "relevant": True,
                        "selected_rank": invalid_rank,
                        "confidence": 0.9,
                        "answer": "Jawaban",
                        "reason": "invalid",
                        "reformulated_question": "",
                    })

                filtering.call_filter_llm = call_filter_llm
                result = asyncio.run(filtering.ai_check_batch_relevance(
                    "hotel di medan apa saja?",
                    _candidates(),
                ))
                self.assertIsNone(result)

    def test_batch_accepts_explicit_no_relevant_candidate(self):
        filtering = _load_filtering_module()
        self.assertTrue(hasattr(filtering, "ai_check_batch_relevance"))

        async def call_filter_llm(**_kwargs):
            return json.dumps({
                "relevant": False,
                "selected_rank": None,
                "confidence": 0.0,
                "answer": "",
                "reason": "Tidak ada kandidat yang menjawab.",
                "reformulated_question": "Daftar hotel apa saja di Medan?",
                "candidate_assessments": [
                    {"rank": 1, "relevant": False, "confidence": 0.2, "answer": "", "reason": "Tidak menjawab."},
                    {"rank": 2, "relevant": False, "confidence": 0.2, "answer": "", "reason": "Tidak menjawab."},
                ],
            })

        filtering.call_filter_llm = call_filter_llm
        result = asyncio.run(filtering.ai_check_batch_relevance(
            "hotel di medan apa saja?",
            _candidates(),
        ))

        self.assertFalse(result["relevant"])
        self.assertIsNone(result["selected_rank"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["answer"], "")

    def test_batch_rejects_invalid_confidence_or_missing_answer(self):
        filtering = _load_filtering_module()

        invalid_payloads = [
            {"confidence": -0.1, "answer": "Jawaban"},
            {"confidence": 1.1, "answer": "Jawaban"},
            {"confidence": "0.9", "answer": "Jawaban"},
            {"confidence": 0.9, "answer": ""},
        ]
        for invalid_values in invalid_payloads:
            with self.subTest(invalid_values=invalid_values):
                async def call_filter_llm(**_kwargs):
                    return json.dumps({
                        "relevant": True,
                        "selected_rank": 2,
                        "reason": "Kandidat relevan.",
                        "reformulated_question": "",
                        **invalid_values,
                    })

                filtering.call_filter_llm = call_filter_llm
                result = asyncio.run(filtering.ai_check_batch_relevance(
                    "hotel di medan apa saja?",
                    _candidates(),
                ))
                self.assertIsNone(result)

    def test_batch_uses_default_prompt_when_override_is_empty(self):
        filtering = _load_filtering_module()
        captured = {}

        async def call_filter_llm(**kwargs):
            captured.update(kwargs)
            return json.dumps({
                "relevant": False,
                "selected_rank": None,
                "confidence": 0.0,
                "answer": "",
                "reason": "Tidak ada kandidat yang menjawab.",
                "reformulated_question": "",
                "candidate_assessments": [
                    {"rank": 1, "relevant": False, "confidence": 0.2, "answer": "", "reason": "Tidak menjawab."},
                    {"rank": 2, "relevant": False, "confidence": 0.2, "answer": "", "reason": "Tidak menjawab."},
                ],
            })

        filtering.get_cached_variable = lambda _key: None
        filtering.call_filter_llm = call_filter_llm
        asyncio.run(filtering.ai_check_batch_relevance(
            "hotel di medan apa saja?",
            _candidates(),
        ))

        self.assertEqual(captured["system_prompt"], "DEFAULT BATCH RELEVANCE")

    def test_batch_returns_none_for_empty_or_malformed_response(self):
        filtering = _load_filtering_module()

        for llm_response in [None, "", "bukan json", '{"relevant": "true"}']:
            with self.subTest(llm_response=llm_response):
                async def call_filter_llm(**_kwargs):
                    return llm_response

                filtering.call_filter_llm = call_filter_llm
                result = asyncio.run(filtering.ai_check_batch_relevance(
                    "hotel di medan apa saja?",
                    _candidates(),
                ))
                self.assertIsNone(result)

    def test_batch_rejects_non_string_reason_or_reformulated_question(self):
        filtering = _load_filtering_module()

        invalid_fields = [
            {"reason": ["invalid"], "reformulated_question": ""},
            {"reason": "invalid", "reformulated_question": {"text": "invalid"}},
        ]
        for invalid_values in invalid_fields:
            with self.subTest(invalid_values=invalid_values):
                async def call_filter_llm(**_kwargs):
                    return json.dumps({
                        "relevant": False,
                        "selected_rank": None,
                        "confidence": 0.0,
                        "answer": "",
                        **invalid_values,
                    })

                filtering.call_filter_llm = call_filter_llm
                result = asyncio.run(filtering.ai_check_batch_relevance(
                    "hotel di medan apa saja?",
                    _candidates(),
                ))
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
