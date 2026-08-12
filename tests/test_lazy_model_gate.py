import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from config import config
from shared.bootstrap import LazyModel


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.constructed = True


class LazyModelGateTests(unittest.TestCase):
    """Gate USE_SHARED_EMBEDDING: model lokal tidak boleh dimuat saat shared aktif."""

    def setUp(self):
        self._original_mode = config.USE_SHARED_EMBEDDING
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = _FakeSentenceTransformer
        self._fake_st = fake_module

    def tearDown(self):
        config.USE_SHARED_EMBEDDING = self._original_mode

    async def _get(self, holder):
        return await holder.get()

    def test_shared_mode_never_loads_local_model(self):
        config.USE_SHARED_EMBEDDING = True
        holder = LazyModel("dummy/path")
        with patch.dict(
            sys.modules,
            {"sentence_transformers": self._fake_st},
        ):
            result = asyncio.run(self._get(holder))
        self.assertIsNone(result)
        self.assertFalse(holder.loaded)

    def test_local_mode_loads_once_and_wires_on_load(self):
        config.USE_SHARED_EMBEDDING = False
        calls = []
        holder = LazyModel("dummy/path", on_load=lambda m: calls.append(m))
        with patch.dict(
            sys.modules,
            {"sentence_transformers": self._fake_st},
        ):
            first = asyncio.run(self._get(holder))
            second = asyncio.run(self._get(holder))
        self.assertIsInstance(first, _FakeSentenceTransformer)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertTrue(holder.loaded)


if __name__ == "__main__":
    unittest.main()
