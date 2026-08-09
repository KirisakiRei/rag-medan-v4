import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import config
from shared.security import InternalAuthMiddleware, _is_allowed_path


def _make_app():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/internal/ping")
    async def ping():
        return {"status": "ok"}

    app.add_middleware(InternalAuthMiddleware)
    return app


class InternalAuthMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self._original_key = config.INTERNAL_API_KEY

    def tearDown(self):
        config.INTERNAL_API_KEY = self._original_key

    def test_fail_closed_when_key_empty(self):
        config.INTERNAL_API_KEY = ""
        client = TestClient(_make_app())
        response = client.get("/internal/ping")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json(),
            {"status": "error", "message": "Unauthorized"},
        )

    def test_missing_header_rejected(self):
        config.INTERNAL_API_KEY = "secret-key"
        client = TestClient(_make_app())
        response = client.get("/internal/ping")
        self.assertEqual(response.status_code, 401)

    def test_wrong_key_rejected(self):
        config.INTERNAL_API_KEY = "secret-key"
        client = TestClient(_make_app())
        response = client.get("/internal/ping", headers={"X-API-Key": "wrong-key"})
        self.assertEqual(response.status_code, 401)

    def test_correct_key_accepted(self):
        config.INTERNAL_API_KEY = "secret-key"
        client = TestClient(_make_app())
        response = client.get("/internal/ping", headers={"X-API-Key": "secret-key"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_is_public(self):
        config.INTERNAL_API_KEY = "secret-key"
        client = TestClient(_make_app())
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_options_preflight_allowed(self):
        config.INTERNAL_API_KEY = "secret-key"
        client = TestClient(_make_app())
        response = client.options("/internal/ping")
        self.assertNotEqual(response.status_code, 401)

    def test_allowlist_helper(self):
        self.assertTrue(_is_allowed_path("/"))
        self.assertTrue(_is_allowed_path("/health"))
        self.assertTrue(_is_allowed_path("/docs"))
        self.assertTrue(_is_allowed_path("/docs/oauth2-redirect"))
        self.assertTrue(_is_allowed_path("/openapi.json"))
        self.assertFalse(_is_allowed_path("/internal/ping"))
        self.assertFalse(_is_allowed_path("/api/search"))


if __name__ == "__main__":
    unittest.main()
