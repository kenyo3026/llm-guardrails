"""
Unit tests for CascadeGuard API.

Tests cover:
- Health check endpoint
- List guardrails endpoint
- Apply endpoint (POST and GET)

Usage:
    pytest src/cascadeguard/tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from cascadeguard.api import create_app


@pytest.fixture
def mock_main():
    """Mock Main to avoid real config and LLM calls."""
    with patch("cascadeguard.api.resolve_config_path", return_value="/tmp/config.yaml"):
        with patch("cascadeguard.api.Main") as MockMain:
            instance = MagicMock()
            instance.list_guardrails.return_value = ["relevance_guardrail"]
            instance.apply.return_value = [
                {
                    "idx": 0,
                    "prompt": "What is AI?",
                    "output": "AI is artificial intelligence.",
                    "is_valid": True,
                    "risk_score": 0.1,
                }
            ]
            MockMain.return_value = instance
            yield instance


@pytest.fixture
def client(mock_main):
    """TestClient with mocked Main."""
    app = create_app()
    return TestClient(app)


class TestHealth:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestGuardrails:
    """Tests for list guardrails endpoints."""

    def test_guardrails_returns_list(self, client, mock_main):
        resp = client.get("/guardrails")
        assert resp.status_code == 200
        data = resp.json()
        assert "guardrails" in data
        assert data["guardrails"] == ["relevance_guardrail"]
        mock_main.list_guardrails.assert_called_once()

    def test_list_alias_returns_same(self, client, mock_main):
        resp = client.get("/list")
        assert resp.status_code == 200
        assert resp.json()["guardrails"] == ["relevance_guardrail"]


class TestApply:
    """Tests for apply endpoint."""

    def test_apply_post_returns_results(self, client, mock_main):
        payload = {
            "pairs": [{"prompt": "What is AI?", "output": "AI is artificial intelligence."}],
            "winnow_down": True,
        }
        resp = client.post("/apply", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["prompt"] == "What is AI?"
        mock_main.apply.assert_called_once()

    def test_apply_post_with_guardrail_param(self, client, mock_main):
        payload = {
            "pairs": [{"prompt": "Q", "output": "A"}],
            "guardrail": "relevance_guardrail",
            "winnow_down": True,
        }
        resp = client.post("/apply", json=payload)
        assert resp.status_code == 200
        call_kwargs = mock_main.apply.call_args[1]
        assert call_kwargs["guardrail_name"] == "relevance_guardrail"

    def test_apply_get_returns_results(self, client, mock_main):
        resp = client.get("/apply", params={"prompt": "Q", "output": "A"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1

    def test_apply_post_empty_pairs_returns_empty_results(self, client, mock_main):
        mock_main.apply.return_value = []
        resp = client.post("/apply", json={"pairs": []})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_apply_get_missing_params_returns_422(self, client):
        resp = client.get("/apply")
        assert resp.status_code == 422


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from .base import run_tests_with_report

    sys.exit(run_tests_with_report(__file__, "api"))
