"""Tests for the FastAPI app."""

import pytest
from fastapi.testclient import TestClient

from ragorchestrator.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    """Health endpoint should return ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_metrics(client):
    """Metrics endpoint should return Prometheus format."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ragorchestrator_queries_total" in resp.text


def test_chat_completions_no_messages(client):
    """Empty messages should return 400."""
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400
