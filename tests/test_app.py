"""Tests for the FastAPI app."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from ragorchestrator.app import _sse_chunk, app


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


def test_sse_chunk_format():
    """SSE chunk helper should produce valid OpenAI streaming format."""
    chunk = _sse_chunk("chatcmpl-abc", "test-model", "hello")
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    data = json.loads(chunk[6:])
    assert data["object"] == "chat.completion.chunk"
    assert data["choices"][0]["delta"]["content"] == "hello"
    assert data["choices"][0]["finish_reason"] is None


def test_sse_chunk_finish():
    """SSE chunk with finish_reason should set it and have empty delta."""
    chunk = _sse_chunk("chatcmpl-abc", "test-model", "", finish_reason="stop")
    data = json.loads(chunk[6:])
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["delta"] == {}


def test_stream_simple_returns_sse(client):
    """Streaming simple query should return text/event-stream."""
    ragpipe_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in ragpipe_lines:
            yield line

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.stream = lambda *args, **kwargs: mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "who is adam"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "data: " in body
    assert "data: [DONE]" in body


def test_stream_false_returns_json(client):
    """Non-streaming request should still return JSON."""
    from langchain_core.messages import AIMessage

    mock_result = {
        "messages": [AIMessage(content="test answer")],
        "rag_metadata": {"grounding": "corpus"},
    }

    with patch("ragorchestrator.app._simple_path", new_callable=AsyncMock, return_value=mock_result):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "who is adam"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "test answer"


def test_complexity_field_in_response(client):
    """Non-streaming response must include complexity field (fixes #21)."""
    from langchain_core.messages import AIMessage

    mock_result = {
        "messages": [AIMessage(content="Paris is the capital of France.")],
        "rag_metadata": {"grounding": "general"},
    }

    with patch("ragorchestrator.app._simple_path", new_callable=AsyncMock, return_value=mock_result):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "what is the capital of France"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "complexity" in data, "Response missing complexity field"
    assert data["complexity"] == "simple"


def test_complexity_field_complex_query(client):
    """Complex queries should show complexity=complex in response (fixes #21)."""
    from langchain_core.messages import AIMessage

    mock_result = {
        "messages": [AIMessage(content="Comparing NATO and patent law...")],
    }

    with patch("ragorchestrator.app._agentic_path", new_callable=AsyncMock, return_value=mock_result):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Compare and analyze NATO Article 5 with patent claims requirements"}
                ],
                "stream": False,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "complexity" in data
    assert data["complexity"] == "complex"


def test_error_response_has_choices(client):
    """Error responses should still include choices array for OpenAI compatibility."""
    with patch("ragorchestrator.app._simple_path", new_callable=AsyncMock, side_effect=Exception("test error")):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "who is adam"}],
                "stream": False,
            },
        )

    assert resp.status_code == 500
    data = resp.json()
    assert "choices" in data, f"Error response missing 'choices': {data}"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "error" in data
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")

