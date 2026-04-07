"""Live integration tests for ragorchestrator.

These tests hit the live service at :8095 — no mocking.
Run with: pytest tests/test_integration.py -v --live
Skip automatically when --live flag is not provided.

Note: complex queries run through the full LangGraph agentic loop which
makes 5+ sequential LLM calls. With Qwen3-32B, each call takes 15-45s,
so complex tests need ~5 minute timeouts.
"""

import json
import os
import urllib.error
import urllib.request

import pytest

RAGORCHESTRATOR_URL = os.environ.get("RAGORCHESTRATOR_URL", "http://127.0.0.1:8095")
RAGPIPE_URL = os.environ.get("RAGPIPE_URL", "http://127.0.0.1:8090")


@pytest.fixture(scope="module")
def live_service():
    """Check that the live service is reachable before running tests."""
    try:
        resp = urllib.request.urlopen(f"{RAGORCHESTRATOR_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        if data.get("status") != "ok":
            pytest.skip("ragorchestrator not healthy")
    except Exception:
        pytest.skip("ragorchestrator not reachable")


def _chat(query: str, timeout: int = 120) -> dict:
    """Send a chat completion request to the live ragorchestrator."""
    payload = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pytest.fail(f"HTTP {e.code}: {body[:500]}")


@pytest.mark.live
def test_health_endpoint(live_service):
    """GET /health returns {status: ok}."""
    resp = urllib.request.urlopen(f"{RAGORCHESTRATOR_URL}/health", timeout=5)
    data = json.loads(resp.read().decode())
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.live
def test_chat_completions_returns_choices(live_service):
    """POST /v1/chat/completions returns 200 with choices array (simple path)."""
    data = _chat("what is NATO article 5")
    assert "choices" in data, f"Response missing 'choices': {data}"
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["choices"][0]["finish_reason"] == "stop"


@pytest.mark.live
def test_chat_completions_simple_query(live_service):
    """Simple factual query returns grounded response."""
    data = _chat("who is in the personnel database")
    assert "choices" in data, f"Response missing 'choices': {data}"
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 10


@pytest.mark.live
def test_chat_completions_complex_query(live_service):
    """Complex multi-topic query returns response via agentic path.

    This test exercises the full LangGraph loop: supervisor → decompose →
    multi_tools → generate → reflect. Requires ~5 sequential LLM calls.
    """
    data = _chat("compare the roles and responsibilities of personnel in the NATO documents", timeout=300)
    assert "choices" in data, f"Response missing 'choices': {data}"
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 10


@pytest.mark.live
def test_ragpipe_reachable_from_host(live_service):
    """ragpipe is reachable at RAGPIPE_URL."""
    try:
        resp = urllib.request.urlopen(f"{RAGPIPE_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
    except Exception as e:
        pytest.fail(f"ragpipe not reachable at {RAGPIPE_URL}: {e}")


@pytest.mark.live
def test_response_has_openai_fields(live_service):
    """Response has required OpenAI-compatible fields (agentic path)."""
    data = _chat("hello", timeout=120)
    assert "choices" in data, f"Response missing 'choices': {data}"
    assert "id" in data
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert "model" in data
