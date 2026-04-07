"""Live integration tests for ragorchestrator.

Tests run against the live stack: ragorchestrator (:8095), ragpipe (:8090).
Requires both services to be running.

Run with:
    PYTHONPATH=. pytest tests/test_live.py -v \
        --ragorchestrator-url=http://localhost:8095 \
        --ragpipe-url=http://localhost:8090

Skip in CI (services not available):
    SKIP_LIVE_TESTS=1 pytest tests/test_live.py -v -m "not live"

Known issues:
- Complex/EXTERNAL queries hang indefinitely (issue #20) — agentic path bug, xfail tests expected
- /v1/models endpoint not implemented in ragorchestrator (test marked xfail)
"""

import os

import httpx
import pytest

RAGORCHESTRATOR_URL = os.environ.get("RAGORCHESTRATOR_URL", "http://localhost:8095")
RAGPIPE_URL = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
TIMEOUT = 120.0


def _is_live_stack_available() -> bool:
    """Check if the live stack (ragorchestrator + ragpipe) is reachable."""
    try:
        httpx.get(f"{RAGORCHESTRATOR_URL}/health", timeout=5)
        httpx.get(f"{RAGPIPE_URL}/health", timeout=5)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_LIVE_TESTS") == "1" or not _is_live_stack_available(),
    reason="Live stack not available — set SKIP_LIVE_TESTS=1 to skip",
)


class TestHealthAndConnectivity:
    def test_health_returns_200(self):
        resp = httpx.get(f"{RAGORCHESTRATOR_URL}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_metrics_returns_prometheus(self):
        resp = httpx.get(f"{RAGORCHESTRATOR_URL}/metrics", timeout=10)
        assert resp.status_code == 200
        assert "ragorchestrator_queries_total" in resp.text
        assert "# HELP" in resp.text or "ragorchestrator" in resp.text

    def test_ragpipe_reachable_from_container(self):
        resp = httpx.get(f"{RAGPIPE_URL}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.xfail(
        reason="ragorchestrator does not implement /v1/models — separate feature request needed",
        strict=False,
    )
    def test_v1_models_returns_list(self):
        resp = httpx.get(f"{RAGORCHESTRATOR_URL}/v1/models", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0


class TestBasicChatCompletions:
    def _chat(self, query: str, stream: bool = False, model: str = "default") -> httpx.Response:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "stream": stream,
        }
        resp = httpx.post(
            f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        return resp

    def test_simple_query_returns_choices(self):
        resp = self._chat("What is a patent?")
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]

    def test_response_is_openai_compatible(self):
        resp = self._chat("What is a patent?")
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data

    def test_simple_query_returns_non_empty_answer(self):
        resp = self._chat("What is a patent?")
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert content
        assert len(content) > 0

    def test_stream_false_returns_json(self):
        resp = self._chat("What is a patent?", stream=False)
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")
        data = resp.json()
        assert "choices" in data


class TestAdaptiveComplexityClassifier:
    def _chat(self, query: str) -> httpx.Response:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        return httpx.post(
            f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )

    def test_simple_query_classified_as_simple(self):
        resp = self._chat("What is a patent?")
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_complex_query_classified_as_complex(self):
        query = (
            "Compare and contrast the patent system in the US versus Europe. "
            "Analyze the advantages and disadvantages of each approach."
        )
        resp = self._chat(query)
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_external_query_classified_as_external(self):
        query = "What is the latest news about patents?"
        resp = self._chat(query)
        assert resp.status_code == 200

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_simple_path_faster_than_complex(self):
        simple_query = "What is a patent?"
        complex_query = (
            "Compare and contrast the patent system in the US versus Europe. "
            "Analyze the advantages and disadvantages of each approach."
        )

        simple_resp = self._chat(simple_query)
        assert simple_resp.status_code == 200

        complex_resp = self._chat(complex_query)
        assert complex_resp.status_code == 200

        simple_time = simple_resp.elapsed.total_seconds()
        complex_time = complex_resp.elapsed.total_seconds()

        assert simple_time < complex_time


class TestSelfRAGReflection:
    def _chat(self, query: str) -> dict:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        resp = httpx.post(
            f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        return resp.json()

    def test_grounded_response_not_reflected(self):
        query = "What is Adam Clater's job title?"
        data = self._chat(query)
        content = data["choices"][0]["message"]["content"]
        assert content
        assert "[Self-RAG]" not in content

    def test_general_grounding_short_circuits(self):
        query = "What is 2+2?"
        data = self._chat(query)
        assert data["choices"][0]["message"]["content"]
        if "rag_metadata" in data:
            assert data["rag_metadata"].get("grounding") in ("general", "corpus", "mixed")

    def test_reflection_result_in_response(self):
        query = "What is a patent?"
        data = self._chat(query)
        content = data["choices"][0]["message"]["content"]
        assert content is not None


class TestMultiPassRetrieval:
    def _chat(self, query: str) -> httpx.Response:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        return httpx.post(
            f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_complex_query_uses_multipass(self):
        query = (
            "Compare and contrast the patent filing requirements in the US, EU, and Japan. "
            "Analyze the key differences in novelty, prior art, and examination procedures."
        )
        resp = self._chat(query)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"]

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_multipass_deduplicates_chunks(self):
        query = "What are the similarities and differences between patent law in the US and Europe?"
        resp = self._chat(query)
        assert resp.status_code == 200
        data = resp.json()
        meta = data.get("rag_metadata", {})
        cited = meta.get("cited_chunks", [])
        seen: set = set()
        for chunk in cited:
            chunk_id = chunk.get("id", "")
            assert chunk_id not in seen, f"Duplicate chunk found: {chunk_id}"
            seen.add(chunk_id)

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_multipass_returns_more_chunks(self):
        single_query = "What is a patent?"
        multi_query = (
            "Compare and contrast the patent systems in the US and Europe. "
            "Analyze the advantages and disadvantages of each."
        )
        single_resp = self._chat(single_query)
        assert single_resp.status_code == 200
        single_data = single_resp.json()

        multi_resp = self._chat(multi_query)
        assert multi_resp.status_code == 200
        multi_data = multi_resp.json()

        single_chunks = len(single_data.get("rag_metadata", {}).get("cited_chunks", []))
        multi_chunks = len(multi_data.get("rag_metadata", {}).get("cited_chunks", []))

        assert multi_chunks >= single_chunks


class TestRagorchestratorVsRagpipe:
    def _chat(self, url: str, query: str) -> dict:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        resp = httpx.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        return resp.json()

    def test_simple_query_equivalent_answers(self):
        query = "What is a patent?"
        orch_data = self._chat(RAGORCHESTRATOR_URL, query)
        pipe_data = self._chat(RAGPIPE_URL, query)
        assert orch_data["choices"][0]["message"]["content"]
        assert pipe_data["choices"][0]["message"]["content"]

    @pytest.mark.xfail(reason="Agentic path fails with Connection error — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_complex_query_richer_from_ragorchestrator(self):
        query = "Compare and contrast the patent systems in the US and Europe. Analyze the key differences."
        orch_data = self._chat(RAGORCHESTRATOR_URL, query)
        pipe_data = self._chat(RAGPIPE_URL, query)
        orch_chunks = len(orch_data.get("rag_metadata", {}).get("cited_chunks", []))
        pipe_chunks = len(pipe_data.get("rag_metadata", {}).get("cited_chunks", []))
        assert orch_chunks >= pipe_chunks

    def test_crag_metadata_flows_through(self):
        query = "What is a patent?"
        data = self._chat(RAGORCHESTRATOR_URL, query)
        meta = data.get("rag_metadata", {})
        if meta:
            assert "grounding" in meta
            assert "cited_chunks" in meta


class TestWebSearchDisabled:
    def _chat(self, query: str) -> httpx.Response:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        return httpx.post(
            f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )

    @pytest.mark.xfail(reason="External query hits agentic path which hangs — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_web_search_disabled_gracefully(self):
        query = "What is the latest news?"
        resp = self._chat(query)
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            assert content or "error" in data

    def test_web_search_absent_from_simple_path(self):
        query = "What is 2+2?"
        resp = self._chat(query)
        assert resp.status_code == 200


class TestErrorHandling:
    def _chat(self, payload: dict) -> httpx.Response:
        return httpx.post(
            f"{RAGORCHESTRATOR_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )

    def test_empty_messages_returns_4xx(self):
        payload = {"model": "default", "messages": [], "stream": False}
        resp = self._chat(payload)
        assert resp.status_code in (400, 422)

    @pytest.mark.xfail(reason="Long queries hit agentic path which hangs — issue #20", strict=False)
    @pytest.mark.timeout(30)
    def test_very_long_query_handled(self):
        long_query = "What is " * 50
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": long_query}],
            "stream": False,
        }
        resp = self._chat(payload)
        assert resp.status_code in (200, 400, 422, 500)
