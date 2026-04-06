"""Tests for tool wrappers."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragorchestrator.tools.ragpipe_tool import ragpipe_retrieval
from ragorchestrator.tools.web_search_tool import _web_search_enabled, get_web_search_tool


@pytest.mark.asyncio
async def test_ragpipe_tool_returns_json():
    """ragpipe_retrieval should return JSON with answer and metadata."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Test answer [abc:0]"}}],
        "rag_metadata": {
            "grounding": "corpus",
            "cited_chunks": [{"id": "abc:0", "title": "Test", "source": "test://"}],
            "corpus_coverage": "full",
            "retrieval_attempts": 1,
            "query_rewritten": False,
        },
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("ragorchestrator.tools.ragpipe_tool._client", mock_client):
        result = await ragpipe_retrieval.ainvoke("test query")

    data = json.loads(result)
    assert data["grounding"] == "corpus"
    assert data["answer"] == "Test answer [abc:0]"
    assert len(data["cited_chunks"]) == 1


@pytest.mark.asyncio
async def test_ragpipe_tool_handles_error():
    """ragpipe_retrieval should return error JSON on failure."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))

    with patch("ragorchestrator.tools.ragpipe_tool._client", mock_client):
        result = await ragpipe_retrieval.ainvoke("test query")

    data = json.loads(result)
    assert "error" in data


# ── Web search tool tests ────────────────────────────────────────────────────


class TestWebSearchEnabled:
    def test_disabled_when_no_api_key(self):
        """Web search should be disabled when TAVILY_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _web_search_enabled() is False

    def test_disabled_when_explicit_disable(self):
        """Web search should be disabled when DISABLE_WEB_SEARCH=true."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "DISABLE_WEB_SEARCH": "true"}):
            assert _web_search_enabled() is False

    def test_disabled_when_disable_yes(self):
        """Web search should be disabled when DISABLE_WEB_SEARCH=yes."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "DISABLE_WEB_SEARCH": "yes"}):
            assert _web_search_enabled() is False

    def test_disabled_when_disable_1(self):
        """Web search should be disabled when DISABLE_WEB_SEARCH=1."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "DISABLE_WEB_SEARCH": "1"}):
            assert _web_search_enabled() is False

    def test_enabled_when_api_key_set(self):
        """Web search should be enabled when TAVILY_API_KEY is set and not disabled."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True):
            assert _web_search_enabled() is True

    def test_enabled_when_disable_false(self):
        """Web search should be enabled when DISABLE_WEB_SEARCH=false."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key", "DISABLE_WEB_SEARCH": "false"}):
            assert _web_search_enabled() is True


class TestGetWebSearchTool:
    def test_returns_none_when_disabled(self):
        """get_web_search_tool should return None when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_web_search_tool() is None

    def test_returns_tool_when_enabled(self):
        """get_web_search_tool should return a tool when enabled."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True):
            tool = get_web_search_tool()
            assert tool is not None
            assert "web" in tool.description.lower()


class TestGraphWithWebSearch:
    def test_graph_compiles_with_web_search(self):
        """Graph should compile when web search is enabled."""
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True),
            patch("ragorchestrator.graph.ChatOpenAI"),
        ):
            from ragorchestrator.graph import build_graph

            graph = build_graph()
            assert graph is not None

    def test_graph_compiles_without_web_search(self):
        """Graph should compile when web search is disabled."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ragorchestrator.graph.ChatOpenAI"),
        ):
            from ragorchestrator.graph import build_graph

            graph = build_graph()
            assert graph is not None
