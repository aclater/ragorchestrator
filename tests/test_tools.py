"""Tests for tool wrappers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragorchestrator.tools.ragpipe_tool import ragpipe_retrieval


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
