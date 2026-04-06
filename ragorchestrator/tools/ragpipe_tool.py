"""ragpipe tool wrapper — calls ragpipe /v1/chat/completions as a LangGraph tool.

ragpipe is the sovereign retrieval engine. This wrapper makes it
available as a LangGraph-compatible tool so the supervisor can call it
for corpus-grounded answers with citations.
"""

import json
import os

import httpx
from langchain_core.tools import tool

RAGPIPE_URL = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
RAGPIPE_TOKEN = os.environ.get("RAGPIPE_ADMIN_TOKEN", "")

_client = httpx.AsyncClient(timeout=120)


@tool
async def ragpipe_retrieval(query: str) -> str:
    """Search the document corpus using ragpipe.

    Returns a grounded answer with citations and RAG metadata.
    Use for questions about internal documents, personnel, patents,
    defense analysis, or any domain-specific knowledge.
    """
    headers = {"Content-Type": "application/json"}
    if RAGPIPE_TOKEN:
        headers["Authorization"] = f"Bearer {RAGPIPE_TOKEN}"

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
        "stream": False,
    }

    try:
        resp = await _client.post(
            f"{RAGPIPE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        meta = data.get("rag_metadata", {})

        result = {
            "answer": content,
            "grounding": meta.get("grounding", "unknown"),
            "cited_chunks": meta.get("cited_chunks", []),
            "corpus_coverage": meta.get("corpus_coverage", "none"),
            "retrieval_attempts": meta.get("retrieval_attempts", 1),
            "query_rewritten": meta.get("query_rewritten", False),
        }
        return json.dumps(result)
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"ragpipe returned {e.response.status_code}", "answer": ""})
    except Exception as e:
        return json.dumps({"error": str(e), "answer": ""})
