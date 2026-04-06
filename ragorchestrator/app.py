"""FastAPI app — OpenAI-compatible API surface for ragorchestrator.

Drop-in replacement for ragpipe for clients that want agentic behavior.
Exposes /v1/chat/completions with the same request/response schema.
"""

import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest

from ragorchestrator import __version__
from ragorchestrator.graph import get_graph

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

PORT = int(os.environ.get("RAGORCHESTRATOR_PORT", "8095"))

# ── Prometheus metrics ───────────────────────────────────────────────────────

queries_total = Counter(
    "ragorchestrator_queries_total",
    "Total queries processed",
    ["status"],
)
query_latency = Histogram(
    "ragorchestrator_query_latency_seconds",
    "Query latency in seconds",
)
tool_calls_total = Counter(
    "ragorchestrator_tool_calls_total",
    "Total tool calls made by supervisor",
    ["tool"],
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ragorchestrator",
    version=__version__,
    description="LangGraph supervisor agent for ragpipe",
)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "version": __version__}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from starlette.responses import Response

    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.

    Runs the query through the LangGraph supervisor graph, which
    decides whether to call ragpipe, answer directly, or use other tools.
    """
    start = time.monotonic()
    body = await request.json()

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    # Convert to LangChain message format
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    try:
        graph = get_graph()
        result = await graph.ainvoke({"messages": lc_messages})

        # Extract final response
        final_msg = result["messages"][-1]
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        # Count tool calls
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_total.labels(tool=tc.get("name", "unknown")).inc()

        # Extract rag_metadata from ragpipe tool response if available
        rag_metadata = _extract_rag_metadata(result["messages"])

        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="success").inc()

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": body.get("model", "ragorchestrator"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

        if rag_metadata:
            response["rag_metadata"] = rag_metadata

        return JSONResponse(response)

    except Exception as e:
        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="error").inc()
        log.exception("Supervisor graph failed")
        return JSONResponse({"error": str(e)}, status_code=500)


def _extract_rag_metadata(messages) -> dict | None:
    """Extract rag_metadata from ragpipe tool call results."""
    from langchain_core.messages import ToolMessage

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "ragpipe_retrieval":
            try:
                data = json.loads(msg.content)
                if "grounding" in data:
                    return {
                        "grounding": data.get("grounding"),
                        "cited_chunks": data.get("cited_chunks", []),
                        "corpus_coverage": data.get("corpus_coverage"),
                        "retrieval_attempts": data.get("retrieval_attempts", 1),
                        "query_rewritten": data.get("query_rewritten", False),
                    }
            except (json.JSONDecodeError, TypeError):
                pass
    return None
