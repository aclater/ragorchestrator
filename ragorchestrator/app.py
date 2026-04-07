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
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest

from ragorchestrator import __version__
from ragorchestrator.classifier import Complexity, classify
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
complexity_classified = Counter(
    "ragorchestrator_complexity_classified_total",
    "Total queries classified by complexity",
    ["complexity"],
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

    Runs the query through adaptive routing:
    - SIMPLE queries: direct ragpipe call (fast path)
    - COMPLEX/EXTERNAL queries: full LangGraph agentic loop
    """
    start = time.monotonic()
    body = await request.json()

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    # Extract last user message for classification
    user_query = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    complexity = classify(user_query)
    complexity_classified.labels(complexity=complexity.value).inc()
    log.info("Query complexity: %s", complexity.value)

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

    stream = body.get("stream", False)
    model_name = body.get("model", "ragorchestrator")

    if stream:
        if complexity == Complexity.SIMPLE:
            return StreamingResponse(
                _stream_simple_path(user_query, model_name, start),
                media_type="text/event-stream",
            )
        else:
            return StreamingResponse(
                _stream_agentic_path(lc_messages, model_name, start),
                media_type="text/event-stream",
            )

    try:
        if complexity == Complexity.SIMPLE:
            result = await _simple_path(user_query, model_name)
        else:
            result = await _agentic_path(lc_messages)

        # Extract final response
        final_msg = result["messages"][-1]
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        # Count tool calls
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_total.labels(tool=tc.get("name", "unknown")).inc()

        # Extract rag_metadata: simple path has it directly, agentic path extracts from messages
        if complexity == Complexity.SIMPLE:
            rag_metadata = result.get("rag_metadata")
        else:
            rag_metadata = _extract_rag_metadata(result["messages"])

        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="success").inc()

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": model_name,
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
        log.exception("Query processing failed")
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"I encountered an error processing your request: {e}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "error": str(e),
            },
            status_code=500,
        )


def _sse_chunk(chunk_id: str, model: str, content: str, finish_reason: str | None = None) -> str:
    """Format a single SSE chunk in OpenAI streaming format."""
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


async def _stream_simple_path(query: str, model: str, start: float):
    """Stream SSE events by proxying ragpipe's streaming response."""
    import httpx

    ragpipe_url = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
    ragpipe_token = os.environ.get("RAGPIPE_ADMIN_TOKEN", "")

    headers = {"Content-Type": "application/json"}
    if ragpipe_token:
        headers["Authorization"] = f"Bearer {ragpipe_token}"

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
        "stream": True,
    }

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{ragpipe_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield _sse_chunk(chunk_id, model, content)
                    except json.JSONDecodeError:
                        continue

        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="success").inc()
    except Exception as e:
        log.exception("Streaming simple path failed")
        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="error").inc()
        yield _sse_chunk(chunk_id, model, f"\n\n[Error: {e}]")

    yield _sse_chunk(chunk_id, model, "", finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _stream_agentic_path(lc_messages: list, model: str, start: float):
    """Run agentic graph, then stream the final content as SSE chunks."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        graph = get_graph()
        result = await graph.ainvoke({"messages": lc_messages})

        # Count tool calls
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_total.labels(tool=tc.get("name", "unknown")).inc()

        final_msg = result["messages"][-1]
        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        # Emit content in word-sized chunks for a natural streaming feel
        words = content.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else f" {word}"
            yield _sse_chunk(chunk_id, model, token)

        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="success").inc()
        log.info("Streaming agentic path completed")
    except Exception as e:
        log.exception("Streaming agentic path failed")
        elapsed = time.monotonic() - start
        query_latency.observe(elapsed)
        queries_total.labels(status="error").inc()
        yield _sse_chunk(chunk_id, model, f"Error: {e}")

    yield _sse_chunk(chunk_id, model, "", finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _simple_path(query: str, model: str) -> dict:
    """Fast path: call ragpipe directly for simple queries.

    Bypasses the full LangGraph agentic loop for speed.
    Returns a result dict with messages list containing the response.
    """
    import httpx
    from langchain_core.messages import AIMessage

    ragpipe_url = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
    ragpipe_token = os.environ.get("RAGPIPE_ADMIN_TOKEN", "")

    headers = {"Content-Type": "application/json"}
    if ragpipe_token:
        headers["Authorization"] = f"Bearer {ragpipe_token}"

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ragpipe_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            meta = data.get("rag_metadata", {})

            result = {
                "messages": [
                    AIMessage(content=content),
                ],
                "rag_metadata": meta,
            }
            log.info("Simple path: direct ragpipe call completed")
            return result
    except Exception as e:
        log.warning("Simple path ragpipe call failed: %s", e)
        raise


async def _agentic_path(lc_messages: list) -> dict:
    """Agentic path: run full LangGraph supervisor loop.

    For COMPLEX and EXTERNAL queries that need multi-step reasoning.
    """
    graph = get_graph()
    result = await graph.ainvoke({"messages": lc_messages})
    log.info("Agentic path: full LangGraph loop completed")
    return result


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
