# API Reference

## POST /v1/chat/completions

OpenAI-compatible chat completions endpoint. Drop-in replacement for ragpipe.

### Request

```bash
curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is NATO article 5?"}],
    "stream": false
  }'
```

### Response

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "model": "default",
  "complexity": "simple",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "NATO Article 5 states that..."
      },
      "finish_reason": "stop"
    }
  ],
  "rag_metadata": {
    "grounding": "corpus",
    "cited_chunks": [
      {"id": "abc-123:0", "title": "NATO Treaty", "source": "gdrive://nato.pdf"}
    ],
    "corpus_coverage": "full",
    "retrieval_attempts": 1,
    "query_rewritten": false
  }
}
```

### Response fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique completion ID |
| `object` | string | Always `chat.completion` |
| `model` | string | Model name from request |
| `complexity` | string | Query classification: `simple`, `complex`, or `external` |
| `choices` | array | OpenAI-compatible choices array |
| `rag_metadata` | object | RAG metadata from ragpipe (optional, present when ragpipe was called) |

### rag_metadata fields

Passed through from ragpipe when the supervisor calls the ragpipe_retrieval tool:

| Field | Type | Description |
|-------|------|-------------|
| `grounding` | string | `corpus`, `general`, or `mixed` |
| `cited_chunks` | array | List of `{id, title, source}` objects |
| `corpus_coverage` | string | `full` or `none` |
| `retrieval_attempts` | int | 1 = normal, 2 = CRAG retry |
| `query_rewritten` | bool | Whether CRAG triggered a query rewrite |

### Error responses

Errors return HTTP 500 with an OpenAI-compatible body that includes both `choices` and `error`:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "default",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "I encountered an error..."}, "finish_reason": "stop"}],
  "error": "Connection error."
}
```

### Streaming

Streaming responses (`"stream": true`) return SSE events. The `complexity` field is emitted as a separate SSE event before `[DONE]`.

## GET /health

```json
{"status": "ok", "version": "0.1.0"}
```

## GET /metrics

Prometheus text format. See [architecture.md](architecture.md#metrics) for metric descriptions.
