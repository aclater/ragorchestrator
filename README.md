# ragorchestrator

LangGraph supervisor agent for [ragpipe](https://github.com/aclater/ragpipe) — agentic RAG orchestration with tool selection, query decomposition, and adaptive complexity routing.

## Architecture

```
Client → ragorchestrator (:8095) → LangGraph supervisor
                                        ↓
                                  [ragpipe_retrieval]
                                        ↓
                                  ragpipe (:8090)
                                        ↓
                                  corpus answer + citations
```

ragorchestrator exposes an OpenAI-compatible API at port 8095. It's a drop-in replacement for ragpipe for clients that want agentic behavior. The supervisor decides whether to call ragpipe for corpus-grounded answers or respond directly for general knowledge queries.

## Quick start

```bash
# Prerequisites: ragpipe running on :8090, LLM on :8080
pip install '.[dev]'
python -m ragorchestrator

# Query
curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is Adam Claters job title?"}]}'
```

## Key features

- **LangGraph supervisor**: Adaptive routing via tool-calling LLM
- **ragpipe as a tool**: Corpus retrieval with citations preserved
- **OpenAI-compatible API**: Same request/response schema as ragpipe
- **Prometheus metrics**: Query latency, tool call counts, error rates
- **Sovereign deployment**: All LLM calls to local model, no external APIs

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | `http://localhost:8080` | LLM endpoint |
| `MODEL_NAME` | `model.file` | Model name for LangChain |
| `RAGPIPE_URL` | `http://localhost:8090` | ragpipe endpoint |
| `RAGPIPE_ADMIN_TOKEN` | | Bearer token for ragpipe |
| `RAGORCHESTRATOR_PORT` | `8095` | Listen port |

## Health and metrics

```bash
curl http://localhost:8095/health
curl http://localhost:8095/metrics
```

## License

AGPL-3.0-or-later
