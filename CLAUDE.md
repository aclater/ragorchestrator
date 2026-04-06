# ragorchestrator

LangGraph supervisor agent that sits above ragpipe, implementing agentic
RAG orchestration with tool selection, query decomposition, and adaptive
complexity routing.

## Architecture
```
Client → ragorchestrator (:8095) → supervisor LLM
                                      ↓
                              [ragpipe_retrieval tool]  ←── corpus queries
                              [web_search tool]         ←── real-time queries (optional)
                                      ↓
                              synthesis → response with rag_metadata
```

## Key design decisions
- ragpipe is a tool: supervisor calls it via LangGraph ToolNode
- Local model only: LLM calls via MODEL_URL (llama-vulkan :8080)
- OpenAI-compatible API: drop-in replacement for ragpipe at :8095
- Graceful degradation: if ragpipe is down, supervisor answers from general knowledge
- No LangSmith: sovereign deployment, optional telemetry only

## Package structure
```
ragorchestrator/
  __init__.py    — version
  __main__.py    — uvicorn entry point
  app.py         — FastAPI app, /v1/chat/completions, health, metrics
  graph.py       — LangGraph supervisor graph, state schema, tool binding
  tools/
    __init__.py
    ragpipe_tool.py    — ragpipe wrapper as LangGraph tool
    web_search_tool.py — Tavily web search (optional, needs TAVILY_API_KEY)
tests/
  test_app.py    — API endpoint tests
  test_graph.py  — graph compilation tests
  test_tools.py  — tool wrapper tests
```

## Environment variables
- `MODEL_URL`: LLM endpoint (default: http://localhost:8080)
- `MODEL_NAME`: Model name for LangChain (default: model.file)
- `RAGPIPE_URL`: ragpipe endpoint (default: http://localhost:8090)
- `RAGPIPE_ADMIN_TOKEN`: Bearer token for ragpipe
- `RAGORCHESTRATOR_PORT`: Listen port (default: 8095)
- `TAVILY_API_KEY`: Tavily API key for web search (optional, tool disabled if unset)
- `DISABLE_WEB_SEARCH`: Set to `true` to force-disable web search (sovereign/air-gapped mode)

## Running
```bash
pip install '.[dev]'                          # core only
pip install '.[dev,web-search]'               # with Tavily web search
TAVILY_API_KEY=tvly-... python -m ragorchestrator  # start with web search
DISABLE_WEB_SEARCH=true python -m ragorchestrator  # force sovereign mode
python -m pytest tests/ -v                    # run tests
```

## Endpoints
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics
