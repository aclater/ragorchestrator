# Configuration

All configuration is via environment variables. No config files.

## Required

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | `http://127.0.0.1:8080` | LLM endpoint (OpenAI-compatible). Must use IPv4 — llama-vulkan only listens on IPv4. |
| `RAGPIPE_URL` | `http://localhost:8090` | ragpipe endpoint for corpus retrieval |

## Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `model.file` | Model identifier passed to ChatOpenAI |
| `RAGPIPE_ADMIN_TOKEN` | (empty) | Bearer token for ragpipe authentication |
| `RAGORCHESTRATOR_PORT` | `8095` | Listen port for the FastAPI server |
| `TAVILY_API_KEY` | (unset) | Tavily API key for web search. Tool disabled when unset. |
| `DISABLE_WEB_SEARCH` | (unset) | Set to `true`, `yes`, or `1` to force-disable web search even when API key is present |
| `LANGSMITH_TRACING` | (unset) | Set to `true` to enable LangSmith tracing. Disabled by default for sovereign deployment. |

## Deployment notes

### IPv4 requirement

`MODEL_URL` must use `127.0.0.1` (not `localhost`) when targeting llama-vulkan on Fedora. Python resolves `localhost` to `::1` (IPv6) first, but llama-vulkan only listens on IPv4. This applies to all services running as rootless Podman containers with `Network=host`.

### Sovereign mode

Set `DISABLE_WEB_SEARCH=true` to ensure no external API calls are made. In this mode:
- Only the ragpipe_retrieval tool is available
- EXTERNAL queries fall back to ragpipe-only retrieval
- No data leaves the local network

### Container deployment

The Podman quadlet sets all required variables:

```ini
Environment=MODEL_URL=http://127.0.0.1:8080
Environment=RAGPIPE_URL=http://host.containers.internal:8090
Environment=DISABLE_WEB_SEARCH=true
EnvironmentFile=%h/.config/llm-stack/ragstack.env
```

Secrets (`RAGPIPE_ADMIN_TOKEN`, `TAVILY_API_KEY`) are sourced from `ragstack.env` — never committed to git.
