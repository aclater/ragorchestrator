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


## Always verify current versions before using them

This is a hard requirement, not a suggestion. Using stale version numbers
wastes time, breaks builds, and has caused real incidents on this stack.

- BEFORE referencing any version number — for a container image, Python
  package, GitHub Action, ROCm release, npm package, LLM model, or any
  other software — look it up. Do not use version numbers from training
  knowledge. They are outdated.
- For GitHub Actions: verify via API before writing any workflow file:
    gh api repos/<owner>/<action>/releases/latest | jq .tag_name
  Use the exact tag returned. Never use what you think the version is.
  This has caused broken CI multiple times on this stack.
- For container images: check the registry (quay.io, ghcr.io,
  registry.access.redhat.com, docker.io) for the current stable tag.
  Never use :latest in production quadlets — pin to a specific tag or digest.
- For Python packages: check PyPI for the current stable release before pinning.
- For ROCm: check https://rocm.docs.amd.com for the current stable release.
- For npm packages: run npm show <package> version before pinning.
- For LLM models: check Hugging Face directly for current releases.
- If you cannot verify a version, say so explicitly. Do not guess.


## GPU acceleration

This system uses a Ryzen AI Max+ 395 APU (gfx1151) with 128GB unified memory.
There is no discrete VRAM — all GPU memory is GTT (system RAM mapped for GPU
access via ROCm). This is normal and expected for this hardware.

Memory architecture:
- VRAM: 512MB (GPU housekeeping only)
- GTT: ~113GB (all model weights, KV cache, and inference use GTT)
- GPU executes compute against GTT — this is full GPU inference, not CPU fallback
- rocm-smi --showmeminfo gtt confirms current GTT allocation

ROCm constraints on gfx1151:
- HSA_OVERRIDE_GFX_VERSION=11.5.1 required in all quadlets and scripts using ROCm
- MIGraphXExecutionProvider is the only working AMD GPU path for ONNX Runtime on ROCm 7.x
- ROCMExecutionProvider is deprecated and removed since ORT 1.23 — do not use it
- MIGRAPHX_BATCH_SIZE=64 — MIGraphX uses static shapes, pad all batches to this size
- ORT_MIGRAPHX_MODEL_CACHE_PATH — use this env var for MXR caching (not the
  model_cache_dir provider option — AMD does not compile that into their .so)
- MXR cache: 149MB .mxr file, cached on ragpipe-model-cache volume
  Cold start (no cache): ~3 minutes 53 seconds
  Warm start (cache hit): ~6 seconds (39x improvement)
  Do not treat a 6-second ragpipe startup as a problem — the cache is working

GPU detection for multi-vendor code:
- Detection priority: NVIDIA CUDA > AMD ROCm/MIGraphX > Intel XPU > CPU
- Never hardcode a vendor — detect at runtime
- For ONNX Runtime: CUDAExecutionProvider > MIGraphXExecutionProvider >
  OpenVINOExecutionProvider > CPUExecutionProvider

Container GPU passthrough:
- AMD ROCm: --device /dev/kfd --device /dev/dri
- NVIDIA: --device /dev/nvidia0 (or --gpus all with nvidia-container-toolkit)
- Intel: --device /dev/dri


## Repository location

All permanent repositories live under ~/git/.

- Never clone or initialize a repository anywhere else — not in ~/,
  not in /tmp, not in ~/Documents.
- Temporary PR work goes in ~/git-work/<issue-number>-<description>/
  (see Working directory conventions below)
- When referencing local repos, always use ~/git/<reponame> as the path.


## Working directory conventions

- ~/git/          — permanent repositories only. Long-term work lives here.
- ~/git-work/     — temporary clones for PR work only.
                    Always use ~/git-work/<issue-number>-<description>/
                    Clean up after the PR is merged.
- ~/.local/bin/   — user scripts and tools. Never use ~/bin/.

Never create git-* directories directly in ~/. They clutter the home
directory and never get cleaned up.

When starting any task requiring a repo clone:
  mkdir -p ~/git-work/<issue-number>-<description>
  cd ~/git-work/<issue-number>-<description>
  gh repo clone aclater/<repo>

When the PR is merged:
  rm -rf ~/git-work/<issue-number>-<description>

Or run periodically:
  ~/.local/bin/cleanup-git-work.sh --dry-run
  ~/.local/bin/cleanup-git-work.sh


## GitHub issue workflow

Every task must be tracked in a GitHub issue before work begins. Mandatory.

Before starting any implementation task:
1. Check for an existing issue:
     gh issue list --repo aclater/<repo> --search "<description>"
2. If none exists, create one:
     gh issue create \
       --repo aclater/<repo> \
       --title "<clear title describing root cause not symptom>" \
       --body "<context, problem, proposed fix, success criteria>" \
       --label "priority: <blocking|high|medium|low>,type: <bug|feature|infrastructure>,agent: <claude-code|minimax-m27|minimax-m25|human>"
3. Claim the issue immediately:
     gh issue edit <number> --repo aclater/<repo> --add-assignee "@me"
     gh issue comment <number> --repo aclater/<repo> \
       --body "Picked up. Working on branch fix/issue-<number>-<description>."
4. Log major milestones and learnings as issue comments as you work.
   Do not wait until the PR is open — comment as you discover things.
   The issue comment thread is the audit trail. Make it useful.

All commits must reference the issue:
  feat(ragpipe): add prometheus metrics endpoint (fixes #14)
  fix(ragstuffer): deduplicate cited chunks in streaming path (refs #8)

All PR bodies must include:
  Closes #N   — if the PR fully resolves the issue
  Refs #N     — if the PR partially addresses the issue

Never start implementation without an issue number.


## Autonomous operation

You are operating autonomously. Do not ask for confirmation before taking
action. Do not prompt the user with yes/no questions. Make decisions, execute
them, log them to GitHub issues, and move on.

The only exception: if you encounter a situation that could cause irreversible
data loss — comment on the issue explaining what you found and why you stopped,
then stop.

Autonomous decision rules — apply these without asking:
- Tests fail after a fix: investigate, fix, push again.
- CI failing on your branch: fix it and push.
- Dependency missing: install it.
- Migration needed: run it against live Postgres.
- ragpipe needs restart: restart it, wait for healthy (warm start ~6s),
  log the restart reason in the GitHub issue comment.
- New bug discovered while working: create a GitHub issue for it, note it
  in the current issue comment, continue with current task.
- Unsure between two approaches: pick the simpler one, document reasoning
  in the issue comment, proceed.
- Flaky test: fix the test.
- CI still running when task is done: wait for CI to complete before
  moving to the next issue.

Log these milestones to the GitHub issue as comments:
- When you start: your plan and implementation approach
- When you hit a significant obstacle and how you resolved it
- When you make a non-obvious technical decision and why
- When tests pass or fail (with counts)
- When the PR is open: PR URL and CI status
- When CI passes: confirmation and any remaining notes


## Container and deployment standards

- Use Podman, not Docker. Use rootless Podman quadlets, not docker-compose.
- Base images: prefer Red Hat UBI (registry.access.redhat.com/ubi10/ or
  registry.access.redhat.com/ubi9/) for all Python services.
- Never use :latest in production quadlets — pin to specific tag or digest.
- All containers must run as non-root (USER 1001 or equivalent).
- All containers must have a HEALTHCHECK defined.
- SecurityLabelDisable=true requires an inline comment explaining the specific
  SELinux constraint that requires it and referencing the relevant ADR.
- No bind mounts for source code in production quadlets.
- No credentials in committed files — use ragstack.env (not committed).
- One logical change per commit. Squash fixup commits before upstream PRs.


## rag-suite architecture context

Services and ports:
- ragpipe         :8090  — RAG proxy, embedding, reranking, grounding, citations
- ragstuffer      :8091  — ingestion (Drive, git, web)
- ragstuffer-mpep :8093  — second ragstuffer instance for USPTO/MPEP collection
- ragwatch        :9090  — Prometheus metrics aggregator
- ragdeck         :8092  — admin UI (FastAPI + frontend)
- Ollama/Vulkan   :8080  — LLM inference (Qwen3-32B dense Q4_K_M, ~19GB GTT)
- Qdrant          :6333  — vector store (4 collections: personnel, nato, mpep, documents)
- Postgres        :5432  — docstore (chunks+titles, collections, query_log partitioned)
- LiteLLM         :4000  — model proxy
- Open WebUI      :3000  — chat interface

Key architectural decisions:
- Collections split: personnel/nato/mpep/documents — separate Qdrant collections
  per domain. Reranker scores improved dramatically after this split.
- Title hydration: chunks have title column. ragpipe surfaces titles in
  rag_metadata.cited_chunks as objects {id, title, source}. System prompt
  instructs model to cite by title in prose while emitting [doc_id:chunk_id].
- Citation format: [doc_id:chunk_id] e.g. [133abba5-9eeb-5a99-8a5c:2]
  NOT [doc_id:133abba5...:chunk_id:2] — the verbose format is a bug.
- Grounding classification: corpus | general | mixed
- Hot-reload: POST /admin/reload-routes and POST /admin/reload-prompt
  avoid restarts for config changes. Use these instead of restarting ragpipe.
- MXR cache: ORT_MIGRAPHX_MODEL_CACHE_PATH env var enables caching.
  Warm start is ~6 seconds. Cold start (empty cache) is ~3m53s.
- LLM model: Qwen3-32B dense Q4_K_M (~19GB GTT). 32B fully activated
  parameters. Use /nothink flag for structured output tasks to prevent
  thinking mode consuming all output tokens.
- Qdrant IPv4: always use curl -4 or set QDRANT__SERVICE__HOST=:: in quadlet.
  Qdrant binds IPv4 only; Fedora resolves localhost to ::1 by default.
- Phase 0 Ragas baseline (ragprobe PR #11):
    Faithfulness: 0.700 | Answer Relevance: 0.843
    Context Precision: 0.714 | Context Recall: 0.250
  Personnel route strongest (F=0.967). MPEP/patent weakest (F=0.333).
  CRAG implementation (Phase 1) targets MPEP improvement.
