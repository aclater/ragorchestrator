# Architecture

ragorchestrator is a LangGraph-based supervisor agent that sits above [ragpipe](https://github.com/aclater/ragpipe), adding adaptive complexity routing, multi-pass retrieval, and Self-RAG reflection to the RAG pipeline.

## System context

```
Client → ragorchestrator (:8095) → LangGraph supervisor
                                        ↓
                                  ragpipe (:8090) ← corpus retrieval
                                        ↓
                                  LLM (:8080) ← generation + reflection
```

ragorchestrator exposes the same OpenAI-compatible `/v1/chat/completions` API as ragpipe. Clients can switch between the two by changing the port — no other changes needed.

## LangGraph state machine

The supervisor graph has 6 nodes connected by conditional edges:

```
                    ┌─────────────┐
                    │  supervisor  │  ← entry point
                    └──────┬──────┘
                           │
                    should_retrieve?
                    ╱              ╲
              tool_calls        no tools
                 ╱                    ╲
        ┌──────────┐          ┌──────────┐
        │ decompose │          │ generate  │
        └─────┬────┘          └─────┬────┘
              │                     │
       should_use_multipass?        │
        ╱              ╲            │
   ┌──────────┐  ┌─────────┐       │
   │multi_tools│  │  tools   │      │
   └─────┬────┘  └────┬────┘       │
         │            │             │
         └──────┬─────┘             │
                ↓                   │
         ┌──────────┐               │
         │ generate  │ ←────────────┘
         └─────┬────┘
               │
         ┌──────────┐
         │  reflect  │  ← Self-RAG grading
         └─────┬────┘
               │
        should_regenerate?
        ╱       │        ╲
   generate  decompose    END
   (retry)   (re-retrieve)
```

### Nodes

| Node | Purpose |
|------|---------|
| `supervisor` | LLM with bound tools decides whether to call ragpipe or respond directly |
| `decompose` | Splits complex queries into 2-3 sub-queries via LLM |
| `tools` | Single ragpipe call via LangGraph ToolNode |
| `multi_tools` | Parallel ragpipe calls for each sub-query (asyncio.gather) |
| `generate` | LLM generates answer from retrieved documents |
| `reflect` | Self-RAG grading: hallucination + usefulness checks |

### Conditional edges

| Edge | Condition | Routes to |
|------|-----------|-----------|
| `should_retrieve` | supervisor made tool_calls → decompose; otherwise → generate |
| `should_use_multipass` | >1 sub-query → multi_tools; otherwise → tools |
| `should_regenerate` | UNGROUNDED → generate (retry); NOT_USEFUL → decompose (re-retrieve); else → END |

### State schema

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangChain message history
    question: str                            # extracted user question
    generation: str                          # current generation text
    documents: list                          # retrieved document chunks
    loop_count: int                          # retry counter (max 2)
```

## Request flow

### Simple queries (SIMPLE complexity)

```
POST /v1/chat/completions
  → classify("who is Adam?") → SIMPLE
  → _simple_path(): direct HTTP call to ragpipe
  → return ragpipe response with rag_metadata
```

Latency: ~2-5s (single ragpipe call)

### Complex queries (COMPLEX/EXTERNAL complexity)

```
POST /v1/chat/completions
  → classify("compare NATO article 5 with patent claims") → COMPLEX
  → _agentic_path(): LangGraph ainvoke()
    → supervisor: LLM decides to call ragpipe_retrieval
    → decompose: split into sub-queries
    → multi_tools: parallel ragpipe calls
    → generate: synthesize answer from documents
    → reflect: grade hallucination + usefulness
    → (retry if needed, max 2 iterations)
  → extract rag_metadata from tool results
  → return OpenAI-compatible response
```

Latency: ~30-120s (5+ sequential LLM calls with Qwen3-32B)

## Metrics

Prometheus metrics exposed at `/metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `ragorchestrator_queries_total` | Counter | status | Total queries processed |
| `ragorchestrator_query_latency_seconds` | Histogram | | Query latency |
| `ragorchestrator_tool_calls_total` | Counter | tool | Tool calls by supervisor |
| `ragorchestrator_complexity_classified_total` | Counter | complexity | Queries by complexity class |
