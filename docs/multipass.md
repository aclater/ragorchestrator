# Multi-Pass Retrieval

Complex queries are decomposed into simpler sub-queries, each retrieved in parallel from ragpipe, then merged and deduplicated before generation.

## Flow

```
decompose_query("compare personnel roles with NATO obligations")
  → ["What are the personnel roles?", "What are the NATO obligations?"]

multi_tools_node:
  → asyncio.gather(
      ragpipe("What are the personnel roles?"),
      ragpipe("What are the NATO obligations?"),
    )
  → merge cited_chunks from all responses
  → deduplicate by (doc_id, chunk_id), preserving first occurrence

generate:
  → synthesize answer from merged document set
```

## Decomposition

The LLM decomposes complex queries into 2-3 sub-questions using a structured JSON prompt. The response must be a JSON array of strings:

```json
["What are the personnel roles?", "What are the NATO obligations?"]
```

`MAX_SUB_QUERIES = 3` — additional sub-queries are truncated.

If decomposition fails (LLM returns invalid JSON, timeout), the original query is used as the sole sub-query — single-pass retrieval as fallback.

## Parallel retrieval

Sub-queries are sent to ragpipe concurrently via `asyncio.gather`. Each call hits ragpipe's full pipeline (semantic routing, Qdrant search, docstore hydration, reranking). Individual failures return empty results without blocking other sub-queries.

## Deduplication

Documents from all sub-queries are merged and deduplicated by `(doc_id, chunk_id)`. First occurrence is kept, preserving the ordering from the most relevant sub-query.

## When multi-pass activates

The `should_use_multipass` edge checks the number of sub-queries:
- 1 sub-query → single `tools` node (standard ToolNode execution)
- 2+ sub-queries → `multi_tools` node (parallel retrieval)
