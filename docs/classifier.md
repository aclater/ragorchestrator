# Adaptive Complexity Classifier

Deterministic keyword-based query classifier that routes queries to the appropriate processing path.

## Complexity levels

| Level | Path | Description |
|-------|------|-------------|
| `SIMPLE` | Direct ragpipe call | Factual lookups, single-topic questions |
| `COMPLEX` | Full LangGraph loop | Multi-topic analysis, comparisons, reasoning |
| `EXTERNAL` | LangGraph + web search | Current events, stock prices, weather |

## Classification priority

The classifier evaluates patterns in order — first match wins:

1. **EXTERNAL indicators**: `news`, `weather`, `stock price`, `celebrity`, `competitor`
2. **COMPLEX indicators**: `compare`, `analyze`, `pros/cons`, `impact`, `relationship`
3. **Length threshold**: queries >200 characters → COMPLEX
4. **SIMPLE indicators**: question starters (`who`, `what`, `when`), commands (`list`, `find`, `get`)
5. **Short questions**: <=5 words with `?` → SIMPLE
6. **Non-alpha**: no alphabetic characters → SIMPLE
7. **Default**: COMPLEX

## Examples

```
"Who is Adam Clater?"                              → SIMPLE
"What is NATO article 5?"                          → SIMPLE
"List all employees"                               → SIMPLE
"Compare NATO article 5 with patent claims"        → COMPLEX
"Analyze the pros and cons of this approach"        → COMPLEX
"What is the latest news?"                         → EXTERNAL
"hello"                                            → COMPLEX (default)
```

## Tuning

The classifier uses regex patterns defined in `classifier.py`. To adjust:

- Add patterns to `SIMPLE_INDICATORS` to route more queries to the fast path
- Add patterns to `COMPLEX_INDICATORS` to trigger the agentic loop
- Adjust `LLM_THRESHOLD_QUERY_LENGTH` (default 200) for the length heuristic

The `complexity` field in the API response shows which path was taken, useful for debugging classification behavior.
