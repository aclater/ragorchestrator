# Self-RAG Reflection

After generating an answer, the supervisor grades its own response for groundedness and usefulness. If the response fails either check, it retries — up to 2 times.

## Grading pipeline

```
generate → reflect
              │
        grade_hallucination(question, documents, generation)
              │
         GROUNDED?
         ╱        ╲
       yes         no → regenerate (stricter prompt)
        │
   grade_answer(question, generation)
        │
      USEFUL?
      ╱      ╲
    yes       no → re-retrieve (rewritten query)
     │
    END
```

## Grades

| Grade | Meaning | Action |
|-------|---------|--------|
| `GROUNDED` | Response supported by retrieved documents | Continue to answer grading |
| `UNGROUNDED` | Response contains claims not in documents | Regenerate with stricter prompt |
| `USEFUL` | Response addresses the original question | Return to client |
| `NOT_USEFUL` | Response is off-topic or incomplete | Re-retrieve with new sub-queries |

## Grading prompts

Both graders use the LLM with temperature=0 and structured JSON output:

- **Hallucination grader**: "Is the response factually supported by the documents? Return `{\"grade\": \"yes\"}` or `{\"grade\": \"no\"}`"
- **Answer grader**: "Does the response meaningfully address the question? Return `{\"grade\": \"yes\"}` or `{\"grade\": \"no\"}`"

## Short-circuit on grounding=general

When ragpipe classifies the query as `grounding=general` (no corpus match), the hallucination grader is skipped entirely — there are no documents to grade against. The reflect node triggers re-retrieval directly, saving one LLM call.

## Retry limits

`MAX_RETRIES = 2` — the reflect node increments `loop_count` on each iteration. After 2 retries, the current generation is returned regardless of grade. This prevents infinite loops on genuinely unanswerable queries.

## Error handling

If either grader call fails (LLM timeout, malformed response), the grade defaults to the optimistic case:
- Hallucination grader failure → `GROUNDED`
- Answer grader failure → `USEFUL`

This ensures errors don't cause unnecessary retries or block response delivery.
