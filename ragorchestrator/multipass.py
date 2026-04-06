"""Multi-pass retrieval with query decomposition.

For complex queries, decomposes into sub-queries and retrieves
for each in parallel, then merges and deduplicates results.
"""

import json
import logging
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "model.file")
MAX_SUB_QUERIES = 3


DECOMPOSITION_PROMPT = """Break down the following complex question into 2-3 simpler sub-questions.

Return ONLY a JSON array of strings (the sub-questions). No explanation, no markdown.

Example output: ["What is X?", "How does Y affect Z?", "What are the benefits of Z?"]

Question: '{question}'

JSON array of sub-questions:"""


def _get_llm(temperature=0):
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=f"{MODEL_URL}/v1",
        api_key="not-needed",
        temperature=temperature,
    )


async def decompose_query(question: str) -> list[str]:
    """Decompose a complex question into sub-queries.

    Uses LLM to generate 2-3 simpler sub-questions that together
    answer the original complex question.
    """
    try:
        llm = _get_llm(temperature=0)
        response = await llm.ainvoke([HumanMessage(content=DECOMPOSITION_PROMPT.format(question=question))])
        content = response.content.strip()

        sub_queries = json.loads(content)
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            return sub_queries[:MAX_SUB_QUERIES]
        log.warning("Unexpected decomposition format: %s", content)
        return [question]
    except (json.JSONDecodeError, Exception) as e:
        log.warning("Query decomposition failed: %s, falling back to original", e)
        return [question]


def deduplicate_documents(documents: list[dict]) -> list[dict]:
    """Deduplicate documents by (doc_id, chunk_id).

    Maintains order - first occurrence of each doc_id:chunk_id is kept.
    """
    seen: set[tuple[str, int]] = set()
    unique_docs: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id", "")
        chunk_id = doc.get("chunk_id", 0)
        key = (doc_id, chunk_id)
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    log.info("Deduplicated %d docs to %d", len(documents), len(unique_docs))
    return unique_docs
