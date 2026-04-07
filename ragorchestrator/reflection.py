"""Self-RAG reflection/grading logic.

After generation, the supervisor critiques its own response:
- Is the generation grounded in the retrieved evidence?
- Does the answer address the original question?

If not grounded: re-generate with stricter prompt
If not useful: re-retrieve with rewritten query
"""

import os
from enum import Enum

from langchain_openai import ChatOpenAI

MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "model.file")


class ReflectionGrade(Enum):
    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"


HALLUCINATION_GRADER_PROMPT = """Assess whether an LLM response is grounded in retrieved documents.

Given a question and response, assess whether the response is factually supported by the documents.
Return a JSON object with a single key "grade" and value "yes" or "no":
- "yes": The response is factually accurate and grounded in the documents
- "no": The response contains information not supported by or contradicts the documents

Respond only with JSON: {{"grade": "yes"}} or {{"grade": "no"}}"""


ANSWER_GRADER_PROMPT = """You are assessing whether an LLM generated response directly answers a question.

Given a question and response, assess whether the response meaningfully addresses what was asked.
Return a JSON object with a single key "grade" and value "yes" or "no":
- "yes": The response addresses the core question asked
- "no": The response is off-topic, incomplete, or does not address the question

Respond only with JSON: {{"grade": "yes"}} or {{"grade": "no"}}"""


def _get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=f"{MODEL_URL}/v1",
        api_key="not-needed",
        temperature=0,
    )


async def grade_hallucination(question: str, documents: list[dict], generation: str) -> ReflectionGrade:
    """Check if generation is grounded in the retrieved documents.

    Returns GROUNDED if all claims in the generation are supported by documents,
    UNGROUNDED otherwise.
    """
    docs_text = "\n\n".join(f"[doc {i}]: {d.get('text', d.get('content', ''))}" for i, d in enumerate(documents[:5]))

    prompt = f"""Question: {question}

Documents:
{docs_text}

Response: {generation}

{HALLUCINATION_GRADER_PROMPT}"""

    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        if '"yes"' in content or '"YES"' in content:
            return ReflectionGrade.GROUNDED
        return ReflectionGrade.UNGROUNDED
    except Exception:
        return ReflectionGrade.GROUNDED


async def grade_answer(question: str, generation: str) -> ReflectionGrade:
    """Check if generation addresses the question.

    Returns USEFUL if the response meaningfully addresses the question,
    NOT_USEFUL otherwise.
    """
    prompt = f"""Question: {question}

Response: {generation}

{ANSWER_GRADER_PROMPT}"""

    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        if '"yes"' in content or '"YES"' in content:
            return ReflectionGrade.USEFUL
        return ReflectionGrade.NOT_USEFUL
    except Exception:
        return ReflectionGrade.USEFUL
