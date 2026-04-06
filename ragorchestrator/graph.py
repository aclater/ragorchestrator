"""LangGraph supervisor graph with Self-RAG reflection.

After generation, the supervisor critiques its own response:
- Is the generation grounded in the retrieved evidence?
- Does the answer address the original question?

If not grounded: re-generate with stricter prompt (max 2 retries)
If not useful: re-retrieve with rewritten query

Architecture:
    supervisor → tools (ragpipe_retrieval) → generate → reflect
                                                        ↓
                              ┌─────────────────────────┼─────────────────────────┐
                              ↓                         ↓                         ↓
                          grounded                  useful                  ungrounded/not_useful
                              ↓                         ↓                         ↓
                            END                       END                    re-generate or re-retrieve
"""

import json
import logging
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ragorchestrator.reflection import ReflectionGrade, grade_answer, grade_hallucination
from ragorchestrator.tools.ragpipe_tool import ragpipe_retrieval

log = logging.getLogger(__name__)

MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "model.file")
MAX_RETRIES = 2

SUPERVISOR_PROMPT = """You are a helpful assistant with access to a document corpus via the ragpipe_retrieval tool.

For questions about internal documents, personnel, patents, defense analysis, or domain-specific knowledge:
- Always call ragpipe_retrieval first
- Use the returned answer and citations in your response
- Preserve citation format [doc_id:chunk_id] exactly as returned

For general knowledge questions (math, science, common facts):
- Answer directly without calling tools

Keep responses concise and accurate. Never fabricate citations."""


GENERATION_PROMPT = """You are a helpful assistant answering questions using retrieved documents.

Given a question and relevant documents, provide an accurate and concise answer.
- Only use information from the provided documents
- Cite specific documents when making claims using [doc_id:chunk_id] format
- If the documents don't contain enough information, say so

Question: {question}

Documents:
{documents}

Answer:"""


REGENERATION_PROMPT = """You are a helpful assistant answering questions using retrieved documents.

You generated an answer that was assessed as not fully grounded in the documents.
Revise your answer to strictly follow the evidence in the documents.
Do not add information not supported by the documents.

Question: {question}

Documents:
{documents}

Previous answer (ignore if not supported by documents): {previous_answer}

Revised answer (grounded in documents only):"""


# ── State ────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    generation: str
    documents: list
    loop_count: int


# ── Graph construction ────────────────────────────────────────────────────────


def _get_llm(temperature=0):
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=f"{MODEL_URL}/v1",
        api_key="not-needed",
        temperature=temperature,
    )


def _extract_question(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _extract_last_tool_result(messages) -> tuple[list, str]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "")
                if tool_name == "ragpipe_retrieval":
                    tool_call_id = tc.get("id", "")
                    for reply_msg in messages:
                        if hasattr(reply_msg, "tool_call_id") and reply_msg.tool_call_id == tool_call_id:
                            try:
                                data = json.loads(reply_msg.content)
                                docs = data.get("cited_chunks", [])
                                return docs, data.get("answer", "")
                            except (json.JSONDecodeError, TypeError):
                                return [], reply_msg.content
    return [], ""


def supervisor(state: AgentState) -> dict:
    messages = state["messages"]
    question = _extract_question(messages)
    if not state.get("question"):
        state["question"] = question

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SUPERVISOR_PROMPT), *list(messages)]

    tools = [ragpipe_retrieval]
    llm = _get_llm().bind_tools(tools)
    response = llm.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    return {}


def generate(state: AgentState) -> dict:
    question = state.get("question", "")
    documents = state.get("documents", [])
    loop_count = state.get("loop_count", 0)

    docs_text = "\n\n".join(
        f"[doc {i}]: {d.get('title', 'Untitled')} — {d.get('source', '')}\n{d.get('text', '')}"
        for i, d in enumerate(documents[:5])
    )

    if loop_count > 0 and state.get("generation"):
        prompt = REGENERATION_PROMPT.format(
            question=question,
            documents=docs_text,
            previous_answer=state["generation"],
        )
    else:
        prompt = GENERATION_PROMPT.format(question=question, documents=docs_text)

    llm = _get_llm(temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    generation = response.content if hasattr(response, "content") else str(response)

    return {"generation": generation, "loop_count": loop_count}


async def reflect(state: AgentState) -> dict:
    question = state.get("question", "")
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    loop_count = state.get("loop_count", 0)

    hallucination = await grade_hallucination(question, documents, generation)
    log.info("Self-RAG reflection: hallucination=%s", hallucination.value)

    if hallucination == ReflectionGrade.UNGROUNDED:
        if loop_count >= MAX_RETRIES:
            log.info("Self-RAG: max retries reached, returning best-effort response")
            return {"loop_count": loop_count + 1}
        return {
            "loop_count": loop_count + 1,
            "messages": [AIMessage(content="[Self-RAG] Response not grounded in documents, regenerating...")],
        }

    answer = await grade_answer(question, generation)
    log.info("Self-RAG reflection: answer_usefulness=%s", answer.value)

    if answer == ReflectionGrade.NOT_USEFUL:
        if loop_count >= MAX_RETRIES:
            return {"loop_count": loop_count + 1}
        return {
            "loop_count": loop_count + 1,
            "messages": [AIMessage(content="[Self-RAG] Response does not address question, re-retrieving...")],
        }

    return {"loop_count": loop_count + 1}


def should_retrieve(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "generate"


def should_regenerate(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and "[Self-RAG]" in last.content:
        if "regenerating" in last.content.lower():
            return "generate"
        elif "re-retriev" in last.content.lower():
            return "supervisor"
    return END


def build_graph():
    tools = [ragpipe_retrieval]
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("tools", tool_node)
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", should_retrieve, {"tools": "tools", "generate": "generate"})
    graph.add_edge("tools", "generate")
    graph.add_edge("generate", "reflect")

    graph.add_conditional_edges(
        "reflect", should_regenerate, {"generate": "generate", "supervisor": "supervisor", END: END}
    )

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("Self-RAG supervisor graph compiled")
    return _graph
