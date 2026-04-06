"""LangGraph supervisor graph with Self-RAG reflection and multi-pass retrieval.

Features:
1. Adaptive complexity routing: SIMPLE queries go direct to ragpipe
2. Self-RAG reflection: grades generation for groundedness and usefulness
3. Multi-pass retrieval: decomposes complex queries into sub-queries,
   retrieves in parallel, merges and deduplicates results

Architecture:
    supervisor → should_retrieve → tools (single ragpipe call)
                              ↓
                         generate → reflect
                              ↓
              ┌─────────────────────────┼─────────────────────────┐
              ↓                         ↓                         ↓
          grounded                  useful                  ungrounded/not_useful
              ↓                         ↓                         ↓
            END                       END                    re-generate or re-retrieve
"""

import asyncio
import logging
import os
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ragorchestrator.multipass import decompose_query, deduplicate_documents
from ragorchestrator.reflection import ReflectionGrade, grade_answer, grade_hallucination
from ragorchestrator.tools.ragpipe_tool import ragpipe_retrieval
from ragorchestrator.tools.web_search_tool import get_web_search_tool

log = logging.getLogger(__name__)

MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "model.file")
MAX_RETRIES = 2

SUPERVISOR_PROMPT = """You are a helpful assistant with access to a document corpus via the ragpipe_retrieval tool.

For questions about internal documents, personnel, patents, defense analysis, or domain-specific knowledge:
- Always call ragpipe_retrieval first
- Use the returned answer and citations in your response
- Preserve citation format [doc_id:chunk_id] exactly as returned

If ragpipe_retrieval returns grounding='general' (no corpus match) and a web search tool is available:
- Call the web search tool to find current information
- Synthesize web results into a clear answer with source attribution

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


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    generation: str
    documents: list
    loop_count: int


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


def _build_tools():
    """Build tool list with optional web search."""
    tools = [ragpipe_retrieval]

    web_search = get_web_search_tool()
    if web_search:
        tools.append(web_search)
        log.info("Web search tool enabled (Tavily)")
    else:
        log.info("Web search tool disabled — ragpipe-only mode")

    return tools


def supervisor(state: AgentState) -> dict:
    messages = state["messages"]
    question = _extract_question(messages)
    if not state.get("question"):
        state["question"] = question

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SUPERVISOR_PROMPT), *list(messages)]

    tools = _build_tools()
    llm = _get_llm().bind_tools(tools)
    response = llm.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    return {}


async def multi_tools_node(state: AgentState) -> dict:
    """Retrieve documents for multiple sub-queries in parallel.

    Takes sub_queries from state, calls ragpipe for each in parallel,
    collects and deduplicates results.
    """
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {}

    ragpipe_url = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
    ragpipe_token = os.environ.get("RAGPIPE_ADMIN_TOKEN", "")

    async def retrieve_one(query: str) -> list[dict]:
        headers = {"Content-Type": "application/json"}
        if ragpipe_token:
            headers["Authorization"] = f"Bearer {ragpipe_token}"

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{ragpipe_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                docs = data.get("rag_metadata", {}).get("cited_chunks", [])
                return docs
        except Exception as e:
            log.warning("Sub-query retrieval failed for '%s': %s", query, e)
            return []

    results = await asyncio.gather(*[retrieve_one(q) for q in sub_queries])
    all_docs = []
    for docs in results:
        all_docs.extend(docs)

    unique_docs = deduplicate_documents(all_docs)
    log.info("Multi-pass: retrieved %d docs, %d unique after dedup", len(all_docs), len(unique_docs))

    return {"documents": unique_docs, "all_documents": unique_docs}


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
            log.info("Self-RAG: max retries reached")
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


async def decompose(state: AgentState) -> dict:
    """Decompose complex query into sub-queries."""
    question = state.get("question", "")
    sub_queries = await decompose_query(question)
    log.info("Multi-pass: decomposed into %d sub-queries", len(sub_queries))
    return {
        "sub_queries": sub_queries,
        "messages": [AIMessage(content=f"[Multi-pass] Decomposed into {len(sub_queries)} sub-queries")],
    }


def should_retrieve(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "decompose"
    return "generate"


def should_use_multipass(state: AgentState) -> str:
    sub_queries = state.get("sub_queries", [])
    if len(sub_queries) > 1:
        return "multi_tools"
    return "tools"


def should_regenerate(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and "[Self-RAG]" in last.content:
        if "regenerating" in last.content.lower():
            return "generate"
        elif "re-retriev" in last.content.lower():
            return "decompose"
    return END


def build_graph():
    tools = _build_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("decompose", decompose)
    graph.add_node("tools", tool_node)
    graph.add_node("multi_tools", multi_tools_node)
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", should_retrieve, {"decompose": "decompose", "generate": "generate"})

    graph.add_conditional_edges("decompose", should_use_multipass, {"multi_tools": "multi_tools", "tools": "tools"})

    graph.add_edge("tools", "generate")
    graph.add_edge("multi_tools", "generate")
    graph.add_edge("generate", "reflect")

    graph.add_conditional_edges(
        "reflect", should_regenerate, {"generate": "generate", "decompose": "decompose", END: END}
    )

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("Supervisor graph compiled")
    return _graph
