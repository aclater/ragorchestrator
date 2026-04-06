"""LangGraph supervisor graph for agentic RAG orchestration.

Implements an adaptive supervisor that routes queries to tools (ragpipe,
web search, etc.) based on query complexity and content. The supervisor
decides which tool to call, synthesizes results, and can retry with
different tools if the first attempt fails.

Architecture:
    Client → supervisor → [ragpipe_retrieval, ...] → synthesis → response

The supervisor uses the local LLM (same model as ragpipe) for routing
decisions. No external API calls — fully sovereign deployment.
"""

import logging
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ragorchestrator.tools.ragpipe_tool import ragpipe_retrieval

log = logging.getLogger(__name__)

MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "model.file")

SUPERVISOR_PROMPT = """You are a helpful assistant with access to a document corpus via the ragpipe_retrieval tool.

For questions about internal documents, personnel, patents, defense analysis, or domain-specific knowledge:
- Always call ragpipe_retrieval first
- Use the returned answer and citations in your response
- Preserve citation format [doc_id:chunk_id] exactly as returned

For general knowledge questions (math, science, common facts):
- Answer directly without calling tools

Keep responses concise and accurate. Never fabricate citations."""


# ── State ────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """State schema for the supervisor graph."""

    messages: Annotated[list, add_messages]


# ── Graph construction ───────────────────────────────────────────────────────


def build_graph():
    """Build and compile the supervisor LangGraph.

    Returns a compiled graph that can be invoked with AgentState.
    """
    tools = [ragpipe_retrieval]

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=f"{MODEL_URL}/v1",
        api_key="not-needed",
        temperature=0,
    ).bind_tools(tools)

    tool_node = ToolNode(tools)

    def supervisor(state: AgentState) -> dict:
        """Supervisor node — calls LLM with tool bindings."""
        messages = state["messages"]
        # Prepend system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SUPERVISOR_PROMPT), *list(messages)]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Edge: decide whether to call tools or finish."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "supervisor")

    return graph.compile()


# Module-level compiled graph (lazy init)
_graph = None


def get_graph():
    """Get or create the compiled supervisor graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("Supervisor graph compiled")
    return _graph
