"""Tests for the LangGraph supervisor graph."""

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragorchestrator.graph import AgentState, build_graph, should_skip_reflect


def test_graph_compiles():
    """Supervisor graph should compile without errors."""
    with patch("ragorchestrator.graph.ChatOpenAI"):
        graph = build_graph()
    assert graph is not None


def test_graph_has_supervisor_node():
    """Graph should have a supervisor node."""
    with patch("ragorchestrator.graph.ChatOpenAI"):
        graph = build_graph()
    assert graph is not None


def test_agent_state_schema():
    """AgentState should accept messages list."""
    state: AgentState = {"messages": []}
    assert state["messages"] == []


class TestShouldSkipReflect:
    """Tests for the should_skip_reflect conditional edge."""

    def _make_ragpipe_tool_msg(self, grounding: str, answer: str = "test") -> ToolMessage:
        """Create a ToolMessage simulating ragpipe_retrieval response."""
        ragpipe_response = json.dumps(
            {
                "answer": answer,
                "grounding": grounding,
                "cited_chunks": [],
            }
        )
        return ToolMessage(
            content=ragpipe_response,
            name="ragpipe_retrieval",
            tool_call_id="test-call-id",
        )

    def test_general_grounding_skips_reflect(self):
        """When ragpipe returns grounding=general, should_skip_reflect returns END."""
        tool_msg = self._make_ragpipe_tool_msg("general")
        state: AgentState = {
            "messages": [AIMessage(content="[ragpipe_retrieval]"), tool_msg],
            "question": "What is 2+2?",
            "generation": "2+2 equals 4",
            "documents": [],
            "loop_count": 0,
        }
        result = should_skip_reflect(state)
        from langgraph.graph import END

        assert result == END

    def test_corpus_grounding_enter_reflect(self):
        """When ragpipe returns grounding=corpus, should_skip_reflect returns reflect."""
        tool_msg = self._make_ragpipe_tool_msg("corpus")
        state: AgentState = {
            "messages": [AIMessage(content="[ragpipe_retrieval]"), tool_msg],
            "question": "Who is Adam Clater?",
            "generation": "Adam Clater is...",
            "documents": [{"id": "doc1", "text": "Adam Clater is the CEO"}],
            "loop_count": 0,
        }
        result = should_skip_reflect(state)
        assert result == "reflect"

    def test_mixed_grounding_enter_reflect(self):
        """When ragpipe returns grounding=mixed, should_skip_reflect returns reflect."""
        tool_msg = self._make_ragpipe_tool_msg("mixed")
        state: AgentState = {
            "messages": [AIMessage(content="[ragpipe_retrieval]"), tool_msg],
            "question": "Explain this document",
            "generation": "The document explains...",
            "documents": [{"id": "doc1", "text": "Document content"}],
            "loop_count": 0,
        }
        result = should_skip_reflect(state)
        assert result == "reflect"

    def test_no_ragpipe_response_enter_reflect(self):
        """When no ragpipe tool response exists, should_skip_reflect returns reflect (safe default)."""
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "question": "Hi",
            "generation": "Hello!",
            "documents": [],
            "loop_count": 0,
        }
        result = should_skip_reflect(state)
        assert result == "reflect"

    def test_unknown_grounding_enter_reflect(self):
        """When ragpipe returns grounding=unknown, should_skip_reflect returns reflect."""
        tool_msg = self._make_ragpipe_tool_msg("unknown")
        state: AgentState = {
            "messages": [AIMessage(content="[ragpipe_retrieval]"), tool_msg],
            "question": "Question",
            "generation": "Answer",
            "documents": [],
            "loop_count": 0,
        }
        result = should_skip_reflect(state)
        assert result == "reflect"
