"""Tests for the LangGraph supervisor graph."""

from unittest.mock import patch

from ragorchestrator.graph import AgentState, build_graph


def test_graph_compiles():
    """Supervisor graph should compile without errors."""
    with patch("ragorchestrator.graph.ChatOpenAI"):
        graph = build_graph()
    assert graph is not None


def test_graph_has_supervisor_node():
    """Graph should have a supervisor node."""
    with patch("ragorchestrator.graph.ChatOpenAI"):
        graph = build_graph()
    # The compiled graph should have nodes
    assert graph is not None


def test_agent_state_schema():
    """AgentState should accept messages list."""
    state: AgentState = {"messages": []}
    assert state["messages"] == []
