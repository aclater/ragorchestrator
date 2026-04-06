"""Tests for the Self-RAG reflection module."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragorchestrator.graph import _extract_grounding, reflect
from ragorchestrator.reflection import ReflectionGrade


class TestReflectionGrade:
    def test_reflection_grade_values(self):
        assert ReflectionGrade.GROUNDED.value == "grounded"
        assert ReflectionGrade.UNGROUNDED.value == "ungrounded"
        assert ReflectionGrade.USEFUL.value == "useful"
        assert ReflectionGrade.NOT_USEFUL.value == "not_useful"


class TestExtractGrounding:
    def test_extracts_corpus(self):
        msgs = [
            HumanMessage(content="test"),
            ToolMessage(
                content=json.dumps({"grounding": "corpus", "answer": "x"}),
                name="ragpipe_retrieval",
                tool_call_id="tc1",
            ),
        ]
        assert _extract_grounding(msgs) == "corpus"

    def test_extracts_general(self):
        msgs = [
            ToolMessage(
                content=json.dumps({"grounding": "general", "answer": "x"}),
                name="ragpipe_retrieval",
                tool_call_id="tc1",
            ),
        ]
        assert _extract_grounding(msgs) == "general"

    def test_extracts_mixed(self):
        msgs = [
            ToolMessage(
                content=json.dumps({"grounding": "mixed", "answer": "x"}),
                name="ragpipe_retrieval",
                tool_call_id="tc1",
            ),
        ]
        assert _extract_grounding(msgs) == "mixed"

    def test_returns_none_when_no_tool_message(self):
        msgs = [HumanMessage(content="test"), AIMessage(content="answer")]
        assert _extract_grounding(msgs) is None

    def test_returns_none_on_invalid_json(self):
        msgs = [
            ToolMessage(
                content="not json",
                name="ragpipe_retrieval",
                tool_call_id="tc1",
            ),
        ]
        assert _extract_grounding(msgs) is None

    def test_uses_last_ragpipe_message(self):
        """When multiple ragpipe calls exist, use the last one."""
        msgs = [
            ToolMessage(
                content=json.dumps({"grounding": "general", "answer": "x"}),
                name="ragpipe_retrieval",
                tool_call_id="tc1",
            ),
            ToolMessage(
                content=json.dumps({"grounding": "corpus", "answer": "y"}),
                name="ragpipe_retrieval",
                tool_call_id="tc2",
            ),
        ]
        assert _extract_grounding(msgs) == "corpus"

    def test_ignores_non_ragpipe_tools(self):
        msgs = [
            ToolMessage(
                content=json.dumps({"grounding": "general"}),
                name="web_search",
                tool_call_id="tc1",
            ),
        ]
        assert _extract_grounding(msgs) is None


class TestReflectShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuits_on_grounding_general(self):
        """reflect() should skip hallucination grader when grounding=general."""
        state = {
            "question": "What is the weather?",
            "generation": "It is sunny.",
            "documents": [],
            "loop_count": 0,
            "messages": [
                HumanMessage(content="What is the weather?"),
                ToolMessage(
                    content=json.dumps({"grounding": "general", "answer": "It is sunny."}),
                    name="ragpipe_retrieval",
                    tool_call_id="tc1",
                ),
            ],
        }

        with patch("ragorchestrator.graph.grade_hallucination") as mock_hall:
            result = await reflect(state)
            mock_hall.assert_not_called()

        assert result["loop_count"] == 1
        assert "[Self-RAG]" in result["messages"][0].content
        assert "re-retriev" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_short_circuit_respects_max_retries(self):
        """reflect() should not re-retrieve when max retries reached on general grounding."""
        state = {
            "question": "What is the weather?",
            "generation": "It is sunny.",
            "documents": [],
            "loop_count": 2,
            "messages": [
                ToolMessage(
                    content=json.dumps({"grounding": "general", "answer": "x"}),
                    name="ragpipe_retrieval",
                    tool_call_id="tc1",
                ),
            ],
        }

        with patch("ragorchestrator.graph.grade_hallucination") as mock_hall:
            result = await reflect(state)
            mock_hall.assert_not_called()

        assert result["loop_count"] == 3
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_no_short_circuit_on_corpus(self):
        """reflect() should call hallucination grader when grounding=corpus."""
        state = {
            "question": "Who is Adam?",
            "generation": "Adam is a person.",
            "documents": [{"text": "Adam is a person."}],
            "loop_count": 0,
            "messages": [
                ToolMessage(
                    content=json.dumps({"grounding": "corpus", "answer": "Adam is a person."}),
                    name="ragpipe_retrieval",
                    tool_call_id="tc1",
                ),
            ],
        }

        mock_hall = AsyncMock(return_value=ReflectionGrade.GROUNDED)
        mock_answer = AsyncMock(return_value=ReflectionGrade.USEFUL)
        with (
            patch("ragorchestrator.graph.grade_hallucination", mock_hall),
            patch("ragorchestrator.graph.grade_answer", mock_answer),
        ):
            result = await reflect(state)
            mock_hall.assert_called_once()

        assert result["loop_count"] == 1

    @pytest.mark.asyncio
    async def test_no_short_circuit_on_mixed(self):
        """reflect() should call hallucination grader when grounding=mixed."""
        state = {
            "question": "Compare X and Y",
            "generation": "X and Y differ in...",
            "documents": [{"text": "X is..."}],
            "loop_count": 0,
            "messages": [
                ToolMessage(
                    content=json.dumps({"grounding": "mixed", "answer": "X and Y differ..."}),
                    name="ragpipe_retrieval",
                    tool_call_id="tc1",
                ),
            ],
        }

        mock_hall = AsyncMock(return_value=ReflectionGrade.GROUNDED)
        mock_answer = AsyncMock(return_value=ReflectionGrade.USEFUL)
        with (
            patch("ragorchestrator.graph.grade_hallucination", mock_hall),
            patch("ragorchestrator.graph.grade_answer", mock_answer),
        ):
            await reflect(state)
            mock_hall.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_short_circuit_when_no_grounding(self):
        """reflect() should call hallucination grader when no ragpipe tool message exists."""
        state = {
            "question": "What is X?",
            "generation": "X is...",
            "documents": [{"text": "X is..."}],
            "loop_count": 0,
            "messages": [HumanMessage(content="What is X?")],
        }

        mock_hall = AsyncMock(return_value=ReflectionGrade.GROUNDED)
        mock_answer = AsyncMock(return_value=ReflectionGrade.USEFUL)
        with (
            patch("ragorchestrator.graph.grade_hallucination", mock_hall),
            patch("ragorchestrator.graph.grade_answer", mock_answer),
        ):
            await reflect(state)
            mock_hall.assert_called_once()
