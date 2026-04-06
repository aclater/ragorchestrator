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
    def test_extracts_grounding_from_tool_message(self):
        messages = [
            HumanMessage(content="Who is CEO?"),
            ToolMessage(
                content=json.dumps({"answer": "...", "grounding": "corpus"}),
                tool_call_id="1",
            ),
        ]
        assert _extract_grounding(messages) == "corpus"

    def test_extracts_general_grounding(self):
        messages = [
            ToolMessage(
                content=json.dumps({"answer": "...", "grounding": "general"}),
                tool_call_id="1",
            ),
        ]
        assert _extract_grounding(messages) == "general"

    def test_returns_none_when_no_tool_messages(self):
        messages = [HumanMessage(content="hello")]
        assert _extract_grounding(messages) is None

    def test_returns_none_for_empty_messages(self):
        assert _extract_grounding([]) is None

    def test_returns_none_for_non_json_tool_message(self):
        messages = [
            ToolMessage(content="not json", tool_call_id="1"),
        ]
        assert _extract_grounding(messages) is None

    def test_returns_none_when_grounding_key_missing(self):
        messages = [
            ToolMessage(
                content=json.dumps({"answer": "..."}),
                tool_call_id="1",
            ),
        ]
        assert _extract_grounding(messages) is None

    def test_uses_last_tool_message(self):
        messages = [
            ToolMessage(
                content=json.dumps({"answer": "old", "grounding": "corpus"}),
                tool_call_id="1",
            ),
            ToolMessage(
                content=json.dumps({"answer": "new", "grounding": "general"}),
                tool_call_id="2",
            ),
        ]
        assert _extract_grounding(messages) == "general"


class TestReflectShortCircuit:
    @pytest.mark.asyncio
    async def test_general_grounding_skips_hallucination_grader(self):
        """When grounding=general, reflect should NOT call grade_hallucination."""
        state = {
            "messages": [
                HumanMessage(content="What is quantum computing?"),
                ToolMessage(
                    content=json.dumps({"answer": "...", "grounding": "general"}),
                    tool_call_id="1",
                ),
            ],
            "question": "What is quantum computing?",
            "generation": "Quantum computing uses qubits...",
            "documents": [],
            "loop_count": 0,
        }

        with (
            patch("ragorchestrator.graph.grade_hallucination") as mock_halluc,
            patch("ragorchestrator.graph.grade_answer") as mock_answer,
        ):
            result = await reflect(state)

            mock_halluc.assert_not_called()
            mock_answer.assert_not_called()

        assert result["loop_count"] == 1
        assert "[Self-RAG]" in result["messages"][0].content
        assert "re-retrieving" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_general_grounding_respects_max_retries(self):
        """When grounding=general and max retries reached, just increment loop_count."""
        state = {
            "messages": [
                ToolMessage(
                    content=json.dumps({"answer": "...", "grounding": "general"}),
                    tool_call_id="1",
                ),
            ],
            "question": "What is quantum computing?",
            "generation": "...",
            "documents": [],
            "loop_count": 2,  # MAX_RETRIES = 2
        }

        with (
            patch("ragorchestrator.graph.grade_hallucination") as mock_halluc,
            patch("ragorchestrator.graph.grade_answer") as mock_answer,
        ):
            result = await reflect(state)

            mock_halluc.assert_not_called()
            mock_answer.assert_not_called()

        assert result["loop_count"] == 3
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_corpus_grounding_uses_normal_path(self):
        """When grounding=corpus, reflect should call grade_hallucination as normal."""
        state = {
            "messages": [
                ToolMessage(
                    content=json.dumps({"answer": "...", "grounding": "corpus"}),
                    tool_call_id="1",
                ),
            ],
            "question": "Who is the CEO?",
            "generation": "The CEO is ...",
            "documents": [{"text": "The CEO is ..."}],
            "loop_count": 0,
        }

        with (
            patch(
                "ragorchestrator.graph.grade_hallucination",
                new_callable=AsyncMock,
                return_value=ReflectionGrade.GROUNDED,
            ) as mock_halluc,
            patch(
                "ragorchestrator.graph.grade_answer",
                new_callable=AsyncMock,
                return_value=ReflectionGrade.USEFUL,
            ) as mock_answer,
        ):
            result = await reflect(state)

            mock_halluc.assert_called_once()
            mock_answer.assert_called_once()

        assert result["loop_count"] == 1

    @pytest.mark.asyncio
    async def test_no_tool_message_uses_normal_path(self):
        """When no ToolMessage exists, reflect should call grade_hallucination as normal."""
        state = {
            "messages": [HumanMessage(content="hello")],
            "question": "hello",
            "generation": "hi",
            "documents": [],
            "loop_count": 0,
        }

        with (
            patch(
                "ragorchestrator.graph.grade_hallucination",
                new_callable=AsyncMock,
                return_value=ReflectionGrade.GROUNDED,
            ) as mock_halluc,
            patch(
                "ragorchestrator.graph.grade_answer",
                new_callable=AsyncMock,
                return_value=ReflectionGrade.USEFUL,
            ),
        ):
            await reflect(state)
            mock_halluc.assert_called_once()
