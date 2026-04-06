"""Tests for the query complexity classifier."""

import pytest

from ragorchestrator.classifier import Complexity, classify


class TestClassifier:
    @pytest.mark.parametrize(
        "query",
        [
            "What is Adam Claters job title?",
            "Who is the CEO?",
            "Give me the latest report",
            "List all employees",
            "Show me the budget",
            "Find the document",
            "Get the address",
            "When was the meeting?",
            "Where is the office?",
            "Is the project complete?",
            "10 employees in engineering",
            "5 years of history",
        ],
    )
    def test_simple_queries(self, query):
        result = classify(query)
        assert result == Complexity.SIMPLE, f"Expected SIMPLE for: {query}"

    @pytest.mark.parametrize(
        "query",
        [
            "Compare the approach in documents A and B",
            "What is the difference between Strategy A and Strategy B?",
            "Analyze the pros and cons",
            "Explain how the mechanism works",
            "What are the advantages and disadvantages?",
            "Summarize the key points",
            "What is the impact of this decision?",
            "First identify the problem, also consider the solution, moreover evaluate the risks",
            "What is the relationship between data and outcomes?",
        ],
    )
    def test_complex_queries(self, query):
        result = classify(query)
        assert result == Complexity.COMPLEX, f"Expected COMPLEX for: {query}"

    @pytest.mark.parametrize(
        "query",
        [
            "What is the latest news?",
            "What is the current weather?",
            "What is the stock price?",
            "Who is a famous celebrity?",
        ],
    )
    def test_external_queries(self, query):
        result = classify(query)
        assert result == Complexity.EXTERNAL, f"Expected EXTERNAL for: {query}"

    def test_long_query_is_complex(self):
        long_query = " ".join(["explain"] * 50)
        result = classify(long_query)
        assert result == Complexity.COMPLEX

    def test_multiple_questions(self):
        result = classify("What is X? How does Y work? Why is Z important?")
        assert result == Complexity.COMPLEX
