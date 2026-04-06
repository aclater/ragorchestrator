"""Tests for the Self-RAG reflection module."""

from ragorchestrator.reflection import ReflectionGrade


class TestReflectionGrade:
    def test_reflection_grade_values(self):
        assert ReflectionGrade.GROUNDED.value == "grounded"
        assert ReflectionGrade.UNGROUNDED.value == "ungrounded"
        assert ReflectionGrade.USEFUL.value == "useful"
        assert ReflectionGrade.NOT_USEFUL.value == "not_useful"
