"""Query complexity classifier for adaptive routing.

Classifies incoming queries to determine execution path:
- SIMPLE:   direct ragpipe call (fast, cheap)
- COMPLEX:  full agentic loop with decomposition and multi-tool
- EXTERNAL: requires web search (future Tavily integration)
"""

import re
from enum import Enum


class Complexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    EXTERNAL = "external"


SIMPLE_INDICATORS = [
    r"^(who|what|when|where|is|are|do|does|can)\s",
    r"^give me (the )?(latest |current )?\w+",
    r"^list \w+",
    r"^show me \w+",
    r"^find \w+",
    r"^get \w+",
    r"^\w+ (name|address|phone|email|date|birth)",
    r"^\d+\s+\w+",  # "10 employees", "5 years"
]

COMPLEX_INDICATORS = [
    r"\b(compare|contrast|versus|vs\.?)\b",
    r"\b(difference|similarity|similarities)\b",
    r"\b(analyze|analysis|evaluate|assessment|critique)\b",
    r"\b(explain|describe).*(how|why|mechanism|process)\b",
    r"\b(advantages?|disadvantages?|pros?|cons?|benefits?)\b",
    r"\b(summary|tl;dr|brief)\b",
    r"\b(impact|effect|influence|consequence)\b",
    r"\b(first|second|third|finally|additionally)\b.*\b(also|moreover|further)\b",
    r"\b(multi-?step|multi)\b",
    r"\b(decomposition|decompose)\b",
    r"\?.*\?",  # multiple questions
    r"(relationship|between|connection|correlation|linked)\b.*\b(and|with)\b",
]

EXTERNAL_INDICATORS = [
    r"\b(news|current events)\b",
    r"\b(weather|forecast)\b",
    r"\b(stock price|market|trading)\b",
    r"\b(celebrity|celebrities|famous person)\b",
    r"\b(competitor|competitors|competitive analysis)\b",
    r"\b(benchmark|benchmarks?)\b",
]

LLM_THRESHOLD_QUERY_LENGTH = 200


def classify(query: str) -> Complexity:
    """Classify query complexity using keyword/pattern matching.

    Fast deterministic classification — no LLM call needed for clear cases.
    Returns Complexity enum value.
    """
    query_lower = query.lower().strip()

    if any(re.search(p, query_lower, re.IGNORECASE) for p in EXTERNAL_INDICATORS):
        return Complexity.EXTERNAL

    if any(re.search(p, query_lower) for p in COMPLEX_INDICATORS):
        return Complexity.COMPLEX

    if len(query) > LLM_THRESHOLD_QUERY_LENGTH:
        return Complexity.COMPLEX

    if any(re.search(p, query_lower) for p in SIMPLE_INDICATORS):
        return Complexity.SIMPLE

    words = query_lower.split()
    if len(words) <= 5 and "?" in query:
        return Complexity.SIMPLE

    if not any(c.isalpha() for c in query):
        return Complexity.SIMPLE

    return Complexity.COMPLEX


def route_for_complexity(query: str) -> str:
    """Return routing decision: 'simple' or 'agentic'.

    This is the public API used by app.py to decide execution path.
    """
    complexity = classify(query)
    return complexity.value
