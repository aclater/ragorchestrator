"""Web search tool — Tavily search for real-time information augmentation.

Provides web search capability when ragpipe corpus does not contain the
answer. Implements the 'web augmentation' leg of CRAG.

Disabled when:
- TAVILY_API_KEY is not set
- DISABLE_WEB_SEARCH=true

When disabled, the tool is not registered with the supervisor graph,
so the LLM never sees it as an option.
"""

import logging
import os

log = logging.getLogger(__name__)


def _web_search_enabled() -> bool:
    """Check if web search should be enabled."""
    if os.environ.get("DISABLE_WEB_SEARCH", "").lower() in ("true", "1", "yes"):
        log.info("Web search disabled via DISABLE_WEB_SEARCH")
        return False
    if not os.environ.get("TAVILY_API_KEY"):
        log.info("Web search disabled: TAVILY_API_KEY not set")
        return False
    return True


def get_web_search_tool():
    """Return the Tavily web search tool if enabled, else None.

    Returns None when TAVILY_API_KEY is not set or DISABLE_WEB_SEARCH=true,
    so the supervisor graph simply omits the tool.
    """
    if not _web_search_enabled():
        return None

    from langchain_tavily import TavilySearch

    tool = TavilySearch(
        max_results=3,
        topic="general",
    )
    # Override the tool description so the supervisor knows when to use it
    tool.description = (
        "Search the web for current or real-time information. "
        "Use when the document corpus does not contain the answer, "
        "the query asks about recent events or current data, "
        "or ragpipe returned grounding='general' indicating no corpus match."
    )
    return tool
