"""
Tavily Search API client.
Returns AI-optimized results with full page content snippets.
"""
from __future__ import annotations

import os

from logging_config import get_logger

log = get_logger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def is_configured() -> bool:
    return bool(TAVILY_API_KEY)


def search(query: str, count: int = 10) -> list[dict]:
    """Search via Tavily. Returns normalized {title, url, description, extra_snippets}."""
    if not is_configured():
        raise RuntimeError("TAVILY_API_KEY is not set")
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(query, max_results=min(count, 20), include_answer=False)
        results = []
        for item in resp.get("results", []):
            results.append({
                "title": (item.get("title") or "").strip(),
                "url": item.get("url", ""),
                "description": (item.get("content") or "").strip(),
                "extra_snippets": [],
                "age": item.get("published_date", ""),
                "source": "tavily",
            })
        return results
    except Exception:
        log.exception("tavily search failed for %r", query)
        return []
