"""
Brave Search API client.
Docs: https://api.search.brave.com/app/documentation/web-search/get-started
"""
from __future__ import annotations

import httpx

from config import BRAVE_API_KEY, BRAVE_MAX_RESULTS
from http_client import get_client
from logging_config import get_logger

log = get_logger(__name__)

_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def is_configured() -> bool:
    return bool(BRAVE_API_KEY)


def _parse_items(payload: dict) -> list[dict]:
    out: list[dict] = []
    for item in payload.get("web", {}).get("results", []):
        out.append({
            "title": item.get("title", "").strip(),
            "url": item.get("url", ""),
            "description": item.get("description", "").strip(),
            "extra_snippets": item.get("extra_snippets", []),
            "age": item.get("age", ""),
        })
    return out


def search(query: str, count: int | None = None) -> list[dict]:
    """
    Search the web via Brave Search API.
    Returns list of dicts: {title, url, description, extra_snippets, age}
    """
    if not is_configured():
        raise RuntimeError("BRAVE_API_KEY is not set")

    n = count or BRAVE_MAX_RESULTS
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": min(n, 20),  # Brave API max per request is 20
        "search_lang": "en",
        "result_filter": "web",
        "extra_snippets": "true",
    }

    client = get_client()
    r = client.get(_SEARCH_URL, headers=headers, params=params)
    r.raise_for_status()
    results = _parse_items(r.json())

    # If we need more than 20, paginate
    if n > 20:
        params["count"] = min(n - 20, 20)
        params["offset"] = 20
        try:
            r2 = client.get(_SEARCH_URL, headers=headers, params=params)
            r2.raise_for_status()
            results.extend(_parse_items(r2.json()))
        except httpx.HTTPError:
            log.exception("brave search pagination failed for %r", query)

    return results
