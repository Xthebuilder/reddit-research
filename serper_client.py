"""
Serper.dev Google Search API client.
Docs: https://serper.dev
"""
from __future__ import annotations

import os

import httpx

from http_client import get_client
from logging_config import get_logger

log = get_logger(__name__)

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
_SEARCH_URL = "https://google.serper.dev/search"


def is_configured() -> bool:
    return bool(SERPER_API_KEY)


def search(query: str, count: int = 10) -> list[dict]:
    """Search via Serper (Google). Returns normalized {title, url, description, extra_snippets}."""
    if not is_configured():
        raise RuntimeError("SERPER_API_KEY is not set")
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": min(count, 20)}
    try:
        r = get_client().post(_SEARCH_URL, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic", []):
            snippets = []
            if item.get("sitelinks"):
                snippets = [s.get("snippet", "") for s in item["sitelinks"][:3] if s.get("snippet")]
            results.append({
                "title": (item.get("title") or "").strip(),
                "url": item.get("link", ""),
                "description": (item.get("snippet") or "").strip(),
                "extra_snippets": snippets,
                "age": item.get("date", ""),
                "source": "serper",
            })
        return results
    except httpx.HTTPError:
        log.exception("serper search failed for %r", query)
        return []
    except Exception:
        log.exception("serper search unexpected error for %r", query)
        return []
