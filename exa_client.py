"""
Exa.ai semantic/neural search client.
Docs: https://docs.exa.ai
"""
from __future__ import annotations

import os

from logging_config import get_logger

log = get_logger(__name__)

EXA_API_KEY = os.getenv("EXA_API_KEY", "")


def is_configured() -> bool:
    return bool(EXA_API_KEY)


def search(query: str, count: int = 10) -> list[dict]:
    """Search via Exa (neural). Returns normalized {title, url, description, extra_snippets}."""
    if not is_configured():
        raise RuntimeError("EXA_API_KEY is not set")
    try:
        from exa_py import Exa  # type: ignore
        client = Exa(api_key=EXA_API_KEY)
        resp = client.search_and_contents(
            query,
            num_results=min(count, 20),
            text={"max_characters": 600},
            highlights={"num_sentences": 3},
        )
        results = []
        for item in resp.results:
            highlights = getattr(item, "highlights", None) or []
            description = ""
            if hasattr(item, "text") and item.text:
                description = item.text[:600]
            elif highlights:
                description = " ".join(highlights[:2])
            results.append({
                "title": (getattr(item, "title", "") or "").strip(),
                "url": getattr(item, "url", ""),
                "description": description.strip(),
                "extra_snippets": highlights[:3] if highlights else [],
                "age": getattr(item, "published_date", "") or "",
                "source": "exa",
            })
        return results
    except Exception:
        log.exception("exa search failed for %r", query)
        return []
