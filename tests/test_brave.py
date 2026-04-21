"""Unit tests for search/brave.py."""
from __future__ import annotations

import pytest


def test_is_configured_false(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    import sys
    for m in list(sys.modules):
        if "reddit_research" in m:
            sys.modules.pop(m)
    from reddit_research.search import brave
    assert brave.is_configured() is False


def test_search_raises_without_key(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    import sys
    for m in list(sys.modules):
        if "reddit_research" in m:
            sys.modules.pop(m)
    from reddit_research.search import brave
    with pytest.raises(RuntimeError):
        brave.search("anything")


def test_parse_items_shape():
    from reddit_research.search import brave
    sample = {"web": {"results": [
        {"title": " T ", "url": "u", "description": " d ", "extra_snippets": ["s"], "age": "1d"}
    ]}}
    items = brave._parse_items(sample)
    assert items == [{"title": "T", "url": "u", "description": "d", "extra_snippets": ["s"], "age": "1d"}]
