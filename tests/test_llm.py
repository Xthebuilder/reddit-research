"""Unit tests for llm.py parsing/format helpers (no network)."""
from __future__ import annotations


def test_build_context_orders_by_relevance():
    from reddit_research import llm
    posts = [
        {"title": "low", "subreddit": "x", "url": "u1", "score": 1, "relevance_score": 1, "comments": []},
        {"title": "high", "subreddit": "x", "url": "u2", "score": 2, "relevance_score": 9, "comments": []},
    ]
    ctx = llm.build_context(posts, [], "topic")
    assert ctx.index("high") < ctx.index("low")


def test_build_context_uses_summary_over_content():
    from reddit_research import llm
    posts = [{
        "title": "t", "subreddit": "x", "url": "u", "score": 1,
        "relevance_score": 5, "summary": "SUMMARY_HERE",
        "content": "RAW_CONTENT_HERE", "comments": [],
    }]
    ctx = llm.build_context(posts, [], "topic")
    assert "SUMMARY_HERE" in ctx
    assert "RAW_CONTENT_HERE" not in ctx
