"""Unit tests for search/reddit.py — uses pytest-httpx to mock the public JSON API."""
from __future__ import annotations

import pytest


def test_parse_public_post_skips_empty():
    from reddit_research.search import reddit
    assert reddit._parse_public_post({"data": {}}, "x", fetch_comments=False) is None
    assert reddit._parse_public_post({"data": {"id": "a"}}, "x", fetch_comments=False) is None


def test_parse_public_post_strips_deleted():
    from reddit_research.search import reddit
    child = {"data": {
        "id": "abc", "title": "t", "url": "u",
        "selftext": "[deleted]", "score": 1,
    }}
    p = reddit._parse_public_post(child, "x", fetch_comments=False)
    assert p["content"] == ""
    assert p["reddit_id"] == "abc"


def test_use_praw_toggle(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    import sys
    for m in list(sys.modules):
        if "reddit_research" in m:
            sys.modules.pop(m)
    from reddit_research.search import reddit as r
    assert r.use_praw() is False


@pytest.fixture
def httpx_mock_fixture():
    pytest.importorskip("pytest_httpx")


def test_public_search_handles_http_error(httpx_mock_fixture, httpx_mock):
    from reddit_research.search import reddit
    httpx_mock.add_response(status_code=500)
    assert reddit._public_search("x", "q", 10) == []
