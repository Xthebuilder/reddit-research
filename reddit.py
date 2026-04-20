"""
Reddit fetcher — uses PRAW when credentials are set, falls back to
Reddit's public JSON API (no auth required, ~60 req/min).
"""
from __future__ import annotations

import time

import httpx

from config import (
    MAX_COMMENTS_PER_POST,
    MAX_POSTS_PER_SUB,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)
from http_client import get_client
from logging_config import get_logger

log = get_logger(__name__)

_HEADERS = {"User-Agent": REDDIT_USER_AGENT}
_BASE = "https://www.reddit.com"


# ---------------------------------------------------------------------------
# Public JSON fallback (no auth)
# ---------------------------------------------------------------------------

def _public_search(subreddit: str, query: str, limit: int, time_filter: str = "year") -> list[dict]:
    url = f"{_BASE}/r/{subreddit}/search.json"
    params = {"q": query, "restrict_sr": 1, "sort": "relevance", "limit": limit, "t": time_filter}
    try:
        r = get_client().get(url, params=params, headers=_HEADERS)
        r.raise_for_status()
        data = r.json()
        return data.get("data", {}).get("children", [])
    except httpx.HTTPStatusError as e:
        log.warning("reddit search HTTP %s for r/%s q=%r", e.response.status_code, subreddit, query)
    except httpx.HTTPError:
        log.exception("reddit search transport error r/%s q=%r", subreddit, query)
    except Exception:
        log.exception("reddit search unexpected error r/%s q=%r", subreddit, query)
    return []


def _extract_comment_bodies(children: list, depth: int = 0, max_depth: int = 1) -> list[str]:
    """Recursively extract comment bodies up to max_depth levels."""
    bodies: list[str] = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        body = data.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            bodies.append(body.strip())
        if depth < max_depth:
            replies = data.get("replies")
            if isinstance(replies, dict):
                sub_children = replies.get("data", {}).get("children", [])
                bodies.extend(_extract_comment_bodies(sub_children, depth + 1, max_depth))
    return bodies


def _public_comments(subreddit: str, post_id: str, deep: bool = False) -> list[str]:
    url = f"{_BASE}/r/{subreddit}/comments/{post_id}.json"
    try:
        r = get_client().get(url, headers=_HEADERS)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2:
            return []
        children = data[1].get("data", {}).get("children", [])
        max_depth = 1 if deep else 0
        comments = _extract_comment_bodies(children[:MAX_COMMENTS_PER_POST * 2], max_depth=max_depth)
        return comments[:MAX_COMMENTS_PER_POST * (2 if deep else 1)]
    except httpx.HTTPStatusError as e:
        log.warning("reddit comments HTTP %s for %s/%s", e.response.status_code, subreddit, post_id)
    except httpx.HTTPError:
        log.exception("reddit comments transport error %s/%s", subreddit, post_id)
    except Exception:
        log.exception("reddit comments unexpected error %s/%s", subreddit, post_id)
    return []


def _parse_public_post(child: dict, subreddit: str, fetch_comments: bool = True) -> dict | None:
    d = child.get("data", {})
    post_id = d.get("id", "")
    title = d.get("title", "").strip()
    if not title or not post_id:
        return None
    content = d.get("selftext", "").strip()
    if content in ("[deleted]", "[removed]"):
        content = ""
    url = d.get("url", "")
    score = d.get("score", 0)
    num_comments = int(d.get("num_comments", 0) or 0)

    comments: list[str] = []
    if fetch_comments:
        # Fetch deeper replies for high-engagement posts
        deep = num_comments > 50
        comments = _public_comments(subreddit, post_id, deep=deep)
        time.sleep(0.5)  # be polite to Reddit's public API

    return {
        "reddit_id": post_id,
        "subreddit": subreddit,
        "title": title,
        "url": url,
        "score": score,
        "num_comments": num_comments,
        "content": content,
        "comments": comments,
    }


# ---------------------------------------------------------------------------
# PRAW path (when credentials are available)
# ---------------------------------------------------------------------------

def _praw_search(subreddit: str, query: str, limit: int, time_filter: str = "year") -> list[dict]:
    import praw  # type: ignore

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    results: list[dict] = []
    sub = reddit.subreddit(subreddit)
    for post in sub.search(query, sort="relevance", time_filter=time_filter, limit=limit):
        post.comments.replace_more(limit=0)
        num_comments = int(getattr(post, "num_comments", 0) or 0)
        comment_limit = MAX_COMMENTS_PER_POST * 2 if num_comments > 50 else MAX_COMMENTS_PER_POST
        comments = [
            c.body.strip()
            for c in post.comments[:comment_limit]
            if hasattr(c, "body") and c.body not in ("[deleted]", "[removed]")
        ]
        results.append({
            "reddit_id": post.id,
            "subreddit": subreddit,
            "title": post.title,
            "url": post.url,
            "score": post.score,
            "num_comments": num_comments,
            "content": (post.selftext or "").strip(),
            "comments": comments,
        })
    return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def subreddit_exists(subreddit: str) -> bool:
    """Quick check if a subreddit exists and is accessible."""
    url = f"{_BASE}/r/{subreddit}/about.json"
    try:
        r = get_client().get(url, headers=_HEADERS, timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", {})
            return data.get("subreddit_type") not in (None, "private", "restricted")
        return False
    except Exception:
        return False


_SAFE_FALLBACKS = ["travel", "solotravel", "backpacking", "AskReddit", "explainlikeimfive"]

def filter_valid_subreddits(subreddits: list[str]) -> list[str]:
    """Remove subreddits that don't exist or are inaccessible."""
    valid = []
    for sub in subreddits:
        if subreddit_exists(sub):
            valid.append(sub)
        else:
            log.warning("subreddit r/%s not found or inaccessible — skipping", sub)
    if not valid:
        log.warning("all suggested subreddits failed validation — using safe fallbacks")
        return [s for s in _SAFE_FALLBACKS if subreddit_exists(s)][:4]
    return valid


def use_praw() -> bool:
    return bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)


def search_subreddit(subreddit: str, query: str, time_filter: str = "year") -> list[dict]:
    """Return a list of post dicts for a given subreddit + query."""
    if use_praw():
        try:
            return _praw_search(subreddit, query, MAX_POSTS_PER_SUB, time_filter=time_filter)
        except Exception:
            log.exception("PRAW search failed, falling back to public API for r/%s", subreddit)

    children = _public_search(subreddit, query, MAX_POSTS_PER_SUB, time_filter=time_filter)
    posts = []
    for child in children:
        post = _parse_public_post(child, subreddit, fetch_comments=True)
        if post:
            posts.append(post)
    return posts


def fetch_topic(
    query: str,
    subreddits: list[str],
    progress_cb=None,
    time_filter: str = "year",
    seen_urls: set[str] | None = None,
) -> list[dict]:
    """
    Fetch posts across multiple subreddits, deduplicating by URL across calls.
    progress_cb(subreddit, done, total) is called after each subreddit.
    Pass seen_urls to share dedup state across multiple fetch_topic calls.
    """
    if seen_urls is None:
        seen_urls = set()
    all_posts: list[dict] = []
    for i, sub in enumerate(subreddits):
        posts = search_subreddit(sub, query, time_filter=time_filter)
        for post in posts:
            url = post.get("url", "")
            reddit_id = post.get("reddit_id", "")
            dedup_key = reddit_id or url
            if dedup_key and dedup_key in seen_urls:
                continue
            if dedup_key:
                seen_urls.add(dedup_key)
            all_posts.append(post)
        if progress_cb:
            progress_cb(sub, i + 1, len(subreddits))
        time.sleep(1)  # rate limit courtesy pause
    return all_posts
