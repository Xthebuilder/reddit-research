"""
Research pipeline — the core fetch/judge/embed/summarize orchestration.

Both the TUI (ui/app.py) and the headless CLI (headless.py) use these
functions. A progress callback pattern keeps the logic UI-agnostic:

    def progress(msg: str) -> None:
        ...  # update status bar, print to stdout, etc.

    fetch_and_process_posts(topic_id, query, subreddits, progress=progress)
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from reddit_research import db, llm
from reddit_research.config import BRAVE_MAX_RESULTS, RELEVANCE_THRESHOLD
from reddit_research.search import brave, exa, reddit, serper, tavily
from reddit_research.ui.keywords import AUTO_WEBSITE_KEYWORDS, DEFAULT_SUBREDDITS, DEFAULT_WEB_SITES
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)

ProgressFn = Callable[[str], None]

_NOOP: ProgressFn = lambda _: None  # noqa: E731


# ---------------------------------------------------------------------------
# Subreddit / site auto-selection
# ---------------------------------------------------------------------------

def auto_subreddits(
    query: str,
    progress: ProgressFn = _NOOP,
    llm_fallback: bool = False,
) -> list[str]:
    """Keyword-based subreddit suggestion. llm_fallback=True requires a background thread."""
    from reddit_research.ui.keywords import AUTO_SUBREDDIT_KEYWORDS
    normalized = query.lower().strip()
    scored: list[tuple[int, str]] = []

    for subreddit, keywords in AUTO_SUBREDDIT_KEYWORDS.items():
        score = sum(2 for kw in keywords if kw in normalized)
        if subreddit.lower() in normalized:
            score += 5
        if score:
            scored.append((score, subreddit))

    if not scored:
        if llm_fallback:
            try:
                suggested = llm.suggest_subreddits(query, num=6)
                if suggested:
                    return suggested
            except Exception:
                pass
        return DEFAULT_SUBREDDITS[:4]

    scored.sort(key=lambda x: -x[0])
    return [sub for _, sub in scored[:6]]


def auto_sites(query: str) -> list[str]:
    """Keyword-based website suggestion."""
    normalized = query.lower().strip()
    scored: list[tuple[int, str]] = []

    for site, keywords in AUTO_WEBSITE_KEYWORDS.items():
        score = sum(2 for kw in keywords if kw in normalized)
        if site.replace(".", "") in normalized.replace(".", ""):
            score += 4
        if score:
            scored.append((score, site))

    if not scored:
        return DEFAULT_WEB_SITES[:]

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [site for _, site in scored[:4]]


# ---------------------------------------------------------------------------
# Web search helpers
# ---------------------------------------------------------------------------

def search_web_for_sites(query: str, sites: list[str]) -> list[dict]:
    """Search Brave for each site separately and deduplicate."""
    if not brave.is_configured():
        return []

    per_site = max(3, min(8, BRAVE_MAX_RESULTS // max(1, len(sites))))
    seen_urls: set[str] = set()
    combined: list[dict] = []

    for site in sites:
        scoped_query = f"{query} site:{site}"
        try:
            results = brave.search(scoped_query, count=per_site)
        except Exception:
            continue
        for result in results:
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            combined.append(result)
            if len(combined) >= BRAVE_MAX_RESULTS:
                return combined

    return combined


def fetch_all_web(
    query: str,
    sites: list[str],
    progress: ProgressFn = _NOOP,
) -> list[dict]:
    """Fire all configured search APIs in parallel and deduplicate by URL."""
    tasks: list[tuple[str, Callable]] = []
    if brave.is_configured():
        tasks.append(("brave", lambda q=query: search_web_for_sites(q, sites)))
    if tavily.is_configured():
        tasks.append(("tavily", lambda q=query: tavily.search(q, count=15)))
    if serper.is_configured():
        tasks.append(("serper", lambda q=query: serper.search(q, count=15)))
    if exa.is_configured():
        tasks.append(("exa", lambda q=query: exa.search(q, count=10)))

    if not tasks:
        return []

    api_names = ", ".join(n for n, _ in tasks)
    progress(f"Searching [{api_names}] in parallel...")

    seen: set[str] = set()
    combined: list[dict] = []

    def _run(name: str, fn: Callable) -> list[dict]:
        try:
            results = fn()
            for r in results:
                r.setdefault("source", name)
            return results
        except Exception:
            log.exception("%s search failed for %r", name, query)
            return []

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(_run, name, fn): name for name, fn in tasks}
        for fut in as_completed(futures):
            for r in fut.result():
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    combined.append(r)

    return combined


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

def fetch_and_process_posts(
    topic_id: int,
    query: str,
    subreddits: list[str],
    progress: ProgressFn = _NOOP,
    tag: str = "",
    original_topic: str | None = None,
    seen_urls: set | None = None,
) -> int:
    """Fetch Reddit posts, then judge + embed + summarize. Returns post count."""
    prefix = f"[{tag}] " if tag else ""

    def reddit_progress(sub, done, total):
        progress(f"{prefix}Fetching r/{sub} ({done}/{total})...")

    posts = reddit.fetch_topic(query, subreddits, progress_cb=reddit_progress, seen_urls=seen_urls)
    progress(f"{prefix}Fetched {len(posts)} Reddit posts (deduped) — processing...")

    max_upvotes = max((p.get("score", 0) for p in posts), default=1)

    for i, post in enumerate(posts):
        post_id = db.save_post(topic_id, post)
        llm_score = llm.judge_relevance(post, query, original_topic=original_topic)
        hybrid = llm.blend_scores(llm_score, post.get("score", 0), max_upvotes)
        db.update_relevance(post_id, hybrid)
        progress(f"{prefix}[Reddit] {i+1}/{len(posts)}: {post['title'][:35]}... → {hybrid:.1f}/10")
        try:
            embedding = llm.embed_post(post)
            db.update_post_embedding(post_id, embedding)
        except Exception:
            pass
        if hybrid >= RELEVANCE_THRESHOLD:
            try:
                summary = llm.summarize_post(post)
                if summary:
                    db.update_post_summary(post_id, summary)
            except Exception:
                pass

    return len(posts)


def fetch_and_process_web(
    topic_id: int,
    query: str,
    sites: list[str],
    progress: ProgressFn = _NOOP,
    tag: str = "",
    original_topic: str | None = None,
) -> int:
    """Fetch web results from all APIs, then judge + embed + summarize. Returns result count."""
    prefix = f"[{tag}] " if tag else ""

    web_results = fetch_all_web(query, sites, progress)
    if not web_results:
        progress(f"{prefix}No web APIs configured — skipping web search")
        return 0

    web_count = len(web_results)
    progress(f"{prefix}Fetched {web_count} unique web results — processing...")

    for i, result in enumerate(web_results):
        rid = db.save_web_result(topic_id, result)
        score = llm.judge_web_relevance(result, query, original_topic=original_topic)
        db.update_web_relevance(rid, score)
        progress(f"{prefix}[Web] {i+1}/{web_count}: {result['title'][:35]}... → {score:.0f}/10")
        try:
            embedding = llm.embed_web_result(result)
            db.update_web_embedding(rid, embedding)
        except Exception:
            pass
        if score >= RELEVANCE_THRESHOLD:
            try:
                summary = llm.summarize_web_result(result)
                if summary:
                    db.update_web_summary(rid, summary)
            except Exception:
                pass

    return web_count
