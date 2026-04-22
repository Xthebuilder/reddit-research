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
from reddit_research.utils.resources import safe_worker_count
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

def _post_embed_text(post: dict) -> str:
    text = post["title"]
    if post.get("content"):
        text += "\n" + post["content"][:500]
    if post.get("comments"):
        text += "\n" + " ".join(c[:150] for c in post["comments"][:2])
    return text


def _web_embed_text(result: dict) -> str:
    text = result["title"]
    if result.get("description"):
        text += "\n" + result["description"][:500]
    if result.get("extra_snippets"):
        text += "\n" + " ".join(s[:150] for s in result["extra_snippets"][:2])
    return text


def _judge_and_summarize_post(
    post_id: int,
    post: dict,
    state: dict,
    query: str,
    original_topic: str | None,
    max_upvotes: int,
) -> tuple[str, float, bool]:
    """Judge + summarize one post. Embedding handled separately by batch path."""
    if state["relevance_score"] == -1:
        llm_score = llm.judge_relevance(post, query, original_topic=original_topic)
        hybrid = llm.blend_scores(llm_score, post.get("score", 0), max_upvotes)
        db.update_relevance(post_id, hybrid)
        fresh = True
    else:
        hybrid = state["relevance_score"]
        fresh = False

    if hybrid >= RELEVANCE_THRESHOLD and not state["summary"]:
        try:
            summary = llm.summarize_post(post)
            if summary:
                db.update_post_summary(post_id, summary)
        except Exception:
            log.exception("summarize_post failed for %s", post.get("reddit_id"))

    return post["title"][:40], hybrid, fresh


def _judge_and_summarize_web(
    rid: int,
    result: dict,
    state: dict,
    query: str,
    original_topic: str | None,
) -> tuple[str, float, bool]:
    """Judge + summarize one web result. Embedding handled separately by batch path."""
    if state["relevance_score"] == -1:
        score = llm.judge_web_relevance(result, query, original_topic=original_topic)
        db.update_web_relevance(rid, score)
        fresh = True
    else:
        score = state["relevance_score"]
        fresh = False

    if score >= RELEVANCE_THRESHOLD and not state["summary"]:
        try:
            summary = llm.summarize_web_result(result)
            if summary:
                db.update_web_summary(rid, summary)
        except Exception:
            log.exception("summarize_web_result failed for %s", result.get("url"))

    return result["title"][:40], score, fresh


# Back-compat shims — headless._process_web_batch still calls _process_single_web
def _process_single_post(topic_id, post, query, original_topic, max_upvotes):
    post_id = db.save_post(topic_id, post)
    state = db.get_post_processing_state(post_id)
    if not state["embedding"]:
        try:
            db.update_post_embedding(post_id, llm.embed_post(post))
        except Exception:
            log.exception("embed_post failed for %s", post.get("reddit_id"))
        state = db.get_post_processing_state(post_id)
    return _judge_and_summarize_post(post_id, post, state, query, original_topic, max_upvotes)


def _process_single_web(topic_id, result, query, original_topic):
    rid = db.save_web_result(topic_id, result)
    state = db.get_web_processing_state(rid)
    if not state["embedding"]:
        try:
            db.update_web_embedding(rid, llm.embed_web_result(result))
        except Exception:
            log.exception("embed_web_result failed for %s", result.get("url"))
        state = db.get_web_processing_state(rid)
    return _judge_and_summarize_web(rid, result, state, query, original_topic)


def fetch_and_process_posts(
    topic_id: int,
    query: str,
    subreddits: list[str],
    progress: ProgressFn = _NOOP,
    tag: str = "",
    original_topic: str | None = None,
    seen_urls: set | None = None,
) -> int:
    """Fetch Reddit posts, then judge + embed + summarize in parallel. Returns post count."""
    prefix = f"[{tag}] " if tag else ""

    def reddit_progress(sub, done, total):
        progress(f"{prefix}Fetching r/{sub} ({done}/{total})...")

    posts = reddit.fetch_topic(query, subreddits, progress_cb=reddit_progress, seen_urls=seen_urls)
    if not posts:
        return 0

    # Step 1: persist all posts and snapshot processing state
    saved: list[tuple[int, dict, dict]] = []
    for post in posts:
        post_id = db.save_post(topic_id, post)
        saved.append((post_id, post, db.get_post_processing_state(post_id)))

    # Step 2: batch-embed everything that needs an embedding (one Ollama call)
    need_embed = [(pid, p) for pid, p, s in saved if not s["embedding"]]
    if need_embed:
        progress(f"{prefix}Batch-embedding {len(need_embed)} posts in 1 API call...")
        try:
            vectors = llm.embed_batch([_post_embed_text(p) for _, p in need_embed])
            for (pid, _), vec in zip(need_embed, vectors):
                if vec:
                    db.update_post_embedding(pid, vec)
            # refresh state so the judge step sees embeddings as present
            saved = [(pid, p, db.get_post_processing_state(pid)) for pid, p, _ in saved]
        except Exception:
            log.exception("batch embed failed; falling back to per-item embeds in worker pool")

    # Step 3: parallel judge + summarize
    max_upvotes = max((p.get("score", 0) for p in posts), default=1) or 1
    workers, worker_reason = safe_worker_count()
    progress(f"{prefix}Judging+summarizing {len(saved)} Reddit posts — {worker_reason}")

    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_judge_and_summarize_post, pid, p, s, query, original_topic, max_upvotes): p
            for pid, p, s in saved
        }
        for fut in as_completed(futures):
            done_count += 1
            try:
                title, score, fresh = fut.result()
                label = f"→ {score:.1f}/10" if fresh else f"→ {score:.1f}/10 (cached)"
                progress(f"{prefix}[Reddit] {done_count}/{len(posts)}: {title}... {label}")
            except Exception:
                log.exception("post processing failed")
                progress(f"{prefix}[Reddit] {done_count}/{len(posts)}: ERROR")

    return len(posts)


def fetch_and_process_web(
    topic_id: int,
    query: str,
    sites: list[str],
    progress: ProgressFn = _NOOP,
    tag: str = "",
    original_topic: str | None = None,
) -> int:
    """Fetch web results from all APIs, then judge + embed + summarize in parallel."""
    prefix = f"[{tag}] " if tag else ""

    web_results = fetch_all_web(query, sites, progress)
    if not web_results:
        progress(f"{prefix}No web APIs configured — skipping web search")
        return 0

    web_count = len(web_results)

    # Step 1: persist + snapshot state
    saved: list[tuple[int, dict, dict]] = []
    for result in web_results:
        rid = db.save_web_result(topic_id, result)
        saved.append((rid, result, db.get_web_processing_state(rid)))

    # Step 2: batch-embed everything that needs it
    need_embed = [(rid, r) for rid, r, s in saved if not s["embedding"]]
    if need_embed:
        progress(f"{prefix}Batch-embedding {len(need_embed)} web results in 1 API call...")
        try:
            vectors = llm.embed_batch([_web_embed_text(r) for _, r in need_embed])
            for (rid, _), vec in zip(need_embed, vectors):
                if vec:
                    db.update_web_embedding(rid, vec)
            saved = [(rid, r, db.get_web_processing_state(rid)) for rid, r, _ in saved]
        except Exception:
            log.exception("batch embed failed; falling back to per-item embeds in worker pool")

    # Step 3: parallel judge + summarize
    workers, worker_reason = safe_worker_count()
    progress(f"{prefix}Judging+summarizing {web_count} web results — {worker_reason}")

    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_judge_and_summarize_web, rid, r, s, query, original_topic): r
            for rid, r, s in saved
        }
        for fut in as_completed(futures):
            done_count += 1
            try:
                title, score, fresh = fut.result()
                label = f"→ {score:.0f}/10" if fresh else f"→ {score:.0f}/10 (cached)"
                progress(f"{prefix}[Web] {done_count}/{web_count}: {title}... {label}")
            except Exception:
                log.exception("web processing failed")
                progress(f"{prefix}[Web] {done_count}/{web_count}: ERROR")

    return web_count
