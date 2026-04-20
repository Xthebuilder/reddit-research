#!/usr/bin/env python3
"""
Headless ResearchOS pipeline — runs the full fetch/judge/embed/summarize/
gap-analysis/report cycle from the command line without the TUI.

Usage:
    python headless.py --topic "ZFS snapshots" --subreddits "zfs,linux,sysadmin"

Prints progress lines to stdout prefixed with [STATUS].
Prints the final report path as:  REPORT_PATH:/absolute/path/to/report.md
Exit 0 on success, non-zero on failure.

Designed to be called by JRVS's autonomous_research.py via
asyncio.create_subprocess_exec.
"""
import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from env_loader import load_env
from logging_config import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger(__name__)

import brave  # noqa: E402
import db  # noqa: E402
import exa_client  # noqa: E402
import llm  # noqa: E402
import reddit  # noqa: E402
import report  # noqa: E402
import serper_client  # noqa: E402
import tavily_client  # noqa: E402
from _version import __version__  # noqa: E402
from config import MAX_RESEARCH_ITERATIONS, RELEVANCE_THRESHOLD, validate  # noqa: E402

# ── Subreddit auto-selection (lifted from tui.py) ───────────────────────────
AUTO_SUBREDDIT_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Tech
    "sysadmin": (
        "sysadmin", "admin", "server", "servers", "infrastructure", "network", "dns",
        "backup", "monitoring", "vpn", "proxy", "firewall", "storage", "raid",
    ),
    "linux": ("linux", "kernel", "distro", "desktop", "wayland", "x11", "package", "packages"),
    "selfhosted": (
        "self-hosted", "selfhosted", "docker", "compose", "container", "containers",
        "kubernetes", "k8s", "service", "homelab", "nas", "media server", "plex", "jellyfin",
    ),
    "homelab": ("homelab", "home lab", "proxmox", "vm", "virtual machine", "backup", "nas", "storage"),
    "LocalLLaMA": ("llm", "local llama", "ollama", "lm studio", "langchain", "rag", "ai", "model", "models"),
    "linuxquestions": ("help", "how do i", "how to", "error", "fix", "install", "configure", "setup"),
    "archlinux": ("arch", "pacman", "aur", "arch linux", "archlinux"),
    "debian": ("debian", "apt", "apt-get", "ubuntu", "mint"),
    "commandline": ("command line", "cli", "shell", "terminal", "bash", "zsh", "ssh"),
    "programming": ("programming", "coding", "software", "code", "developer", "algorithm", "api"),
    "datascience": ("data science", "data analysis", "machine learning", "pandas", "data analyst", "analytics"),
    # Travel
    "travel": (
        "travel", "trip", "vacation", "holiday", "visit", "country", "countries",
        "tourist", "tourism", "abroad", "international", "passport", "visa",
    ),
    "solotravel": (
        "solo travel", "solo trip", "traveling alone", "first time travel", "new traveler",
        "backpack", "backpacking", "budget travel", "hostel",
    ),
    "digitalnomad": ("digital nomad", "remote work abroad", "work while traveling", "nomad"),
    # Finance / Career
    "personalfinance": (
        "money", "finance", "budget", "saving", "investing", "debt", "salary",
        "retirement", "401k", "credit", "loan", "mortgage",
    ),
    "financialindependence": ("fire", "financial independence", "retire early", "passive income", "frugal"),
    "careerguidance": ("career", "job", "resume", "interview", "salary negotiation", "career advice", "job search"),
    "cscareerquestions": ("software engineer", "developer", "coding job", "tech job", "swe", "programming career"),
    # Health / Fitness
    "fitness": ("fitness", "workout", "gym", "exercise", "weight loss", "muscle", "training"),
    "nutrition": ("nutrition", "diet", "eating", "food", "calories", "meal", "supplements"),
    "mentalhealth": ("mental health", "anxiety", "depression", "stress", "therapy", "wellbeing"),
    # General
    "AskReddit": ("what do people think", "best way to", "opinions on", "recommend", "advice", "thoughts on"),
}

DEFAULT_SUBREDDITS = list(AUTO_SUBREDDIT_KEYWORDS.keys())

AUTO_WEBSITE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "reddit.com": ("reddit", "subreddit", "thread"),
    "github.com": ("github", "repo", "repository", "source", "code"),
    "wiki.archlinux.org": ("linux", "arch", "systemd", "networking", "zfs"),
    "serverfault.com": ("sysadmin", "network", "dns", "storage", "raid"),
    "stackoverflow.com": ("error", "exception", "bug", "fix", "how to"),
}
DEFAULT_WEB_SITES = ["reddit.com", "github.com", "serverfault.com", "stackoverflow.com"]


def _auto_subreddits(query: str) -> list[str]:
    normalized = query.lower().strip()
    scored: list[tuple[int, str]] = []
    for sub in DEFAULT_SUBREDDITS:
        score = 0
        for kw in AUTO_SUBREDDIT_KEYWORDS.get(sub, ()):
            if kw in normalized:
                score += 2
        if sub.lower() in normalized:
            score += 5
        if score:
            scored.append((score, sub))
    if not scored:
        status("No keyword match — asking LLM for subreddit suggestions...")
        suggested = llm.suggest_subreddits(query, num=6)
        if suggested:
            status(f"LLM suggested: {', '.join(suggested)}")
            return suggested
        return list(DEFAULT_SUBREDDITS)[:4]
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:6]]


def _auto_sites(query: str) -> list[str]:
    normalized = query.lower().strip()
    scored = []
    for site, keywords in AUTO_WEBSITE_KEYWORDS.items():
        score = sum(2 for kw in keywords if kw in normalized)
        if score:
            scored.append((score, site))
    if not scored:
        return DEFAULT_WEB_SITES[:]
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:4]]


def _search_web_for_sites(query: str, sites: list[str]) -> list[dict]:
    if not brave.is_configured():
        return []
    from config import BRAVE_MAX_RESULTS
    per_site = max(3, min(8, BRAVE_MAX_RESULTS // max(1, len(sites))))
    seen, combined = set(), []
    for site in sites:
        try:
            results = brave.search(f"{query} site:{site}", count=per_site)
        except Exception:
            continue
        for r in results:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                combined.append(r)
                if len(combined) >= BRAVE_MAX_RESULTS:
                    return combined
    return combined


def _active_web_apis() -> list[tuple[str, callable]]:
    """Return list of (name, search_fn) for every configured search API."""
    apis = []
    if brave.is_configured():
        apis.append(("brave", lambda q: _search_web_for_sites(q, _auto_sites(q))))
    if tavily_client.is_configured():
        apis.append(("tavily", lambda q: tavily_client.search(q, count=15)))
    if serper_client.is_configured():
        apis.append(("serper", lambda q: serper_client.search(q, count=15)))
    if exa_client.is_configured():
        apis.append(("exa", lambda q: exa_client.search(q, count=10)))
    return apis


def _search_web_parallel(queries: list[str], sites: list[str]) -> list[dict]:
    """
    Fire all queries across all configured search APIs in parallel.
    Results are deduplicated by URL and returned as a flat list.
    """
    apis = _active_web_apis()
    if not apis:
        status("No web search APIs configured — skipping web search")
        return []

    seen: set[str] = set()
    combined: list[dict] = []
    tasks: list[tuple[str, str, callable]] = []  # (api_name, query, fn)

    for q in queries:
        for api_name, fn in apis:
            tasks.append((api_name, q, fn))

    status(f"Firing {len(tasks)} web searches in parallel ({len(apis)} APIs × {len(queries)} queries)...")

    def _fetch(api_name: str, q: str, fn) -> list[dict]:
        try:
            results = fn(q)
            for r in results:
                r.setdefault("source", api_name)
            return results
        except Exception:
            log.exception("%s search failed for %r", api_name, q)
            return []

    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        futures = {pool.submit(_fetch, name, q, fn): (name, q) for name, q, fn in tasks}
        for fut in as_completed(futures):
            api_name, q = futures[fut]
            for r in fut.result():
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    combined.append(r)

    api_names = ", ".join(n for n, _ in apis)
    status(f"Web search complete: {len(combined)} unique results from [{api_names}]")
    return combined


# ── Status printer ───────────────────────────────────────────────────────────
def status(msg: str) -> None:
    print(f"[STATUS] {msg}", flush=True)


# ── Pipeline ─────────────────────────────────────────────────────────────────
def process_posts(
    topic_id: int,
    query: str,
    subreddits: list[str],
    tag: str = "",
    original_topic: str | None = None,
    seen_urls: set | None = None,
) -> int:
    prefix = f"[{tag}] " if tag else ""

    def progress(sub, done, total):
        status(f"{prefix}Fetching r/{sub} ({done}/{total})")

    posts = reddit.fetch_topic(query, subreddits, progress_cb=progress, seen_urls=seen_urls)
    status(f"{prefix}Fetched {len(posts)} Reddit posts (deduped)")

    max_upvotes = max((p.get("score", 0) for p in posts), default=1)

    for i, post in enumerate(posts):
        post_id = db.save_post(topic_id, post)
        llm_score = llm.judge_relevance(post, query, original_topic=original_topic)
        hybrid = llm.blend_scores(llm_score, post.get("score", 0), max_upvotes)
        db.update_relevance(post_id, hybrid)
        status(f"{prefix}Judged {i+1}/{len(posts)}: {hybrid:.1f}/10 (llm={llm_score:.0f} upvotes={post['score']}) — {post['title'][:40]}")
        try:
            emb = llm.embed_post(post)
            db.update_post_embedding(post_id, emb)
        except Exception:
            log.exception("embed_post failed for %s", post.get("reddit_id"))
        if hybrid >= RELEVANCE_THRESHOLD:
            try:
                summary = llm.summarize_post(post)
                if summary:
                    db.update_post_summary(post_id, summary)
            except Exception:
                log.exception("summarize_post failed for %s", post.get("reddit_id"))
    return len(posts)


def process_web(
    topic_id: int,
    results: list[dict],
    query: str,
    tag: str = "",
    original_topic: str | None = None,
) -> int:
    """Score, embed, and save a pre-fetched list of web results."""
    prefix = f"[{tag}] " if tag else ""
    status(f"{prefix}Processing {len(results)} web results")
    for i, result in enumerate(results):
        rid = db.save_web_result(topic_id, result)
        score = llm.judge_web_relevance(result, query, original_topic=original_topic)
        db.update_web_relevance(rid, score)
        status(f"{prefix}Judged web {i+1}/{len(results)}: {score:.0f}/10 — {result['title'][:40]}")
        try:
            emb = llm.embed_web_result(result)
            db.update_web_embedding(rid, emb)
        except Exception:
            log.exception("embed_web_result failed for %s", result.get("url"))
        if score >= RELEVANCE_THRESHOLD:
            try:
                summary = llm.summarize_web_result(result)
                if summary:
                    db.update_web_summary(rid, summary)
            except Exception:
                log.exception("summarize_web_result failed for %s", result.get("url"))
    return len(results)


def run(topic: str, subreddits: list[str]):
    """Run the full research pipeline. Returns the report path."""
    db.init_db()
    validate(log=status)

    ok, msg = llm.check_ollama()
    if not ok:
        raise RuntimeError(f"Ollama unreachable: {msg}")
    status(f"Ollama OK: {msg}")

    corrected, was_corrected = llm.correct_query(topic)
    if was_corrected:
        status(f"Autocorrected query: '{topic}' → '{corrected}'")
        topic = corrected

    # Validate subreddits exist
    status("Validating subreddits...")
    subreddits = reddit.filter_valid_subreddits(subreddits)

    sites = _auto_sites(topic)
    status(f"Subreddits: {', '.join(subreddits)} | Sites: {', '.join(sites)}")

    # Build full query list: original + expansions + sub-questions
    status("Expanding query and decomposing into sub-questions...")
    expanded = llm.expand_query(topic)
    sub_questions = llm.decompose_topic(topic)
    all_queries = list(dict.fromkeys(expanded + sub_questions))  # dedupe, preserve order
    status(f"Total queries ({len(all_queries)}): {' | '.join(q[:35] for q in all_queries)}")

    topic_id = db.upsert_topic(topic, subreddits)

    # Shared dedup set across all Reddit fetches
    seen_reddit_urls: set[str] = set()

    # Pass 1: Reddit — one batch per query, shared dedup
    total_posts = 0
    for qi, q in enumerate(all_queries):
        tag = f"Q{qi+1}/{len(all_queries)}"
        total_posts += process_posts(
            topic_id, q, subreddits, tag=tag,
            original_topic=topic, seen_urls=seen_reddit_urls,
        )

    # Pass 1: Brave web — ALL queries fired in parallel, results deduplicated
    total_web = 0
    if brave.is_configured():
        status(f"Running {len(all_queries)} Brave queries in parallel...")
        web_results = _search_web_parallel(all_queries, sites)
        status(f"Brave returned {len(web_results)} unique results across all queries")
        total_web = process_web(topic_id, web_results, topic, tag="Web", original_topic=topic)
    else:
        status("Brave API not configured — skipping web search")

    db.mark_topic_fetched(topic_id)
    kept_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    kept_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    status(f"Pass 1 done — Reddit: {total_posts} ({kept_r} relevant) | Web: {total_web} ({kept_w} relevant)")

    # Iterative gap analysis
    for iteration in range(MAX_RESEARCH_ITERATIONS):
        status(f"Gap analysis pass {iteration + 1}/{MAX_RESEARCH_ITERATIONS}...")
        all_posts = db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD)
        all_web = db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD)
        gap_queries = llm.analyze_gaps(topic, all_posts, all_web)
        if not gap_queries:
            status("No significant gaps — research complete")
            break
        status(f"Found {len(gap_queries)} gap queries: {' | '.join(q[:40] for q in gap_queries)}")
        for gi, gq in enumerate(gap_queries):
            tag = f"Gap{iteration+1}.{gi+1}"
            process_posts(topic_id, gq, subreddits, tag=tag,
                          original_topic=topic, seen_urls=seen_reddit_urls)
        if brave.is_configured():
            status("Running gap Brave queries in parallel...")
            gap_web = _search_web_parallel(gap_queries, sites)
            process_web(topic_id, gap_web, topic, tag=f"GapWeb{iteration+1}", original_topic=topic)
        db.mark_topic_fetched(topic_id)

    # Generate report
    status("Generating report...")
    path = report.generate(topic_id)
    status(f"Report written: {path.name} ({path.stat().st_size} bytes)")

    final_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    final_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    status(f"Final: {final_r} relevant posts, {final_w} relevant web results")

    return path


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ResearchOS headless pipeline")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument(
        "--subreddits",
        default="",
        help="Comma-separated subreddits (auto-detected if empty)",
    )
    parser.add_argument("--version", action="version", version=f"reddit-research {__version__}")
    args = parser.parse_args()

    topic = args.topic.strip()
    if not topic:
        print("ERROR: --topic cannot be empty", file=sys.stderr)
        sys.exit(1)

    if args.subreddits.strip():
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    else:
        subreddits = _auto_subreddits(topic)
        status(f"Auto-selected subreddits: {', '.join(subreddits)}")

    try:
        report_path = run(topic, subreddits)
        # Final line — JRVS parses this to find the report
        print(f"REPORT_PATH:{report_path}")
    except Exception as e:
        log.exception("pipeline failed")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
