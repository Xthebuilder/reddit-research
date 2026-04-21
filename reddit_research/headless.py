#!/usr/bin/env python3
"""
Headless research pipeline — runs the full fetch/judge/embed/summarize/
gap-analysis/report cycle from the command line without the TUI.

Usage:
    python -m reddit_research.headless --topic "ZFS snapshots" --subreddits "zfs,linux,sysadmin"

Prints progress lines to stdout prefixed with [STATUS].
Prints the final report path as:  REPORT_PATH:/absolute/path/to/report.md
Exit 0 on success, non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from reddit_research import db, llm, report, researcher
from reddit_research._version import __version__
from reddit_research.config import MAX_RESEARCH_ITERATIONS, RELEVANCE_THRESHOLD, validate
from reddit_research.search import brave, reddit
from reddit_research.ui.keywords import DEFAULT_SUBREDDITS
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)


def status(msg: str) -> None:
    print(f"[STATUS] {msg}", flush=True)


def _search_web_parallel(queries: list[str], sites: list[str]) -> list[dict]:
    """Fire all queries across all configured search APIs in parallel."""
    from reddit_research.search import exa, serper, tavily

    apis: list[tuple[str, callable]] = []
    if brave.is_configured():
        apis.append(("brave", lambda q: researcher.search_web_for_sites(q, sites)))
    if tavily.is_configured():
        apis.append(("tavily", lambda q: tavily.search(q, count=15)))
    if serper.is_configured():
        apis.append(("serper", lambda q: serper.search(q, count=15)))
    if exa.is_configured():
        apis.append(("exa", lambda q: exa.search(q, count=10)))

    if not apis:
        status("No web search APIs configured — skipping web search")
        return []

    seen: set[str] = set()
    combined: list[dict] = []
    tasks: list[tuple[str, str, callable]] = [
        (api_name, q, fn) for q in queries for api_name, fn in apis
    ]

    api_names = ", ".join(n for n, _ in apis)
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
            for r in fut.result():
                url = r.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    combined.append(r)

    status(f"Web search complete: {len(combined)} unique results from [{api_names}]")
    return combined


def _process_web_batch(
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


def run(topic: str, subreddits: list[str]) -> "Path":
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

    status("Validating subreddits...")
    subreddits = reddit.filter_valid_subreddits(subreddits)

    sites = researcher.auto_sites(topic)
    status(f"Subreddits: {', '.join(subreddits)} | Sites: {', '.join(sites)}")

    status("Expanding query and decomposing into sub-questions...")
    expanded = llm.expand_query(topic)
    sub_questions = llm.decompose_topic(topic)
    all_queries = list(dict.fromkeys(expanded + sub_questions))
    status(f"Total queries ({len(all_queries)}): {' | '.join(q[:35] for q in all_queries)}")

    topic_id = db.upsert_topic(topic, subreddits)
    seen_reddit_urls: set[str] = set()

    total_posts = 0
    for qi, q in enumerate(all_queries):
        tag = f"Q{qi+1}/{len(all_queries)}"
        total_posts += researcher.fetch_and_process_posts(
            topic_id, q, subreddits,
            progress=status,
            tag=tag,
            original_topic=topic,
            seen_urls=seen_reddit_urls,
        )

    total_web = 0
    if brave.is_configured():
        web_results = _search_web_parallel(all_queries, sites)
        total_web = _process_web_batch(topic_id, web_results, topic, tag="Web", original_topic=topic)
    else:
        status("Brave API not configured — skipping web search")

    db.mark_topic_fetched(topic_id)
    kept_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    kept_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    status(f"Pass 1 done — Reddit: {total_posts} ({kept_r} relevant) | Web: {total_web} ({kept_w} relevant)")

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
            researcher.fetch_and_process_posts(
                topic_id, gq, subreddits, progress=status, tag=tag,
                original_topic=topic, seen_urls=seen_reddit_urls,
            )
        if brave.is_configured():
            gap_web = _search_web_parallel(gap_queries, sites)
            _process_web_batch(topic_id, gap_web, topic, tag=f"GapWeb{iteration+1}", original_topic=topic)
        db.mark_topic_fetched(topic_id)

    status("Generating report...")
    path = report.generate(topic_id, question=topic)
    status(f"Report written: {path.name} ({path.stat().st_size} bytes)")

    final_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    final_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    status(f"Final: {final_r} relevant posts, {final_w} relevant web results")

    return path


def main():
    parser = argparse.ArgumentParser(description="reddit-research headless pipeline")
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
        subreddits = researcher.auto_subreddits(topic)
        status(f"Auto-selected subreddits: {', '.join(subreddits)}")

    try:
        report_path = run(topic, subreddits)
        print(f"REPORT_PATH:{report_path}")
    except Exception as e:
        log.exception("pipeline failed")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
