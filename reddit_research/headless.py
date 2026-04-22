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

# Load .env before any config/search modules are imported so env vars are set
from reddit_research.utils.env_loader import load_env
from reddit_research.utils.logging_config import configure_logging
load_env()
configure_logging()

import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed  # used in _search_web_parallel

# Exit cleanly on SIGTERM (sent by the bash wrapper on Ctrl+C)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

from reddit_research import db, llm, memory, report, researcher
from reddit_research.utils.resources import io_worker_count, safe_worker_count, system_summary
from reddit_research._version import __version__
from reddit_research.config import MAX_RESEARCH_ITERATIONS, RELEVANCE_THRESHOLD, validate
from reddit_research.search import brave, reddit
from reddit_research.ui.keywords import DEFAULT_SUBREDDITS
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)


_PIPE_BROKEN = False

def status(msg: str) -> None:
    global _PIPE_BROKEN
    if _PIPE_BROKEN:
        return
    try:
        print(f"[STATUS] {msg}", flush=True)
    except (BrokenPipeError, OSError):
        _PIPE_BROKEN = True  # pipe closed — stop printing but keep pipeline running


def _search_web_parallel(queries: list[str], sites: list[str], skip_urls: set[str] | None = None) -> list[dict]:
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

    seen: set[str] = set(skip_urls or [])
    combined: list[dict] = []
    tasks: list[tuple[str, str, callable]] = [
        (api_name, q, fn) for q in queries for api_name, fn in apis
    ]

    api_names = ", ".join(n for n, _ in apis)
    io_workers = min(len(tasks), io_worker_count())
    status(f"Firing {len(tasks)} web searches with {io_workers} parallel I/O workers ({len(apis)} APIs × {len(queries)} queries)...")

    def _fetch(api_name: str, q: str, fn) -> list[dict]:
        try:
            results = fn(q)
            for r in results:
                r.setdefault("source", api_name)
            return results
        except Exception:
            log.exception("%s search failed for %r", api_name, q)
            return []

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
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
    """Batch-embed then parallel judge+summarize. Skips items already done."""
    if not results:
        return 0

    # Step 1: persist + snapshot state
    saved: list[tuple[int, dict, dict]] = []
    for r in results:
        rid = db.save_web_result(topic_id, r)
        saved.append((rid, r, db.get_web_processing_state(rid)))

    # Step 2: batch-embed in one Ollama call
    need_embed = [(rid, r) for rid, r, s in saved if not s["embedding"]]
    if need_embed:
        status(f"[{tag}] Batch-embedding {len(need_embed)} web results in 1 API call...")
        try:
            vectors = llm.embed_batch([researcher._web_embed_text(r) for _, r in need_embed])
            for (rid, _), vec in zip(need_embed, vectors):
                if vec:
                    db.update_web_embedding(rid, vec)
            saved = [(rid, r, db.get_web_processing_state(rid)) for rid, r, _ in saved]
        except Exception:
            log.exception("batch embed failed; per-item embeds will run inside worker pool")

    # Step 3: parallel judge + summarize
    workers, worker_reason = safe_worker_count()
    status(f"[{tag}] Judging+summarizing {len(results)} web results — {worker_reason}")

    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(researcher._judge_and_summarize_web, rid, r, s, query, original_topic): r
            for rid, r, s in saved
        }
        for fut in as_completed(futures):
            done_count += 1
            try:
                title, score, fresh = fut.result()
                label = f"{score:.0f}/10" if fresh else f"{score:.0f}/10 (cached)"
                status(f"[{tag}] {done_count}/{len(results)}: {title}... → {label}")
            except Exception:
                log.exception("web result processing failed")
                status(f"[{tag}] {done_count}/{len(results)}: ERROR")

    return len(results)


def run(topic: str, subreddits: list[str]) -> "Path":
    """Run the full research pipeline. Returns the report path."""
    db.init_db()
    validate(log=status)

    ok, msg = llm.check_ollama()
    if not ok:
        raise RuntimeError(f"Ollama unreachable: {msg}")
    status(f"Ollama OK: {msg}")

    workers, worker_reason = safe_worker_count()
    status(f"System: {system_summary()}")
    status(f"Parallel workers: {worker_reason}")

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

    # --- Research Memory: tag domains, embed topic, pull cached sources ---
    status("Building research memory...")
    topic_emb: list[float] | None = None
    mem_stats: dict = {"injected_topics": 0, "injected_sources": 0}
    try:
        topic_emb = llm.embed(topic)
        db.save_topic_embedding(topic_id, topic_emb)
        domains = memory.tag_topic_domains(topic_id, topic)
        if domains:
            status(f"Domain tags: {', '.join(domains)}")
        mem_stats = memory.pull_cross_topic_sources(topic_id, topic_emb)
        if mem_stats["injected_sources"]:
            status(
                f"Memory: loaded {mem_stats['injected_sources']} cached sources "
                f"from {mem_stats['injected_topics']} similar past session(s)"
            )
        else:
            status("Memory: no similar past sessions found — starting fresh")
    except Exception:
        log.exception("memory initialisation failed — continuing without memory")

    # Resume: skip already-fetched Reddit posts for this topic
    seen_reddit_urls: set[str] = db.get_existing_post_urls(topic_id)
    if seen_reddit_urls:
        status(f"Resume: {len(seen_reddit_urls)} Reddit posts already in DB — skipping")

    # Global web URL dedup: never re-fetch a URL seen in any past topic
    seen_web_urls: set[str] = db.get_all_web_urls_globally()
    existing_web_count = len(db.get_existing_web_urls(topic_id))
    if existing_web_count:
        status(f"Resume: {existing_web_count} web results already in DB — skipping")

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
        web_results = _search_web_parallel(all_queries, sites, skip_urls=seen_web_urls)
        seen_web_urls.update(r.get("url", "") for r in web_results)
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
            gap_web = _search_web_parallel(gap_queries, sites, skip_urls=seen_web_urls)
            seen_web_urls.update(r.get("url", "") for r in gap_web)
            _process_web_batch(topic_id, gap_web, topic, tag=f"GapWeb{iteration+1}", original_topic=topic)
        db.mark_topic_fetched(topic_id)

    status("Generating report...")
    path = report.generate(topic_id, question=topic, memory_stats=mem_stats)
    status(f"Report written: {path.name} ({path.stat().st_size} bytes)")

    final_r = len(db.get_posts(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    final_w = len(db.get_web_results(topic_id, min_relevance=RELEVANCE_THRESHOLD))
    status(f"Final: {final_r} relevant posts, {final_w} relevant web results")

    status("Unloading models from VRAM...")
    llm.unload_models()

    return path


def main():
    parser = argparse.ArgumentParser(description="reddit-research headless pipeline")
    parser.add_argument("--topic", required=False, default="", help="Research topic")
    parser.add_argument(
        "--subreddits",
        default="",
        help="Comma-separated subreddits (auto-detected if empty)",
    )
    parser.add_argument(
        "--clear-embeddings",
        action="store_true",
        help="Wipe all stored embeddings (run after changing OLLAMA_EMBED_MODEL)",
    )
    parser.add_argument("--version", action="version", version=f"reddit-research {__version__}")
    args = parser.parse_args()

    if args.clear_embeddings:
        db.init_db()
        db.clear_embeddings()
        status("All embeddings cleared — they will be re-computed on next run")
        return

    topic = args.topic.strip()
    if not topic:
        parser.error("--topic is required (unless using --clear-embeddings)")

    if args.subreddits.strip():
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    else:
        subreddits = researcher.auto_subreddits(topic, llm_fallback=True)
        status(f"Auto-selected subreddits: {', '.join(subreddits)}")

    try:
        report_path = run(topic, subreddits)
        print(f"REPORT_PATH:{report_path}")
        # Print the report content so callers that don't use the bash wrapper get it inline
        try:
            print(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    except Exception as e:
        log.exception("pipeline failed")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
