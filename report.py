"""
Generates and auto-updates markdown research reports from the database.

One .md file per topic, saved to REPORTS_DIR.
Auto-updater polls DB mtime and regenerates stale reports in the background.

Enhanced report template includes:
- Executive Summary (LLM-generated key findings)
- Key Themes (synthesized patterns)
- Recommendations (actionable takeaways)
- Detailed Sources (Reddit + Web)
- Related Questions (from Q&A session)
"""
import atexit
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import db
import llm
from config import BASE_DIR, RELEVANCE_THRESHOLD
from logging_config import get_logger

log = get_logger(__name__)

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))
POLL_INTERVAL = int(os.getenv("REPORT_POLL_INTERVAL", "60"))  # seconds


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip().replace(" ", "_")


def report_path(topic: dict) -> Path:
    return REPORTS_DIR / f"{_safe_filename(topic['name'])}.md"


def _safe_truncate(text: str, max_len: int = 1000) -> str:
    """Safely truncate text for LLM context."""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _generate_summary(topic_name: str, posts: list, web_results: list) -> str:
    """Use LLM to generate executive summary from top sources."""
    if not posts and not web_results:
        return "No sources available to summarize."

    try:
        # Build brief context from top sources
        context_parts = []
        for p in posts[:5]:
            context_parts.append(f"• {p['title']} (r/{p['subreddit']}, score: {p['score']})")
            if p.get("content"):
                context_parts.append(f"  {_safe_truncate(p['content'], 300)}")

        for r in web_results[:5]:
            context_parts.append(f"• {r['title']}")
            if r.get("description"):
                context_parts.append(f"  {_safe_truncate(r['description'], 300)}")

        context_text = "\n".join(context_parts)

        prompt = f"""Based on the following research sources about "{topic_name}", 
provide a concise executive summary (3-4 sentences) with the top 2-3 key findings.
Be specific and actionable.

Sources:
{context_text}

Executive Summary:"""

        summary = llm.ask(prompt)
        return summary.strip()
    except Exception:
        log.exception("_generate_summary failed for %r", topic_name)
        return f"_Summary unavailable (LLM error). Check {len(posts)} posts and {len(web_results)} web results below._"


def _extract_themes(topic_name: str, posts: list, web_results: list) -> str:
    """Extract common themes/patterns from the sources."""
    if not posts and not web_results:
        return "No sources available for theme extraction."

    try:
        # Collect titles and key snippets
        titles = []
        for p in posts[:10]:
            titles.append(p['title'])
        for r in web_results[:10]:
            titles.append(r['title'])

        titles_text = "\n".join([f"- {t}" for t in titles])

        prompt = f"""Analyze the following titles and sources about "{topic_name}".
Identify 3-4 major themes or recurring topics mentioned across them.
Format as a bullet list with brief explanations.

Titles/Topics:
{titles_text}

Key Themes:"""

        themes = llm.ask(prompt)
        return themes.strip()
    except Exception:
        log.exception("_extract_themes failed for %r", topic_name)
        return "_Themes extraction unavailable (LLM error). See detailed sources below._"


def _detect_comparison_topic(topic_name: str) -> bool:
    """Return True if the topic looks like a comparison/ranking/options query."""
    keywords = [
        "best", "top", "vs", "versus", "compare", "comparison", "alternatives",
        "countries", "options", "which", "ranking", "recommend", "should i",
    ]
    normalized = topic_name.lower()
    return any(kw in normalized for kw in keywords)


def _generate_comparison_table(topic_name: str, posts: list, web_results: list) -> str:
    """Generate a structured markdown comparison table from the top sources."""
    context_parts: list[str] = []
    for p in posts[:8]:
        text = p.get("summary") or p.get("content", "")
        context_parts.append(f"• {p['title']}: {text[:300]}")
        if p.get("comments"):
            context_parts.append("  Comments: " + " | ".join(c[:120] for c in p["comments"][:3]))
    for r in web_results[:5]:
        text = r.get("summary") or r.get("description", "")
        context_parts.append(f"• {r['title']}: {text[:300]}")

    context_text = "\n".join(context_parts)
    prompt = (
        f"Based on this research about \"{topic_name}\", extract the main options/items being "
        "compared and create a structured comparison table.\n\n"
        f"Sources:\n{context_text}\n\n"
        "Create a proper markdown table with relevant comparison columns based on what sources discuss. "
        "Only include items explicitly mentioned. Use concise cell values."
    )
    try:
        return llm.ask(prompt).strip()
    except Exception:
        log.exception("_generate_comparison_table failed for %r", topic_name)
        return ""


def _extract_entity_sentiment(topic_name: str, posts: list, web_results: list) -> str:
    """Extract entities and community sentiment from all sources."""
    snippets: list[str] = []
    for p in posts[:12]:
        snippets.append(p.get("summary") or p["title"])
        snippets.extend(c[:150] for c in p.get("comments", [])[:3])
    for r in web_results[:8]:
        snippets.append(r.get("summary") or r.get("description", ""))

    combined = "\n".join(filter(None, snippets))[:3500]
    prompt = (
        f"From the following research about \"{topic_name}\", identify all entities mentioned "
        "(countries, places, products, tools, etc.) and their overall community sentiment.\n\n"
        f"Research text:\n{combined}\n\n"
        "For each entity, output: • EntityName — sentiment (positive/mixed/negative): one-line reason\n"
        "Order by most frequently mentioned first."
    )
    try:
        return llm.ask(prompt).strip()
    except Exception:
        log.exception("_extract_entity_sentiment failed for %r", topic_name)
        return ""


def _generate_recommendations(topic_name: str, posts: list, web_results: list, qa_content: str = "") -> str:
    """Generate actionable recommendations based on sources."""
    if not posts and not web_results:
        return "No sources available for recommendations."

    try:
        # Build context from top sources
        context_parts = []
        for p in posts[:8]:
            context_parts.append(f"Reddit - {p['title']}")
            if p.get("content"):
                context_parts.append(_safe_truncate(p['content'], 250))

        for r in web_results[:8]:
            context_parts.append(f"Web - {r['title']}")
            if r.get("description"):
                context_parts.append(_safe_truncate(r['description'], 250))

        context_text = "\n".join(context_parts)

        prompt = f"""Based on this research about "{topic_name}", provide 3-5 actionable recommendations or best practices.
Be specific, practical, and grounded in what the sources say.

Research context:
{context_text}

{f"User questions: {qa_content}" if qa_content else ""}

Recommendations:"""

        recommendations = llm.ask(prompt)
        return recommendations.strip()
    except Exception:
        log.exception("_generate_recommendations failed for %r", topic_name)
        return "_Recommendations unavailable (LLM error). Review sources and Q&A below for insights._"


def generate(topic_id: int) -> Path:
    """Generate (or regenerate) the markdown report for a topic. Returns the file path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    topic = db.get_topic(topic_id)
    if not topic:
        raise ValueError(f"Topic {topic_id} not found")

    posts = db.get_posts(topic_id, min_relevance=-1)
    relevant_posts = [p for p in posts if p.get("relevance_score", -1) >= RELEVANCE_THRESHOLD]
    all_posts = sorted(posts, key=lambda p: p.get("relevance_score", 0), reverse=True)

    web = db.get_web_results(topic_id, min_relevance=-1)
    relevant_web = [r for r in web if r.get("relevance_score", -1) >= RELEVANCE_THRESHOLD]
    all_web = sorted(web, key=lambda r: r.get("relevance_score", 0), reverse=True)

    session_id = db.get_or_create_session(topic_id)
    messages = db.get_messages(session_id, limit=100)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fetched = topic.get("last_fetched", "never")
    if fetched and fetched != "never":
        fetched = fetched[:16].replace("T", " ") + " UTC"

    lines = [
        f"# Research Report: {topic['name']}",
        "",
        f"**Generated:** {now}  ",
        f"**Last fetched:** {fetched}  ",
        f"**Reddit posts:** {len(posts)} collected, {len(relevant_posts)} above threshold ({RELEVANCE_THRESHOLD}/10)  ",
        f"**Web results:** {len(web)} collected, {len(relevant_web)} above threshold ({RELEVANCE_THRESHOLD}/10)  ",
        f"**Subreddits:** {', '.join(json.loads(topic['subreddits']))}",
        "",
    ]

    # --- Executive Summary ---
    lines += [
        "## 📌 Executive Summary",
        "",
    ]
    if all_posts or all_web:
        summary = _generate_summary(topic['name'], all_posts[:10], all_web[:10])
        lines.append(summary)
    else:
        lines.append("_No sources available to generate summary._")
    lines += ["", "---", ""]

    # --- Key Themes ---
    lines += ["## 🎯 Key Themes", ""]
    if all_posts or all_web:
        themes = _extract_themes(topic['name'], all_posts[:10], all_web[:10])
        lines.append(themes)
    else:
        lines.append("_No sources available to extract themes._")
    lines += ["", "---", ""]

    # --- Entity Sentiment ---
    lines += ["## 🧭 Community Sentiment by Entity", ""]
    if all_posts or all_web:
        sentiment = _extract_entity_sentiment(topic['name'], all_posts[:12], all_web[:8])
        lines.append(sentiment if sentiment else "_Sentiment extraction unavailable._")
    else:
        lines.append("_No sources available._")
    lines += ["", "---", ""]

    # --- Comparison Table (for ranking/comparison topics) ---
    if _detect_comparison_topic(topic['name']) and (all_posts or all_web):
        lines += ["## 📊 Comparison Table", ""]
        table = _generate_comparison_table(topic['name'], all_posts[:8], all_web[:5])
        lines.append(table if table else "_Comparison table unavailable._")
        lines += ["", "---", ""]

    # --- Recommendations ---
    lines += ["## ✅ Recommendations", ""]
    if all_posts or all_web:
        qa_snippets = " | ".join([m['content'][:100] for m in messages if m['role'] == 'user'][:3])
        recommendations = _generate_recommendations(topic['name'], all_posts[:10], all_web[:10], qa_snippets)
        lines.append(recommendations)
    else:
        lines.append("_No sources available to generate recommendations._")
    lines += ["", "---", ""]

    # --- Reddit sources (detailed) ---
    lines += ["## 📖 Detailed Sources", ""]
    lines += ["### Reddit Posts", ""]
    if not all_posts:
        lines.append("_No Reddit posts fetched yet._")
    else:
        for i, p in enumerate(all_posts[:15], 1):
            score = p.get("relevance_score", -1)
            badge = f"{score:.0f}/10" if score >= 0 else "unscored"
            lines.append(f"**R{i}. {p['title']}**")
            lines.append(f"> r/{p['subreddit']} | Score: {p['score']} | Relevance: {badge}")
            lines.append(f"> {p['url']}")
            if p.get("summary"):
                lines += ["", f"**Summary:** {p['summary']}"]
            elif p.get("content"):
                lines += ["", f"{p['content'][:400]}"]
            if p.get("comments"):
                lines.append("")
                lines.append("**Comments:**")
                for c in p["comments"][:2]:
                    lines.append(f"> {c[:250]}")
            lines.append("")

    # --- Web sources (detailed) ---
    lines += ["### Web Sources", ""]
    if not all_web:
        lines.append("_No web results fetched yet._")
    else:
        for i, r in enumerate(all_web[:15], 1):
            score = r.get("relevance_score", -1)
            badge = f"{score:.0f}/10" if score >= 0 else "unscored"
            lines.append(f"**W{i}. {r['title']}**")
            lines.append(f"> Relevance: {badge} | {r['url']}")
            if r.get("summary"):
                lines += ["", f"**Summary:** {r['summary']}"]
            elif r.get("description"):
                lines += ["", f"{r['description'][:400]}"]
            if r.get("extra_snippets"):
                for s in r["extra_snippets"][:1]:
                    lines.append(f"> {s[:250]}")
            lines.append("")

    # --- Q&A Session ---
    lines += ["---", "", "## 💬 Questions & Answers", ""]
    if not messages:
        lines.append("_No questions asked yet._")
    else:
        for msg in messages:
            if msg["role"] == "user":
                lines.append(f"**Q:** {msg['content']}")
            else:
                lines.append(f"**A:** {msg['content']}")
            lines.append("")

    lines += ["---", "", "_Report auto-generated by reddit-research tool_"]

    path = report_path(topic)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_all() -> list[Path]:
    """Regenerate reports for all topics in the DB."""
    paths = []
    for topic in db.list_topics():
        try:
            path = generate(topic["id"])
            paths.append(path)
        except Exception:
            log.exception("generate failed for topic %s", topic.get("id"))
    return paths


# ---------------------------------------------------------------------------
# Auto-updater: background thread watches DB mtime, regenerates on change
# ---------------------------------------------------------------------------

class ReportWatcher:
    def __init__(self, on_update=None):
        """
        on_update(topic_name, path) is called after each report is regenerated.
        Registers an atexit hook so the watcher thread is stopped cleanly
        even if the caller forgets to call stop().
        """
        self._on_update = on_update
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="report-watcher"
        )
        self._last_mtime: float = 0

    def start(self):
        if self._thread.is_alive():
            return
        self._thread.start()
        atexit.register(self.stop)

    def stop(self):
        self._stop.set()
        # Give the loop one poll interval to see the stop flag and exit.
        if self._thread.is_alive():
            self._thread.join(timeout=max(2, POLL_INTERVAL // 10))

    def _run(self):
        from config import DB_PATH
        db_path = Path(DB_PATH)
        while not self._stop.wait(POLL_INTERVAL):
            try:
                if not db_path.exists():
                    continue
                mtime = db_path.stat().st_mtime
                if mtime <= self._last_mtime:
                    continue
                self._last_mtime = mtime
                for topic in db.list_topics():
                    rpath = report_path(topic)
                    # regenerate if report doesn't exist or DB is newer
                    if not rpath.exists() or mtime > rpath.stat().st_mtime:
                        try:
                            path = generate(topic["id"])
                            if self._on_update:
                                self._on_update(topic["name"], path)
                        except Exception:
                            log.exception("watcher: regen failed for topic %s", topic.get("id"))
            except Exception:
                log.exception("watcher: loop iteration failed")
