"""
Generates and auto-updates markdown research reports from the database.
One .md + one .html file per topic, saved to REPORTS_DIR.
"""
import atexit
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import markdown as _md

from reddit_research import db, llm
from reddit_research.config import CONTEXT_POSTS, RELEVANCE_THRESHOLD
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(Path.home() / "reports")))
POLL_INTERVAL = int(os.getenv("REPORT_POLL_INTERVAL", "60"))


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip().replace(" ", "_")


def report_path(topic: dict) -> Path:
    return REPORTS_DIR / f"{_safe_filename(topic['name'])}.md"


def _safe_truncate(text: str, max_len: int = 1000) -> str:
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _generate_summary(topic_name: str, posts: list, web_results: list) -> str:
    if not posts and not web_results:
        return "No sources available to summarize."

    try:
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

        prompt = f"""Based on the following research sources about "{topic_name}", \
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
    if not posts and not web_results:
        return "No sources available for theme extraction."

    try:
        titles = [p['title'] for p in posts[:10]] + [r['title'] for r in web_results[:10]]
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
    keywords = [
        "best", "top", "vs", "versus", "compare", "comparison", "alternatives",
        "countries", "options", "which", "ranking", "recommend", "should i",
    ]
    normalized = topic_name.lower()
    return any(kw in normalized for kw in keywords)


def _generate_comparison_table(topic_name: str, posts: list, web_results: list) -> str:
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
    if not posts and not web_results:
        return "No sources available for recommendations."

    try:
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


def _generate_direct_answer(question: str, posts: list, web_results: list) -> str:
    """Generate a direct, grounded answer to the research question using the best sources."""
    top_posts = posts[:CONTEXT_POSTS]
    top_web = web_results[:CONTEXT_POSTS]

    context_parts: list[str] = []
    for i, p in enumerate(top_posts, 1):
        text = p.get("summary") or p.get("content", "")
        context_parts.append(f"[Reddit {i}] r/{p['subreddit']} — {p['title']}\n{text[:400]}")
        if p.get("comments"):
            context_parts.append("Top comment: " + p["comments"][0][:200])
    for i, r in enumerate(top_web, 1):
        text = r.get("summary") or r.get("description", "")
        context_parts.append(f"[Web {i}] {r['title']}\n{text[:400]}")

    context = "\n\n".join(context_parts)

    prompt = (
        f"Answer the following research question as completely and specifically as possible, "
        f"citing the sources provided. Use [Reddit N] and [Web N] inline citations.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Provide a thorough, structured answer with concrete details from the sources. "
        "If sources conflict, note the disagreement. Prioritize specifics over generalities."
    )
    try:
        return llm.ask(prompt).strip()
    except Exception:
        log.exception("_generate_direct_answer failed for %r", question)
        return "_Direct answer unavailable (LLM error)._"


_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 16px; line-height: 1.7; color: #1a1a2e;
    background: #f8f9fa; padding: 0;
}
.page { max-width: 860px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
h1 { font-size: 2rem; font-weight: 700; color: #0f3460; margin: 1.5rem 0 0.5rem; border-bottom: 3px solid #e94560; padding-bottom: 0.4rem; }
h2 { font-size: 1.35rem; font-weight: 700; color: #16213e; margin: 2rem 0 0.6rem; padding-left: 0.6rem; border-left: 4px solid #e94560; }
h3 { font-size: 1.1rem; font-weight: 600; color: #0f3460; margin: 1.4rem 0 0.4rem; }
p { margin: 0.6rem 0; }
a { color: #e94560; text-decoration: none; }
a:hover { text-decoration: underline; }
strong { color: #0f3460; }
em { color: #555; }
hr { border: none; border-top: 1px solid #dde; margin: 1.8rem 0; }
blockquote {
    border-left: 3px solid #ccd; padding: 0.3rem 1rem;
    color: #555; background: #f0f2f8; margin: 0.5rem 0; border-radius: 0 4px 4px 0;
}
ul, ol { padding-left: 1.5rem; margin: 0.5rem 0; }
li { margin: 0.25rem 0; }
code { background: #eef0f8; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.88em; font-family: "SF Mono", "Fira Code", monospace; }
pre { background: #1a1a2e; color: #e2e8f0; padding: 1rem; border-radius: 6px; overflow-x: auto; margin: 1rem 0; }
pre code { background: none; padding: 0; color: inherit; font-size: 0.85em; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.93rem; }
th { background: #0f3460; color: #fff; padding: 0.5rem 0.75rem; text-align: left; }
td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #dde; }
tr:nth-child(even) td { background: #f0f2f8; }
.meta { background: #fff; border: 1px solid #dde; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0 1.5rem; font-size: 0.9rem; color: #444; line-height: 1.9; }
.memory-badge {
    display: inline-block; background: #e8f5e9; color: #2e7d32;
    border: 1px solid #a5d6a7; border-radius: 20px;
    padding: 0.2rem 0.75rem; font-size: 0.82rem; font-weight: 600; margin-top: 0.4rem;
}
@media (max-width: 600px) {
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.15rem; }
    .page { padding: 1rem 1rem 3rem; }
}
@media print {
    body { background: #fff; }
    .page { max-width: 100%; padding: 0; }
    a { color: #000; }
}
"""


def _to_html(md_text: str, title: str) -> str:
    """Convert markdown report text to a self-contained, mobile-friendly HTML file."""
    # Separate the metadata block (lines starting with **Key:**) from the body
    # so we can render it in a styled card rather than as raw bold text.
    lines = md_text.split("\n")
    meta_lines: list[str] = []
    body_lines: list[str] = []
    in_meta = False
    heading_seen = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not heading_seen:
            heading_seen = True
            body_lines.append(line)
            in_meta = True
            continue
        if in_meta:
            if stripped.startswith("**") and ":**" in stripped:
                meta_lines.append(stripped)
                continue
            elif stripped == "" and meta_lines:
                continue  # blank line inside meta block
            else:
                in_meta = False
        body_lines.append(line)

    # Build meta card HTML
    meta_html = ""
    if meta_lines:
        rows = []
        for m in meta_lines:
            # **Memory:** ... badge vs plain line
            if m.startswith("**Memory:**"):
                val = m.replace("**Memory:**", "").strip().rstrip()
                rows.append(f'<span class="memory-badge">🧠 {val}</span>')
            else:
                rows.append(m.rstrip("  "))
        meta_html = '<div class="meta">' + "<br>".join(rows) + "</div>"

    body_md = "\n".join(body_lines)
    body_html = _md.markdown(
        body_md,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="page">
{meta_html}
{body_html}
</div>
</body>
</html>"""


def generate(topic_id: int, question: str | None = None, memory_stats: dict | None = None) -> Path:
    """Generate (or regenerate) the markdown report for a topic.

    If `question` is provided, the report leads with a direct answer section
    built from semantically retrieved sources rather than just top-scored ones.
    Returns the file path.
    """
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

    # Use semantic retrieval when a question is available — gets the *most relevant*
    # sources for this specific question rather than just the highest-scored ones.
    if question and (all_posts or all_web):
        try:
            q_emb = llm.embed(question)
            sem_posts = db.vector_search_posts(topic_id, q_emb, top_k=CONTEXT_POSTS) or all_posts
            sem_web = db.vector_search_web(topic_id, q_emb, top_k=CONTEXT_POSTS) or all_web
        except Exception:
            sem_posts = all_posts
            sem_web = all_web
    else:
        sem_posts = all_posts
        sem_web = all_web

    session_id = db.get_or_create_session(topic_id)
    messages = db.get_messages(session_id, limit=100)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fetched = topic.get("last_fetched", "never")
    if fetched and fetched != "never":
        fetched = fetched[:16].replace("T", " ") + " UTC"

    # Memory banner — show if sources were injected from past sessions
    injected = (memory_stats or {}).get("injected_sources") or db.get_memory_source_count(topic_id)
    injected_topics = (memory_stats or {}).get("injected_topics", 0)
    memory_line = ""
    if injected:
        sessions_label = f"{injected_topics} past session" + ("s" if injected_topics != 1 else "")
        memory_line = f"**Memory:** Enriched with {injected} cached sources from {sessions_label}  "

    lines = [
        f"# Research Report: {topic['name']}",
        "",
        f"**Generated:** {now}  ",
        f"**Last fetched:** {fetched}  ",
        f"**Reddit posts:** {len(posts)} collected, {len(relevant_posts)} above threshold ({RELEVANCE_THRESHOLD}/10)  ",
        f"**Web results:** {len(web)} collected, {len(relevant_web)} above threshold ({RELEVANCE_THRESHOLD}/10)  ",
        f"**Subreddits:** {', '.join(json.loads(topic['subreddits']))}",
    ]
    if memory_line:
        lines.append(memory_line)
    lines.append("")

    # Direct answer section — only present when a question was provided.
    # Uses semantically retrieved sources so the answer is grounded in the
    # most relevant content, not just the highest-scored posts.
    if question and (sem_posts or sem_web):
        lines += [f"## 💡 Direct Answer: {question}", ""]
        lines.append(_generate_direct_answer(question, sem_posts, sem_web))
        lines += ["", "---", ""]

    lines += ["## 📌 Executive Summary", ""]
    if all_posts or all_web:
        summary = _generate_summary(topic['name'], sem_posts[:10], sem_web[:10])
        lines.append(summary)
    else:
        lines.append("_No sources available to generate summary._")
    lines += ["", "---", ""]

    lines += ["## 🎯 Key Themes", ""]
    if all_posts or all_web:
        themes = _extract_themes(topic['name'], sem_posts[:10], sem_web[:10])
        lines.append(themes)
    else:
        lines.append("_No sources available to extract themes._")
    lines += ["", "---", ""]

    lines += ["## 🧭 Community Sentiment by Entity", ""]
    if all_posts or all_web:
        sentiment = _extract_entity_sentiment(topic['name'], sem_posts[:12], sem_web[:8])
        lines.append(sentiment if sentiment else "_Sentiment extraction unavailable._")
    else:
        lines.append("_No sources available._")
    lines += ["", "---", ""]

    if _detect_comparison_topic(topic['name']) and (all_posts or all_web):
        lines += ["## 📊 Comparison Table", ""]
        table = _generate_comparison_table(topic['name'], sem_posts[:8], sem_web[:5])
        lines.append(table if table else "_Comparison table unavailable._")
        lines += ["", "---", ""]

    lines += ["## ✅ Recommendations", ""]
    if all_posts or all_web:
        qa_snippets = question or " | ".join([m['content'][:100] for m in messages if m['role'] == 'user'][:3])
        recommendations = _generate_recommendations(topic['name'], sem_posts[:10], sem_web[:10], qa_snippets)
        lines.append(recommendations)
    else:
        lines.append("_No sources available to generate recommendations._")
    lines += ["", "---", ""]

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
    md_text = "\n".join(lines)
    path.write_text(md_text, encoding="utf-8")

    # Export a self-contained HTML file alongside the markdown
    html_path = path.with_suffix(".html")
    html_path.write_text(_to_html(md_text, topic["name"]), encoding="utf-8")

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


class ReportWatcher:
    def __init__(self, on_update=None):
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
        if self._thread.is_alive():
            self._thread.join(timeout=max(2, POLL_INTERVAL // 10))

    def _run(self):
        from reddit_research.config import DB_PATH
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
                    if not rpath.exists() or mtime > rpath.stat().st_mtime:
                        try:
                            path = generate(topic["id"])
                            if self._on_update:
                                self._on_update(topic["name"], path)
                        except Exception:
                            log.exception("watcher: regen failed for topic %s", topic.get("id"))
            except Exception:
                log.exception("watcher: loop iteration failed")
