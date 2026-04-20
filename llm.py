"""
Ollama API wrapper.
- ask()               → single completion
- judge_relevance()   → LLM-as-judge: scores a post 0-10 for relevance
- build_context()     → formats posts into an LLM context block
- answer()            → RAG-style answer grounded in fetched posts
- embed()             → generate embeddings via Ollama
- expand_query()      → query expansion/reformulation
- summarize_content() → intelligent summarization before storage
- analyze_gaps()      → identify gaps for iterative research
"""
from __future__ import annotations

import json
import re

import httpx

from config import (
    CONTEXT_POSTS,
    DEFAULT_PERSONA,
    MAX_EXPANDED_QUERIES,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
)
from http_client import LONG_TIMEOUT, get_client
from logging_config import get_logger

log = get_logger(__name__)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"


def _chat(messages: list[dict], stream: bool = False, **kwargs) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        **kwargs,
    }
    r = get_client().post(_CHAT_URL, json=payload, timeout=LONG_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _chat_stream(messages: list[dict], on_token=None):
    """Stream tokens; calls on_token(chunk) for each chunk. Returns full text."""
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True}
    full: list[str] = []
    with get_client().stream("POST", _CHAT_URL, json=payload, timeout=LONG_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = data.get("message", {}).get("content", "")
            if token:
                full.append(token)
                if on_token:
                    on_token(token)
            if data.get("done"):
                break
    return "".join(full)


def check_ollama() -> tuple[bool, str]:
    """Returns (ok, message)."""
    try:
        r = get_client().get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, ", ".join(models) if models else "no models found"
    except Exception as e:
        log.warning("Ollama unreachable at %s: %s", OLLAMA_BASE_URL, e)
        return False, str(e)


def list_models() -> list[str]:
    try:
        r = get_client().get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError:
        log.exception("list_models failed")
        return []


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """Generate an embedding vector for the given text using Ollama."""
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text}
    r = get_client().post(_EMBED_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    embeddings = data.get("embeddings", [])
    if embeddings and isinstance(embeddings[0], list):
        return embeddings[0]
    return embeddings


def embed_post(post: dict) -> list[float]:
    """Create an embedding from a post's key content."""
    text = post["title"]
    if post.get("content"):
        text += "\n" + post["content"][:500]
    if post.get("comments"):
        text += "\n" + " ".join(c[:150] for c in post["comments"][:2])
    return embed(text)


def embed_web_result(result: dict) -> list[float]:
    """Create an embedding from a web result's key content."""
    text = result["title"]
    if result.get("description"):
        text += "\n" + result["description"][:500]
    if result.get("extra_snippets"):
        text += "\n" + " ".join(s[:150] for s in result["extra_snippets"][:2])
    return embed(text)


# ---------------------------------------------------------------------------
# Query expansion / reformulation
# ---------------------------------------------------------------------------

def expand_query(query: str, num_queries: int | None = None) -> list[str]:
    """
    Given a user query, generate expanded/reformulated search queries
    using synonyms, related terms, and sub-questions.
    Returns a list of queries (always includes the original).
    """
    n = num_queries or MAX_EXPANDED_QUERIES
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query expansion expert. Given a research query, "
                "generate alternative search queries that would find relevant results. "
                "Include synonyms, related terms, more specific sub-questions, and "
                "different phrasings. Output ONLY a JSON array of strings, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate {n} alternative search queries for: \"{query}\"\n\n"
                "Return ONLY a JSON array like: [\"query 1\", \"query 2\", \"query 3\"]"
            ),
        },
    ]
    try:
        result = _chat(messages)
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                expanded = [q for q in queries if isinstance(q, str) and q.strip()]
                return [query] + expanded[:n]
    except Exception:
        log.exception("expand_query failed for %r", query)
    return [query]


def correct_query(text: str) -> tuple[str, bool]:
    """
    Spell-correct a query using pyspellchecker.
    Returns (corrected_text, was_changed).
    Preserves capitalisation and punctuation; skips short tokens and URLs.
    """
    try:
        from spellchecker import SpellChecker  # type: ignore
        spell = SpellChecker()
        words = text.split()
        corrected = []
        changed = False
        for word in words:
            # Strip surrounding punctuation for checking
            stripped = word.strip(".,!?;:\"'()")
            if not stripped or len(stripped) < 3 or stripped.startswith("http"):
                corrected.append(word)
                continue
            candidate = spell.correction(stripped.lower())
            if candidate and candidate != stripped.lower():
                # Preserve original casing style
                if stripped.isupper():
                    candidate = candidate.upper()
                elif stripped[0].isupper():
                    candidate = candidate.capitalize()
                fixed = word.replace(stripped, candidate)
                corrected.append(fixed)
                changed = True
            else:
                corrected.append(word)
        return " ".join(corrected), changed
    except Exception:
        log.exception("correct_query failed for %r", text)
        return text, False


def decompose_topic(topic: str) -> list[str]:
    """
    Break a broad topic into specific, targeted sub-questions.
    Each sub-question is independently searchable and covers a distinct angle.
    Returns a list of sub-questions (empty on failure).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research strategist. Break a broad research topic into 3-4 "
                "specific, independently searchable sub-questions that together give complete coverage. "
                "Sub-questions must be concrete and specific to the user's context — not generic. "
                "Output ONLY a JSON array of strings, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Break this topic into specific sub-questions: \"{topic}\"\n\n"
                "Return ONLY a JSON array like: [\"sub-question 1\", \"sub-question 2\"]"
            ),
        },
    ]
    try:
        result = _chat(messages)
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            if isinstance(questions, list):
                return [q for q in questions if isinstance(q, str) and q.strip()][:4]
    except Exception:
        log.exception("decompose_topic failed for %r", topic)
    return []


def suggest_subreddits(topic: str, num: int = 6) -> list[str]:
    """
    Use LLM to suggest the best subreddits for a given topic.
    Returns subreddit names without the r/ prefix.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Reddit expert. Given a research topic, suggest the best subreddits "
                "to find high-quality discussion and answers. "
                "ONLY suggest subreddits you are certain exist and have over 100k subscribers. "
                "Common examples: travel→travel,solotravel,backpacking; "
                "finance→personalfinance,financialindependence; "
                "career→careerguidance,cscareerquestions; "
                "health→fitness,nutrition,mentalhealth. "
                "Return only the subreddit name with no r/ prefix and no descriptions. "
                "Output ONLY a JSON array of strings, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Best subreddits for: \"{topic}\"\n"
                f"Return {num} real subreddit names as ONLY a JSON array: [\"sub1\", \"sub2\"]"
            ),
        },
    ]
    try:
        result = _chat(messages)
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            subs = json.loads(match.group())
            if isinstance(subs, list):
                return [
                    s.lstrip("r/").strip()
                    for s in subs
                    if isinstance(s, str) and s.strip()
                ][:num]
    except Exception:
        log.exception("suggest_subreddits failed for %r", topic)
    return []


# ---------------------------------------------------------------------------
# Intelligent summarization
# ---------------------------------------------------------------------------

def summarize_post(post: dict) -> str:
    """Summarize a Reddit post into a concise, information-dense paragraph."""
    snippet = post["title"]
    if post.get("content"):
        snippet += "\n" + post["content"][:800]
    if post.get("comments"):
        snippet += "\nTop comments:\n" + "\n".join(c[:300] for c in post["comments"][:3])

    messages = [
        {
            "role": "system",
            "content": (
                "Summarize the following Reddit post into 2-3 concise sentences. "
                "Capture the key information, any solutions mentioned, and the "
                "community consensus. Preserve specific technical details, commands, "
                "tool names, and version numbers. Output ONLY the summary."
            ),
        },
        {"role": "user", "content": snippet},
    ]
    try:
        return _chat(messages)
    except Exception:
        log.exception("summarize_post failed for %s", post.get("reddit_id"))
        return ""


def summarize_web_result(result: dict) -> str:
    """Summarize a web result into a concise, information-dense paragraph."""
    snippet = result["title"]
    if result.get("description"):
        snippet += "\n" + result["description"][:800]
    if result.get("extra_snippets"):
        snippet += "\n" + "\n".join(s[:300] for s in result["extra_snippets"][:3])

    messages = [
        {
            "role": "system",
            "content": (
                "Summarize the following web page content into 2-3 concise sentences. "
                "Capture the key information and any actionable details. "
                "Preserve specific technical details. Output ONLY the summary."
            ),
        },
        {"role": "user", "content": snippet},
    ]
    try:
        return _chat(messages)
    except Exception:
        log.exception("summarize_web_result failed for %s", result.get("url"))
        return ""


# ---------------------------------------------------------------------------
# Gap analysis / iterative research
# ---------------------------------------------------------------------------

def analyze_gaps(topic: str, posts: list[dict], web_results: list[dict]) -> list[str]:
    """
    Analyze existing research and identify gaps — returns follow-up queries
    for a second pass of research.
    """
    source_titles: list[str] = []
    for p in posts[:10]:
        source_titles.append(f"Reddit: {p['title']}")
    for r in web_results[:10]:
        source_titles.append(f"Web: {r['title']}")

    if not source_titles:
        return []

    titles_text = "\n".join(f"- {t}" for t in source_titles)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research analyst. Given a topic and the sources already found, "
                "identify what's MISSING — gaps in coverage, unanswered sub-questions, "
                "alternative perspectives not yet explored. "
                "Output ONLY a JSON array of 2-3 follow-up search queries that would "
                "fill these gaps. Output nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\n"
                f"Sources already collected:\n{titles_text}\n\n"
                "What follow-up queries would fill gaps in this research? "
                "Return ONLY a JSON array like: [\"query 1\", \"query 2\"]"
            ),
        },
    ]
    try:
        result = _chat(messages)
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str) and q.strip()][:3]
    except Exception:
        log.exception("analyze_gaps failed for topic=%r", topic)
    return []


# ---------------------------------------------------------------------------
# Relevance judging (original + hybrid with embeddings)
# ---------------------------------------------------------------------------

def blend_scores(llm_score: float, upvotes: int, max_upvotes: int) -> float:
    """Blend LLM relevance (70%) with normalized upvote signal (30%)."""
    if max_upvotes <= 0:
        return llm_score
    upvote_norm = min(1.0, upvotes / max_upvotes) * 10.0
    return round(0.7 * llm_score + 0.3 * upvote_norm, 2)


def _judge(snippet: str, topic: str, source_type: str = "content", original_topic: str | None = None) -> float:
    if original_topic and original_topic != topic:
        context = f"Original question: {original_topic}\nSpecific query: {topic}"
    else:
        context = f"Topic: {topic}"
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a relevance judge. Given a research topic and a {source_type} snippet, "
                "output ONLY a single integer from 0 to 10 representing how relevant "
                "it is to the specific topic AND original intent. "
                "0=completely irrelevant, 10=highly relevant. Output nothing else — just the number."
            ),
        },
        {"role": "user", "content": f"{context}\n\nSnippet:\n{snippet}"},
    ]
    try:
        result = _chat(messages)
        match = re.search(r"\d+", result)
        if match:
            return min(10.0, max(0.0, float(match.group())))
    except Exception:
        log.exception("_judge failed for topic=%r", topic)
    return 5.0


def judge_relevance(post: dict, topic: str, original_topic: str | None = None) -> float:
    """LLM-as-judge for a Reddit post. Returns 0.0-10.0."""
    snippet = post["title"]
    if post.get("content"):
        snippet += "\n" + post["content"][:400]
    if post.get("comments"):
        snippet += "\n" + post["comments"][0][:200]
    return _judge(snippet, topic, source_type="Reddit post", original_topic=original_topic)


def judge_web_relevance(result: dict, topic: str, original_topic: str | None = None) -> float:
    """LLM-as-judge for a Brave web result. Returns 0.0-10.0."""
    snippet = result["title"]
    if result.get("description"):
        snippet += "\n" + result["description"][:400]
    if result.get("extra_snippets"):
        snippet += "\n" + result["extra_snippets"][0][:200]
    return _judge(snippet, topic, source_type="web page", original_topic=original_topic)


# ---------------------------------------------------------------------------
# Context building (now uses summaries when available)
# ---------------------------------------------------------------------------

def build_context(posts: list[dict], web_results: list[dict], topic: str) -> str:
    """Format top Reddit posts + web results into a combined context string."""
    top_posts = sorted(posts, key=lambda p: p.get("relevance_score", 0), reverse=True)[:CONTEXT_POSTS]
    top_web = sorted(web_results, key=lambda r: r.get("relevance_score", 0), reverse=True)[:CONTEXT_POSTS]

    parts = [f"Research context for: {topic}\n"]

    if top_posts:
        parts.append(f"=== REDDIT SOURCES ({len(top_posts)}) ===\n")
        for i, p in enumerate(top_posts, 1):
            parts.append(f"--- Reddit {i} ---")
            parts.append(f"r/{p['subreddit']} | Upvotes: {p['score']} | Relevance: {p.get('relevance_score', '?')}/10")
            parts.append(f"Title: {p['title']}")
            parts.append(f"URL: {p['url']}")
            if p.get("summary"):
                parts.append(f"Summary: {p['summary']}")
            elif p.get("content"):
                parts.append(p["content"][:500])
            if p.get("comments"):
                parts.append("Top comments:")
                for c in p["comments"][:3]:
                    parts.append(f"  • {c[:250]}")
            parts.append("")

    if top_web:
        parts.append(f"=== WEB SOURCES ({len(top_web)}) ===\n")
        for i, r in enumerate(top_web, 1):
            parts.append(f"--- Web {i} ---")
            parts.append(f"Relevance: {r.get('relevance_score', '?')}/10")
            parts.append(f"Title: {r['title']}")
            parts.append(f"URL: {r['url']}")
            if r.get("summary"):
                parts.append(f"Summary: {r['summary']}")
            elif r.get("description"):
                parts.append(r["description"][:500])
            if r.get("extra_snippets"):
                for s in r["extra_snippets"][:2]:
                    parts.append(f"  > {s[:250]}")
            parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# RAG answer (now with configurable persona + semantic retrieval)
# ---------------------------------------------------------------------------

def answer(
    question: str,
    posts: list[dict],
    topic: str,
    history: list[dict],
    on_token=None,
    web_results: list[dict] | None = None,
    persona: str | None = None,
) -> str:
    """RAG-style answer grounded in fetched posts + web results, with conversation history."""
    context = build_context(posts, web_results or [], topic)
    persona_text = persona or DEFAULT_PERSONA
    system = (
        f"{persona_text}\n\n"
        f"Current research topic: {topic}\n\n"
        "Use the sources provided to answer questions accurately. "
        "Cite Reddit sources as [Reddit N] and web sources as [Web N]. "
        "If the sources don't contain enough information, say so clearly.\n\n"
        + context
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": question})

    if on_token:
        return _chat_stream(messages, on_token=on_token)
    return _chat(messages)


def ask(prompt: str, on_token=None) -> str:
    """Simple one-shot question, no context."""
    messages = [{"role": "user", "content": prompt}]
    if on_token:
        return _chat_stream(messages, on_token=on_token)
    return _chat(messages)
