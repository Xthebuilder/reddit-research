"""
Interest detection and frontier topic prediction.

Scores all past topics by how much the user cares about them, then asks the
LLM to suggest adjacent topics the user hasn't researched yet.  Results are
persisted to predicted_topics so the cron runner can pick them up.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone

from reddit_research import db, llm
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)

# Minimum interest score to include a topic in frontier generation
_MIN_INTEREST = 0.3
# Cosine similarity threshold — above this means "already covered"
_ALREADY_KNOWN_SIM = 0.85
# How many frontier queries to generate per LLM call
_FRONTIER_PER_CALL = 4
# Cap on auto-research runs per cron cycle
MAX_PREDICTIONS_PER_RUN = 3


def compute_interest_score(topic: dict) -> float:
    """
    Blend three signals into a single 0–∞ interest score.

    Recency (0.3): exponential decay, half-life ≈ 14 days.
    Frequency (0.5): log-dampened re-research count — rewards habit, not obsession.
    Opened (0.2): binary — did the user open the generated report?
    """
    now = datetime.now(timezone.utc)

    recency = 0.0
    for field in ("last_fetched", "created_at"):
        raw = topic.get(field)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_ago = max(0.0, (now - dt).total_seconds() / 86400)
            recency = math.exp(-days_ago / 14)
            break
        except Exception:
            pass

    times = max(1, topic.get("times_researched") or 1)
    frequency = math.log1p(times)

    opened = 1.0 if topic.get("last_opened_at") else 0.0

    return 0.3 * recency + 0.5 * frequency + 0.2 * opened


def get_interest_ranked_topics(limit: int = 10) -> list[dict]:
    """Return all topics sorted by interest score, highest first."""
    topics = db.get_all_topics_with_signals()
    scored = [(compute_interest_score(t), t) for t in topics]
    scored.sort(key=lambda x: -x[0])
    result = []
    for score, topic in scored[:limit]:
        topic["_interest_score"] = round(score, 4)
        result.append(topic)
    return result


def _already_known_queries(topics: list[dict]) -> set[str]:
    return {t["name"].lower().strip() for t in topics}


def _topic_embedding(topic: dict) -> list[float] | None:
    raw = topic.get("embedding")
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


def suggest_frontier_queries(top_topics: list[dict]) -> list[tuple[str, float, list[int]]]:
    """
    Ask the LLM to suggest adjacent topics the user hasn't researched yet.

    Returns list of (query, predicted_score, source_topic_ids).
    Filters out candidates whose embedding is too similar to an existing topic.
    """
    if not top_topics:
        return []

    known_names = _already_known_queries(db.get_all_topics_with_signals())
    topic_summaries = "\n".join(
        f"- {t['name']} (researched {t.get('times_researched', 1)}×, "
        f"interest={t.get('_interest_score', 0):.2f})"
        for t in top_topics[:6]
    )
    source_ids = [t["id"] for t in top_topics[:6]]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge-growth advisor. Given a list of topics someone has "
                "been researching, suggest adjacent topics that would deepen their "
                "understanding or open up related areas they haven't explored yet. "
                "Think: what would a curious expert naturally want to learn next? "
                "Output ONLY a JSON array of short query strings (4–8 words each). "
                "Do not repeat topics already in the list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The user has been researching:\n{topic_summaries}\n\n"
                f"Suggest {_FRONTIER_PER_CALL} adjacent topics they should explore next. "
                "Return ONLY a JSON array like: [\"topic 1\", \"topic 2\"]"
            ),
        },
    ]

    raw_suggestions: list[str] = []
    try:
        result = llm._chat_fast(messages)
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                raw_suggestions = [s for s in parsed if isinstance(s, str) and s.strip()]
    except Exception:
        log.exception("suggest_frontier_queries LLM call failed")

    if not raw_suggestions:
        # Cold-start fallback: simple suffixes on the top topic
        base = top_topics[0]["name"] if top_topics else "linux"
        raw_suggestions = [
            f"{base} advanced techniques",
            f"{base} alternatives",
            f"{base} best practices 2025",
        ]

    # Score and filter candidates
    all_topic_embeddings = [
        (t["id"], _topic_embedding(t))
        for t in db.get_all_topics_with_signals()
        if _topic_embedding(t)
    ]

    results: list[tuple[str, float, list[int]]] = []
    for suggestion in raw_suggestions:
        if suggestion.lower().strip() in known_names:
            continue
        try:
            cand_emb = llm.embed(suggestion)
        except Exception:
            log.warning("embed failed for candidate %r — using score 0.5", suggestion)
            cand_emb = []

        max_sim = 0.0
        avg_top_sim = 0.0
        top_sims: list[float] = []

        for tid, emb in all_topic_embeddings:
            if not cand_emb or not emb:
                continue
            sim = db._cosine_similarity(cand_emb, emb)
            max_sim = max(max_sim, sim)
            if tid in source_ids:
                top_sims.append(sim)

        if max_sim >= _ALREADY_KNOWN_SIM:
            log.debug("Skipping %r — too similar to existing topic (sim=%.2f)", suggestion, max_sim)
            continue

        avg_top_sim = sum(top_sims) / len(top_sims) if top_sims else 0.5
        # Higher sim to interesting topics = more relevant; novelty penalty from max_sim
        predicted_score = avg_top_sim * (1.0 - max_sim * 0.5)

        results.append((suggestion, round(predicted_score, 4), source_ids))

    results.sort(key=lambda x: -x[1])
    return results


def run_prediction_cycle() -> list[dict]:
    """
    Full prediction cycle: score topics → generate frontier → persist predictions.
    Returns the list of saved predictions.
    """
    top = get_interest_ranked_topics(limit=6)
    interesting = [t for t in top if t.get("_interest_score", 0) >= _MIN_INTEREST]

    if not interesting:
        log.info("predictor: not enough interest signal yet (need more research history)")
        return []

    log.info(
        "predictor: top %d topics by interest: %s",
        len(interesting),
        ", ".join(t["name"] for t in interesting),
    )

    candidates = suggest_frontier_queries(interesting)
    saved: list[dict] = []
    for query, score, source_ids in candidates:
        db.save_prediction(query, source_ids, score)
        saved.append({"query": query, "predicted_score": score, "source_topic_ids": source_ids})
        log.info("predictor: saved prediction %r (score=%.3f)", query, score)

    return saved
