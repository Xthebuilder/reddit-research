"""
Research Memory — cross-topic knowledge reuse and domain tagging.

Each time a research pipeline runs, this module:
  1. Tags the topic with 1-3 domain labels (storage, linux, data-science, ...)
  2. Finds semantically similar past topics via embedding cosine similarity
  3. Injects high-relevance cached sources from those past topics into the
     current topic so the pipeline starts with prior knowledge rather than zero

This makes repeated research on related subjects faster and more accurate
without re-fetching content that's already been processed.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from reddit_research import db, llm
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)

# Controlled vocabulary — LLM must pick from this list to prevent hallucination
KNOWN_DOMAINS = [
    "storage", "linux", "devops", "networking", "security", "programming",
    "machine-learning", "llm", "data-science", "hardware", "software",
    "travel", "finance", "career", "health", "fitness", "science",
    "psychology", "gaming", "writing", "music", "media", "general",
]

_UTM_PARAMS = frozenset([
    "utm_source", "utm_medium", "utm_campaign", "utm_term",
    "utm_content", "ref", "source", "fbclid", "gclid",
])

# Minimum cosine similarity to consider a past topic "related"
_TOPIC_SIM_THRESHOLD = 0.55
# Minimum cosine similarity for a cached source to be worth injecting
_SOURCE_SIM_THRESHOLD = 0.60
# Only inject sources that passed the relevance bar when originally judged
_MIN_SOURCE_RELEVANCE = 5.0


def normalize_url(url: str) -> str:
    """Strip tracking params, lowercase scheme+netloc, remove trailing slash."""
    try:
        p = urlparse(url.strip())
        qs = {
            k: v for k, v in parse_qs(p.query, keep_blank_values=True).items()
            if k.lower() not in _UTM_PARAMS
        }
        cleaned_query = urlencode(sorted(qs.items()), doseq=True)
        return urlunparse((
            p.scheme.lower(),
            p.netloc.lower(),
            p.path.rstrip("/") or "/",
            p.params,
            cleaned_query,
            "",  # drop fragment
        ))
    except Exception:
        return url.lower().rstrip("/")


def tag_topic_domains(topic_id: int, topic_name: str) -> list[str]:
    """Tag a topic with 1-3 domain labels and persist them."""
    try:
        tags = llm.tag_domains(topic_name)
        for tag in tags:
            domain_id = db.upsert_domain(tag)
            db.link_topic_domain(topic_id, domain_id)
        log.info("memory: tagged topic %d with domains: %s", topic_id, tags)
        return tags
    except Exception:
        log.exception("tag_topic_domains failed for topic_id=%d", topic_id)
        return []


def find_similar_topics(
    query_embedding: list[float],
    exclude_topic_id: int,
    top_k: int = 5,
) -> list[int]:
    """Return topic IDs whose embeddings are most similar to query_embedding."""
    all_embeddings = db.get_topic_embeddings()
    scored: list[tuple[float, int]] = []
    for tid, emb in all_embeddings:
        if tid == exclude_topic_id:
            continue
        sim = db._cosine_similarity(query_embedding, emb)
        if sim >= _TOPIC_SIM_THRESHOLD:
            scored.append((sim, tid))
    scored.sort(reverse=True)
    return [tid for _, tid in scored[:top_k]]


def pull_cross_topic_sources(
    topic_id: int,
    query_embedding: list[float],
    top_k_per_topic: int = 30,
) -> dict:
    """
    Find high-relevance cached web results from similar past topics and inject
    them into the current topic. Returns injection stats for the report banner.
    """
    similar_ids = find_similar_topics(query_embedding, exclude_topic_id=topic_id)
    if not similar_ids:
        return {"injected_topics": 0, "injected_sources": 0}

    injected = 0
    for src_topic_id in similar_ids:
        results = db.get_web_results(
            src_topic_id, min_relevance=_MIN_SOURCE_RELEVANCE, limit=top_k_per_topic
        )
        for r in results:
            if not r.get("embedding"):
                continue
            emb = json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
            if db._cosine_similarity(query_embedding, emb) < _SOURCE_SIM_THRESHOLD:
                continue
            row_id = db.save_web_result_from_memory(topic_id, r)
            if row_id is not None:
                injected += 1

    if injected:
        log.info(
            "memory: injected %d cached sources from %d similar topic(s) into topic %d",
            injected, len(similar_ids), topic_id,
        )
    return {"injected_topics": len(similar_ids), "injected_sources": injected}
