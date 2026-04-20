"""Unit tests for db.py schema + CRUD."""
from __future__ import annotations


def test_init_creates_tables(tmp_db):
    with tmp_db.get_conn() as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"topics", "posts", "web_results", "sessions", "messages"}.issubset(tables)


def test_upsert_topic_is_idempotent(tmp_db):
    a = tmp_db.upsert_topic("zfs", ["zfs", "linux"])
    b = tmp_db.upsert_topic("zfs", ["zfs", "linux", "selfhosted"])
    assert a == b
    topic = tmp_db.get_topic(a)
    assert "selfhosted" in topic["subreddits"]


def test_save_post_and_get_posts(tmp_db):
    tid = tmp_db.upsert_topic("t", ["x"])
    post = {
        "reddit_id": "abc",
        "subreddit": "x",
        "title": "hello",
        "url": "https://r/x/abc",
        "score": 1,
        "content": "",
        "comments": [],
    }
    pid = tmp_db.save_post(tid, post)
    tmp_db.update_relevance(pid, 7.0)
    results = tmp_db.get_posts(tid, min_relevance=0)
    assert len(results) == 1
    assert results[0]["title"] == "hello"
    assert results[0]["relevance_score"] == 7.0


def test_save_post_upserts_same_reddit_id(tmp_db):
    tid = tmp_db.upsert_topic("t", ["x"])
    post = {
        "reddit_id": "abc",
        "subreddit": "x",
        "title": "v1",
        "url": "u",
        "score": 1,
        "content": "",
        "comments": [],
    }
    tmp_db.save_post(tid, post)
    post["score"] = 42
    tmp_db.save_post(tid, post)
    results = tmp_db.get_posts(tid, min_relevance=-1)
    assert len(results) == 1
    assert results[0]["score"] == 42


def test_cosine_similarity():
    from db import _cosine_similarity
    assert _cosine_similarity([1, 0], [1, 0]) == 1.0
    assert _cosine_similarity([1, 0], [0, 1]) == 0.0
    assert _cosine_similarity([], [1]) == 0.0
    assert _cosine_similarity([0, 0], [1, 1]) == 0.0


def test_sessions_and_messages(tmp_db):
    tid = tmp_db.upsert_topic("t", ["x"])
    sid = tmp_db.get_or_create_session(tid)
    tmp_db.add_message(sid, "user", "hi")
    tmp_db.add_message(sid, "assistant", "hello")
    msgs = tmp_db.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
