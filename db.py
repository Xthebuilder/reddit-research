import json
import math
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def _now_iso() -> str:
    """Timezone-aware UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                subreddits TEXT NOT NULL,
                persona TEXT,
                created_at TEXT NOT NULL,
                last_fetched TEXT
            );

            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL REFERENCES topics(id),
                reddit_id TEXT NOT NULL,
                subreddit TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                score INTEGER DEFAULT 0,
                num_comments INTEGER DEFAULT 0,
                content TEXT,
                comments TEXT,
                relevance_score REAL DEFAULT -1,
                embedding TEXT,
                summary TEXT,
                fetched_at TEXT NOT NULL,
                UNIQUE(topic_id, reddit_id)
            );

            CREATE TABLE IF NOT EXISTS web_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL REFERENCES topics(id),
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                extra_snippets TEXT,
                age TEXT,
                relevance_score REAL DEFAULT -1,
                embedding TEXT,
                summary TEXT,
                fetched_at TEXT NOT NULL,
                UNIQUE(topic_id, url)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER REFERENCES topics(id),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        # Migrate existing tables: add new columns if missing
        _migrate_add_column(conn, "posts", "embedding", "TEXT")
        _migrate_add_column(conn, "posts", "summary", "TEXT")
        _migrate_add_column(conn, "posts", "num_comments", "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "web_results", "embedding", "TEXT")
        _migrate_add_column(conn, "web_results", "summary", "TEXT")
        _migrate_add_column(conn, "topics", "persona", "TEXT")


def _migrate_add_column(conn, table: str, column: str, col_type: str):
    """Add a column if it doesn't already exist (safe for re-runs)."""
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


# --- Vector math (no numpy needed) ---

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def vector_search_posts(topic_id: int, query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Find the most semantically similar posts to a query embedding."""
    posts = get_posts(topic_id, min_relevance=-1, limit=200)
    scored = []
    for p in posts:
        if not p.get("embedding"):
            continue
        emb = json.loads(p["embedding"]) if isinstance(p["embedding"], str) else p["embedding"]
        sim = _cosine_similarity(query_embedding, emb)
        scored.append((sim, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]]


def vector_search_web(topic_id: int, query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Find the most semantically similar web results to a query embedding."""
    results = get_web_results(topic_id, min_relevance=-1, limit=200)
    scored = []
    for r in results:
        if not r.get("embedding"):
            continue
        emb = json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
        sim = _cosine_similarity(query_embedding, emb)
        scored.append((sim, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


# --- Topics ---

def upsert_topic(name: str, subreddits: list[str], persona: str | None = None) -> int:
    now = _now_iso()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO topics (name, subreddits, persona, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET subreddits=excluded.subreddits,
                 persona=COALESCE(excluded.persona, topics.persona)""",
            (name, json.dumps(subreddits), persona, now),
        )
        row = conn.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
        return row["id"]


def update_topic_persona(topic_id: int, persona: str):
    with get_conn() as conn:
        conn.execute("UPDATE topics SET persona=? WHERE id=?", (persona, topic_id))


def list_topics() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM topics ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_topic(topic_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
        return dict(row) if row else None


def delete_topic(topic_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE topic_id=?)", (topic_id,))
        conn.execute("DELETE FROM sessions WHERE topic_id=?", (topic_id,))
        conn.execute("DELETE FROM posts WHERE topic_id=?", (topic_id,))
        conn.execute("DELETE FROM web_results WHERE topic_id=?", (topic_id,))
        conn.execute("DELETE FROM topics WHERE id=?", (topic_id,))


def mark_topic_fetched(topic_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE topics SET last_fetched=? WHERE id=?",
            (_now_iso(), topic_id),
        )


# --- Posts ---

def save_post(topic_id: int, post: dict) -> int:
    now = _now_iso()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO posts
               (topic_id, reddit_id, subreddit, title, url, score, num_comments,
                content, comments, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(topic_id, reddit_id) DO UPDATE SET
                 score=excluded.score,
                 num_comments=excluded.num_comments,
                 content=excluded.content,
                 comments=excluded.comments,
                 fetched_at=excluded.fetched_at""",
            (
                topic_id,
                post["reddit_id"],
                post["subreddit"],
                post["title"],
                post["url"],
                post.get("score", 0),
                post.get("num_comments", 0),
                post.get("content", ""),
                json.dumps(post.get("comments", [])),
                now,
            ),
        )
        row = conn.execute(
            "SELECT id FROM posts WHERE topic_id=? AND reddit_id=?",
            (topic_id, post["reddit_id"]),
        ).fetchone()
        return row["id"]


def update_relevance(post_id: int, score: float):
    with get_conn() as conn:
        conn.execute("UPDATE posts SET relevance_score=? WHERE id=?", (score, post_id))


def update_post_embedding(post_id: int, embedding: list[float]):
    with get_conn() as conn:
        conn.execute("UPDATE posts SET embedding=? WHERE id=?", (json.dumps(embedding), post_id))


def update_post_summary(post_id: int, summary: str):
    with get_conn() as conn:
        conn.execute("UPDATE posts SET summary=? WHERE id=?", (summary, post_id))


def get_posts(topic_id: int, min_relevance: float = -1, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM posts
               WHERE topic_id=? AND relevance_score >= ?
               ORDER BY relevance_score DESC, score DESC
               LIMIT ?""",
            (topic_id, min_relevance, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["comments"] = json.loads(d["comments"] or "[]")
            result.append(d)
        return result


# --- Web results ---

def save_web_result(topic_id: int, result: dict) -> int:
    now = _now_iso()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO web_results
               (topic_id, url, title, description, extra_snippets, age, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(topic_id, url) DO UPDATE SET
                 title=excluded.title,
                 description=excluded.description,
                 extra_snippets=excluded.extra_snippets,
                 age=excluded.age,
                 fetched_at=excluded.fetched_at""",
            (
                topic_id,
                result["url"],
                result["title"],
                result.get("description", ""),
                json.dumps(result.get("extra_snippets", [])),
                result.get("age", ""),
                now,
            ),
        )
        row = conn.execute(
            "SELECT id FROM web_results WHERE topic_id=? AND url=?",
            (topic_id, result["url"]),
        ).fetchone()
        return row["id"]


def update_web_relevance(result_id: int, score: float):
    with get_conn() as conn:
        conn.execute("UPDATE web_results SET relevance_score=? WHERE id=?", (score, result_id))


def update_web_embedding(result_id: int, embedding: list[float]):
    with get_conn() as conn:
        conn.execute("UPDATE web_results SET embedding=? WHERE id=?", (json.dumps(embedding), result_id))


def update_web_summary(result_id: int, summary: str):
    with get_conn() as conn:
        conn.execute("UPDATE web_results SET summary=? WHERE id=?", (summary, result_id))


def get_web_results(topic_id: int, min_relevance: float = -1, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM web_results
               WHERE topic_id=? AND relevance_score >= ?
               ORDER BY relevance_score DESC
               LIMIT ?""",
            (topic_id, min_relevance, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["extra_snippets"] = json.loads(d["extra_snippets"] or "[]")
            result.append(d)
        return result


# --- Sessions & messages ---

def new_session(topic_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (topic_id, created_at) VALUES (?, ?)",
            (topic_id, _now_iso()),
        )
        return cur.lastrowid


def get_or_create_session(topic_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM sessions WHERE topic_id=? ORDER BY created_at DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        if row:
            return row["id"]
    return new_session(topic_id)


def add_message(session_id: int, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now_iso()),
        )


def get_messages(session_id: int, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
