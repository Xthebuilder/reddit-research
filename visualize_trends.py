"""
Visualize per-subreddit trends from the research DB.

Queries the posts table and renders a two-panel bar chart:
    - Average post score per subreddit
    - Average number of comments (Reddit's raw num_comments) per subreddit

Writes ./subreddit_analysis.png.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import BASE_DIR, DB_PATH

OUTPUT_PATH = BASE_DIR / "subreddit_analysis.png"


def load_posts(db_path: str) -> pd.DataFrame:
    query = """
        SELECT subreddit, score, num_comments, relevance_score, comments
        FROM posts
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def _stored_comment_count(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def build_chart(df: pd.DataFrame, output: Path) -> Path:
    df = df.copy()
    df["num_comments"] = df["num_comments"].fillna(0).astype(int)
    df["stored_comment_count"] = df["comments"].apply(_stored_comment_count)

    agg = (
        df.groupby("subreddit", as_index=False)
        .agg(
            avg_score=("score", "mean"),
            avg_comments=("num_comments", "mean"),
        )
        .sort_values("avg_score", ascending=False)
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_score, ax_comments) = plt.subplots(
        nrows=1, ncols=2, figsize=(14, 6), sharey=False
    )

    sns.barplot(
        data=agg,
        x="subreddit",
        y="avg_score",
        ax=ax_score,
        hue="subreddit",
        palette="viridis",
        legend=False,
    )
    ax_score.set_title("Average Post Score by Subreddit")
    ax_score.set_xlabel("Subreddit")
    ax_score.set_ylabel("Average Score (upvotes)")
    ax_score.tick_params(axis="x", rotation=45)
    for label in ax_score.get_xticklabels():
        label.set_ha("right")

    sns.barplot(
        data=agg.sort_values("avg_comments", ascending=False),
        x="subreddit",
        y="avg_comments",
        ax=ax_comments,
        hue="subreddit",
        palette="magma",
        legend=False,
    )
    ax_comments.set_title("Average Comment Count by Subreddit")
    ax_comments.set_xlabel("Subreddit")
    ax_comments.set_ylabel("Average Comments on Reddit")
    ax_comments.tick_params(axis="x", rotation=45)
    for label in ax_comments.get_xticklabels():
        label.set_ha("right")

    fig.suptitle("Subreddit Trend Analysis", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    df = load_posts(DB_PATH)
    if df.empty:
        print(f"No posts found in {DB_PATH} — run a fetch first.", file=sys.stderr)
        return 1

    path = build_chart(df, OUTPUT_PATH)
    print(f"Wrote {path}  ({len(df)} posts, {df['subreddit'].nunique()} subreddits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
