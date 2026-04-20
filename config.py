from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from _version import __version__

BASE_DIR = Path(__file__).parent


def _xdg_data_home() -> Path:
    return Path(os.getenv("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))


def _default_db_path() -> str:
    # Preserve backwards compat: if a project-local research.db exists,
    # keep using it. Only new installs go to the XDG location.
    legacy = Path(__file__).parent / "research.db"
    if legacy.exists():
        return str(legacy)
    data_dir = _xdg_data_home() / "reddit-research"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "research.db")


# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Reddit API (optional — falls back to public JSON if not set)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", f"reddit-research/{__version__}")

# Brave Search API
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_MAX_RESULTS = int(os.getenv("BRAVE_MAX_RESULTS", "20"))

# Additional search APIs
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# Storage
DB_PATH = os.getenv("DB_PATH") or _default_db_path()

# Research defaults
DEFAULT_SUBREDDITS = [
    "sysadmin",
    "linux",
    "selfhosted",
    "homelab",
    "LocalLLaMA",
    "linuxquestions",
    "archlinux",
    "debian",
    "commandline",
]

MAX_POSTS_PER_SUB = int(os.getenv("MAX_POSTS_PER_SUB", "10"))
MAX_COMMENTS_PER_POST = int(os.getenv("MAX_COMMENTS_PER_POST", "5"))
RELEVANCE_THRESHOLD = int(os.getenv("RELEVANCE_THRESHOLD", "5"))
CONTEXT_POSTS = int(os.getenv("CONTEXT_POSTS", "8"))

# Embeddings
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Persona (configurable per-topic or globally)
DEFAULT_PERSONA = os.getenv(
    "DEFAULT_PERSONA",
    "You are a research assistant. Be concise, technical, and cite sources.",
)

# Query expansion
MAX_EXPANDED_QUERIES = int(os.getenv("MAX_EXPANDED_QUERIES", "3"))

# Iterative research
MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))


def validate(log: Callable[[str], None] = print) -> dict[str, bool]:
    """
    Emit a readable config summary and return a status dict.

    Does not raise — missing optional credentials are fine. The caller can
    decide what to do with the returned dict.
    """
    status = {
        "reddit_api": bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET),
        "brave_api": bool(BRAVE_API_KEY),
        "tavily_api": bool(TAVILY_API_KEY),
        "serper_api": bool(SERPER_API_KEY),
        "exa_api": bool(EXA_API_KEY),
        "db_writable": False,
    }

    try:
        db_file = Path(DB_PATH)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        probe = db_file.parent / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
        status["db_writable"] = True
    except OSError as e:
        log(f"DB path not writable: {DB_PATH} ({e})")

    log(f"reddit-research {__version__}")
    log(f"  DB path         : {DB_PATH} ({'writable' if status['db_writable'] else 'READ-ONLY'})")
    log(f"  Ollama          : {OLLAMA_BASE_URL} (model: {OLLAMA_MODEL})")
    log(f"  Reddit API      : {'PRAW (authenticated)' if status['reddit_api'] else 'public JSON (rate-limited)'}")
    log(f"  Brave Search    : {'configured' if status['brave_api'] else 'not configured'}")
    log(f"  Tavily Search   : {'configured' if status['tavily_api'] else 'not configured'}")
    log(f"  Serper (Google) : {'configured' if status['serper_api'] else 'not configured'}")
    log(f"  Exa.ai Search   : {'configured' if status['exa_api'] else 'not configured'}")

    return status
