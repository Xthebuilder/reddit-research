#!/usr/bin/env python3
"""Root shim — delegates to reddit_research.headless for subprocess callers."""
from reddit_research.utils.env_loader import load_env
from reddit_research.utils.logging_config import configure_logging

load_env()
configure_logging()

from reddit_research.headless import main  # noqa: E402

if __name__ == "__main__":
    main()
