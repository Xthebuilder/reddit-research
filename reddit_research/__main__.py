#!/usr/bin/env python3
"""Entry point for `python -m reddit_research` and the `reddit-research` CLI."""
from __future__ import annotations

import sys

from reddit_research.utils.env_loader import load_env
from reddit_research.utils.logging_config import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger(__name__)


def main() -> None:
    from reddit_research._version import __version__
    from reddit_research.config import validate
    from reddit_research.ui.app import main as tui_main

    if "--version" in sys.argv:
        print(f"reddit-research {__version__}")
        return

    validate(log=log.info)
    tui_main()


if __name__ == "__main__":
    main()
