#!/usr/bin/env python3
"""
Entry point. Loads .env if present, then launches the TUI.
"""
from __future__ import annotations

import sys

from env_loader import load_env
from logging_config import configure_logging, get_logger

load_env()
configure_logging()
log = get_logger(__name__)


def main() -> None:
    from _version import __version__
    from config import validate
    from tui import main as tui_main

    if "--version" in sys.argv:
        print(f"reddit-research {__version__}")
        return

    validate(log=log.info)
    tui_main()


if __name__ == "__main__":
    main()
