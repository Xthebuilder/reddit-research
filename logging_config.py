"""
Centralized logging setup.

Call `configure_logging()` once at process start. Everywhere else:

    from logging_config import get_logger
    log = get_logger(__name__)
    log.info("message")
    log.exception("something blew up")  # inside except:
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def _default_log_dir() -> Path:
    override = os.getenv("LOG_DIR")
    if override:
        return Path(override)
    xdg = os.getenv("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(xdg) / "reddit-research"


def configure_logging(level: str | None = None, to_file: bool = True) -> None:
    """Idempotent logging setup. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    root.setLevel(lvl)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(stream)

    if to_file:
        log_dir = _default_log_dir()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_dir / "app.log",
                maxBytes=2_000_000,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
            root.addHandler(fh)
        except OSError:
            # Read-only FS or permissions — fall back to stderr only.
            pass

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
