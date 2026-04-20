"""
Centralized .env loading.

Order of precedence (later does NOT override earlier — os.environ wins too):
    1. Existing process env
    2. ./.env  (project-local)
    3. $JRVS_ENV_PATH (if set) — shared credentials file, optional
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def load_env(extra_paths: list[Path] | None = None) -> list[Path]:
    """Load .env from project dir + any extras. Returns the paths actually loaded."""
    project_env = Path(__file__).parent / ".env"
    paths: list[Path] = [project_env]

    shared = os.getenv("JRVS_ENV_PATH")
    if shared:
        paths.append(Path(shared))

    if extra_paths:
        paths.extend(extra_paths)

    loaded: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        for key, val in _parse_env_file(p).items():
            os.environ.setdefault(key, val)
        loaded.append(p)
    return loaded
