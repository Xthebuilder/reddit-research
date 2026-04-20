"""Shared pytest fixtures. Isolates each test from the user's real DB."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))

    # Reset cached modules so DB_PATH is re-read.
    for mod in ("config", "db"):
        sys.modules.pop(mod, None)

    import db as db_mod

    db_mod.init_db()
    yield db_mod

    for mod in ("config", "db"):
        sys.modules.pop(mod, None)
