"""Shared pytest fixtures. Isolates each test from the user's real DB."""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))

    # Purge ALL reddit_research submodules so config.DB_PATH is re-read from env.
    for mod in list(sys.modules):
        if mod.startswith("reddit_research"):
            del sys.modules[mod]

    db_mod = importlib.import_module("reddit_research.db")
    db_mod.init_db()
    yield db_mod

    for mod in list(sys.modules):
        if mod.startswith("reddit_research"):
            del sys.modules[mod]
