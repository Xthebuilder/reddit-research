from __future__ import annotations

from pathlib import Path

from reddit_research.utils.env_loader import _parse_env_file, load_env


def test_parse_env_file_basic(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text('FOO=bar\n# comment\nBAZ="quoted"\nEMPTY=\n')
    values = _parse_env_file(p)
    assert values == {"FOO": "bar", "BAZ": "quoted", "EMPTY": ""}


def test_parse_env_file_missing(tmp_path: Path):
    assert _parse_env_file(tmp_path / "nope.env") == {}


def test_load_env_does_not_override(monkeypatch, tmp_path: Path):
    shared = tmp_path / "shared.env"
    shared.write_text("FOO=from-file\n")
    monkeypatch.setenv("FOO", "already-set")
    monkeypatch.setenv("JRVS_ENV_PATH", str(shared))

    load_env()
    import os
    assert os.environ["FOO"] == "already-set"
