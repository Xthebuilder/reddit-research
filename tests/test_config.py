from __future__ import annotations


def test_validate_returns_status(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    import sys
    sys.modules.pop("config", None)
    import config

    logs: list[str] = []
    status = config.validate(log=logs.append)
    assert status["db_writable"] is True
    assert status["brave_api"] is False
    assert status["reddit_api"] is False
    assert any("reddit-research" in line for line in logs)


def test_version_exposed():
    from _version import __version__
    assert __version__.count(".") == 2
