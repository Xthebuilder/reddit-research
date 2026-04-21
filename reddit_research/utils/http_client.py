"""
Shared httpx client with connection pooling + retry-aware transport.

Usage:
    from reddit_research.utils.http_client import get_client
    r = get_client().get(url, params=params, timeout=15)

Call `close_client()` at shutdown (registered via atexit automatically).
"""

from __future__ import annotations

import atexit
import threading

import httpx

_client: httpx.Client | None = None
_lock = threading.Lock()

DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0, read=15.0, write=10.0)
LONG_TIMEOUT = httpx.Timeout(120.0, connect=5.0, read=120.0, write=30.0)


def _build_client(timeout: httpx.Timeout = DEFAULT_TIMEOUT) -> httpx.Client:
    transport = httpx.HTTPTransport(retries=2)
    return httpx.Client(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _build_client()
                atexit.register(close_client)
    return _client


def close_client() -> None:
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


def stream(method: str, url: str, **kwargs):
    """Wrapper so callers don't construct their own client for streaming."""
    return get_client().stream(method, url, **kwargs)
