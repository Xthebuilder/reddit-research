"""
llama.cpp inference backend — talks to llama-server's OpenAI-compatible API.

Start the server with:
    llama-server -m /path/to/model.gguf --n-gpu-layers 99 -c 4096 --port 8080

All functions mirror the Ollama interface so llm.py can swap backends transparently.
"""
from __future__ import annotations

import json

import httpx

from reddit_research.config import LLAMA_CPP_BASE_URL, LLAMA_CPP_MODEL
from reddit_research.utils.http_client import LONG_TIMEOUT, get_client
from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)

_CHAT_URL = f"{LLAMA_CPP_BASE_URL}/v1/chat/completions"
_EMBED_URL = f"{LLAMA_CPP_BASE_URL}/v1/embeddings"
_HEALTH_URL = f"{LLAMA_CPP_BASE_URL}/health"


def check() -> tuple[bool, str]:
    try:
        r = get_client().get(_HEALTH_URL, timeout=5)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", "ok")
            model = LLAMA_CPP_MODEL or "loaded"
            return True, f"{status} | model: {model}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        log.warning("llama.cpp unreachable at %s: %s", LLAMA_CPP_BASE_URL, e)
        return False, str(e)


def _payload(messages: list[dict], stream: bool = False, **kwargs) -> dict:
    p: dict = {"messages": messages, "stream": stream, **kwargs}
    if LLAMA_CPP_MODEL:
        p["model"] = LLAMA_CPP_MODEL
    return p


def chat(messages: list[dict], **kwargs) -> str:
    try:
        r = get_client().post(_CHAT_URL, json=_payload(messages, **kwargs), timeout=LONG_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPError:
        log.exception("llama.cpp chat request failed")
        raise


def chat_stream(messages: list[dict], on_token=None) -> str:
    full: list[str] = []
    try:
        with get_client().stream(
            "POST", _CHAT_URL, json=_payload(messages, stream=True), timeout=LONG_TIMEOUT
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if token:
                    full.append(token)
                    if on_token:
                        on_token(token)
    except httpx.HTTPError:
        log.exception("llama.cpp stream failed")
    return "".join(full)


def embed(text: str) -> list[float]:
    payload: dict = {"input": text}
    if LLAMA_CPP_MODEL:
        payload["model"] = LLAMA_CPP_MODEL
    try:
        r = get_client().post(_EMBED_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("data", [])
        if embeddings:
            return embeddings[0].get("embedding", [])
        return []
    except httpx.HTTPError:
        log.exception("llama.cpp embed failed")
        return []
