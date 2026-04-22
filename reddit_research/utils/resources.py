"""
Runtime resource detection — CPU, RAM, and GPU VRAM.

Computes a safe parallel worker count that leaves enough headroom for other
processes. Called at the start of each processing batch so it adapts if the
system gets busier mid-run.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache

from reddit_research.utils.logging_config import get_logger

log = get_logger(__name__)

# Headroom guardrails — aggressive defaults, override with RESEARCH_MODE=conservative
_AGGRESSIVE = os.getenv("RESEARCH_MODE", "aggressive").lower() != "conservative"

if _AGGRESSIVE:
    _MIN_FREE_RAM_MB  = 512    # only reserve 512 MB RAM for the rest of the system
    _MIN_FREE_VRAM_MB = 256    # only reserve 256 MB VRAM (LLMs already loaded, just need context room)
    _MAX_CPU_FRACTION = 0.95   # use up to 95% of CPU cores
else:
    _MIN_FREE_RAM_MB  = 2048
    _MIN_FREE_VRAM_MB = 1536
    _MAX_CPU_FRACTION = 0.70


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _free_ram_mb() -> int:
    try:
        import psutil
        return psutil.virtual_memory().available // (1024 * 1024)
    except Exception:
        return 4096  # safe fallback: assume 4 GB


def _cpu_cores_available() -> int:
    try:
        import psutil
        usage_pct = psutil.cpu_percent(interval=0.3)
        cores = psutil.cpu_count(logical=True) or 4
        free_fraction = max(0.0, 1.0 - usage_pct / 100.0)
        return max(1, int(cores * free_fraction * _MAX_CPU_FRACTION))
    except Exception:
        return max(1, (os.cpu_count() or 4) // 2)


def _free_vram_mb() -> int | None:
    """Return free VRAM in MB for the best GPU, or None if no GPU found."""
    # NVIDIA
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3, text=True,
        )
        values = []
        for line in out.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                values.append(int(line))
        if values:
            return max(values)
    except Exception:
        pass

    # AMD ROCm
    try:
        import json
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL, timeout=3, text=True,
        )
        data = json.loads(out)
        for card in data.values():
            total = int(card.get("VRAM Total Memory (B)", 0))
            used  = int(card.get("VRAM Total Used Memory (B)", 0))
            if total > 0:
                return (total - used) // (1024 * 1024)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def safe_worker_count(requested: int | None = None) -> tuple[int, str]:
    """
    Return (workers, summary_line).

    Starts from `requested` (or OLLAMA_NUM_PARALLEL) and reduces it if
    CPU load, free RAM, or free VRAM are tight. Always returns at least 1.
    """
    from reddit_research.config import OLLAMA_NUM_PARALLEL
    cap = max(1, requested or OLLAMA_NUM_PARALLEL)

    reasons: list[str] = []

    # --- CPU ---
    cpu_safe = _cpu_cores_available()
    if cpu_safe < cap:
        reasons.append(f"CPU: {cpu_safe} cores headroom")
        cap = cpu_safe

    # --- RAM ---
    free_ram = _free_ram_mb()
    if free_ram < _MIN_FREE_RAM_MB:
        reasons.append(f"RAM critical: {free_ram} MB free → 1 worker")
        cap = 1
    elif free_ram < _MIN_FREE_RAM_MB * 2:
        reasons.append(f"RAM tight: {free_ram} MB free → 2 workers max")
        cap = min(cap, 2)

    # --- VRAM ---
    free_vram = _free_vram_mb()
    vram_note = ""
    if free_vram is not None:
        vram_note = f", VRAM: {free_vram} MB free"
        if free_vram < _MIN_FREE_VRAM_MB:
            reasons.append(f"VRAM critical: {free_vram} MB free → 1 worker")
            cap = 1
        elif free_vram < _MIN_FREE_VRAM_MB * 2:
            reasons.append(f"VRAM tight: {free_vram} MB free → 2 workers max")
            cap = min(cap, 2)

    cap = max(1, cap)

    if reasons:
        summary = f"{cap} workers (reduced: {'; '.join(reasons)})"
    else:
        summary = (
            f"{cap} workers — resources healthy "
            f"(CPU: {cpu_safe} cores free, RAM: {free_ram} MB free{vram_note})"
        )

    return cap, summary


def io_worker_count() -> int:
    """
    Worker count for HTTP-only operations (web search, Reddit fetch).
    These don't compete for VRAM, just sockets. Can be much higher than LLM workers.
    """
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or 4
    except Exception:
        cores = os.cpu_count() or 4
    # 4 sockets per logical core, capped at 64 to avoid overwhelming the upstream API
    return min(64, max(8, cores * 4))


def system_summary() -> str:
    """One-line system resource snapshot for status output."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.3)
        ram_str = f"RAM {ram.available // (1024**2)}/{ram.total // (1024**2)} MB free"
        cpu_str = f"CPU {cpu_pct:.0f}% used ({psutil.cpu_count()} cores)"
    except Exception:
        ram_str = "RAM unknown"
        cpu_str = f"CPU {os.cpu_count() or '?'} cores"

    vram = _free_vram_mb()
    vram_str = f"VRAM {vram} MB free" if vram is not None else "no GPU"

    return f"{cpu_str} | {ram_str} | {vram_str}"
