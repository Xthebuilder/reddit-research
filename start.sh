#!/usr/bin/env bash
# Compat wrapper for the old entry point — runs setup once then launches the TUI.
# Prefer ./setup.sh (first time) and ./run.sh (every other time).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$ROOT_DIR/.venv" || ! -f "$ROOT_DIR/.venv/bin/python" ]]; then
  "$ROOT_DIR/setup.sh"
fi
exec "$ROOT_DIR/run.sh" "$@"
