#!/usr/bin/env bash
# Launch the TUI. Assumes setup.sh has already been run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "No .venv found. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
exec python "$ROOT_DIR/research.py" "$@"
