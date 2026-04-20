#!/usr/bin/env bash
# Safe SQLite online backup (uses .backup — WAL-aware, no locking issues).
# Usage: ./scripts/backup_db.sh [destination_dir]
set -euo pipefail

SRC_DB="${DB_PATH:-${HOME}/.local/share/reddit-research/research.db}"
DEST_DIR="${1:-${HOME}/.local/share/reddit-research/backups}"

if [[ ! -f "$SRC_DB" ]]; then
  echo "No DB found at $SRC_DB" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$DEST_DIR/research-${STAMP}.db"

sqlite3 "$SRC_DB" ".backup '$DEST'"
gzip -9 "$DEST"
echo "Backup: $DEST.gz"

# Retain newest 30
ls -1t "$DEST_DIR"/research-*.db.gz 2>/dev/null | tail -n +31 | xargs -r rm -f
