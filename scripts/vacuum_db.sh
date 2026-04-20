#!/usr/bin/env bash
# Reclaim space and defragment. Safe to run while the app is not writing.
set -euo pipefail

DB="${DB_PATH:-${HOME}/.local/share/reddit-research/research.db}"

if [[ ! -f "$DB" ]]; then
  echo "No DB found at $DB" >&2
  exit 1
fi

BEFORE=$(stat -c%s "$DB")
sqlite3 "$DB" "PRAGMA wal_checkpoint(TRUNCATE); VACUUM;"
AFTER=$(stat -c%s "$DB")
echo "VACUUM: $DB  $BEFORE → $AFTER bytes"
