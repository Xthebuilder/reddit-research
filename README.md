# Reddit Research

Terminal research agent: fetches Reddit + web results, scores them with a local
LLM (Ollama), and produces a markdown report per topic.

## Quick start

```bash
./setup.sh          # first time: creates .venv, installs deps, copies .env.example
./run.sh            # every other time: launches the TUI
```

Shortcuts inside the TUI: `Ctrl+1` Topics · `Ctrl+2` Data · `Ctrl+3` Reports ·
`Ctrl+O` open selected report.

## Headless mode

```bash
./run.sh --headless   # not wired here; use headless.py directly:
python headless.py --topic "ZFS snapshots" --subreddits "zfs,linux,sysadmin"
```

## Configuration

Copy `.env.example` to `.env`. Relevant keys:

| key | purpose |
|-----|---------|
| `OLLAMA_BASE_URL` | defaults to `http://localhost:11434` |
| `OLLAMA_MODEL` | chat model (e.g. `gpt-oss:20b`) |
| `OLLAMA_EMBED_MODEL` | embedding model (e.g. `nomic-embed-text`) |
| `BRAVE_API_KEY` | optional — enables web search |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | optional — PRAW auth |
| `DB_PATH` | override DB location (default: `$XDG_DATA_HOME/reddit-research/research.db`) |
| `REPORTS_DIR` | override report output dir |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` (default `INFO`) |
| `JRVS_ENV_PATH` | optional shared `.env` to load after the local one |

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
pytest
ruff check .
mypy .
```

Run only unit tests (skip ones that hit the network/Ollama):

```bash
pytest -m "not integration"
```

## Docker

```bash
docker build -t reddit-research .
docker run --rm -v $PWD/data:/data \
  -e BRAVE_API_KEY=... \
  reddit-research --topic "k8s backups"
```

## Maintenance

- Backup: `./scripts/backup_db.sh` (keeps 30 newest compressed copies)
- VACUUM: `./scripts/vacuum_db.sh` (reclaim space)

## Architecture

```
research.py / headless.py      ← entry points
  ├─ env_loader  → load .env files
  ├─ logging_config  → rotating file log in $XDG_STATE_HOME/reddit-research/
  ├─ config      → typed settings + validate()
  ├─ http_client → shared httpx.Client with retries
  ├─ db (sqlite) → topics, posts, web_results, sessions, messages
  ├─ reddit      → PRAW or public JSON fallback
  ├─ brave       → Brave Search API
  ├─ llm         → Ollama (chat + embed + RAG)
  └─ report      → markdown generator + background watcher
```
