# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_PATH=/data/research.db \
    REPORTS_DIR=/data/reports

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data/reports \
 && chown -R app:app /data

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app . /app

USER app
VOLUME ["/data"]

# Headless is the production entrypoint; TUI needs a real terminal.
ENTRYPOINT ["python", "/app/headless.py"]
CMD ["--help"]
