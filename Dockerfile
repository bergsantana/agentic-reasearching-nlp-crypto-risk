# App image for the collectors and the Prefect scheduler. It holds only the locked
# dependencies: the code is bind-mounted at /app (docker-compose.yml), so run IDs read the
# real git commit and a dirty working tree shows up as "-dirty".
FROM ghcr.io/astral-sh/uv:0.11.7-python3.12-trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PREFECT_HOME=/tmp/prefect

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

ENTRYPOINT ["python", "-m", "cryptonlp"]
