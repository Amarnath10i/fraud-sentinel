FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# dependencies first so code changes do not invalidate the layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY sql ./sql
COPY src ./src
RUN uv sync --locked --no-dev

# data/ and artifacts/ (processed parquet, model bundles, state snapshots) are mounted at runtime
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "sentinel", "serve", "--host", "0.0.0.0", "--port", "8000"]
