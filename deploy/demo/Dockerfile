# Public demo: API + live replay + static Next.js frontend in one container.
# The build context is assembled by deploy/demo/build.py (see that file).
FROM python:3.12-slim

# LightGBM needs the GNU OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

# Hugging Face Spaces run the container as uid 1000; harmless elsewhere
RUN useradd -m -u 1000 user
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY sql ./sql
COPY configs ./configs
COPY reports ./reports
RUN uv sync --locked --no-dev
COPY web ./web
COPY demo ./demo
RUN chown -R user:user /app
USER user

# no PostgreSQL in the demo: the model comes from the bundle directory, the
# registry from its snapshot, and prediction logging is off
ENV SENTINEL_DATA_DIR=/app/demo/data \
    SENTINEL_ARTIFACTS_DIR=/app/demo/artifacts \
    SENTINEL_BUNDLE_DIR=/app/demo/artifacts/models/production \
    SENTINEL_FRONTEND_DIR=/app/web \
    SENTINEL_LOG_PREDICTIONS=0 \
    SENTINEL_DB_URL=none

# Railway and most platforms pass the port in $PORT; Hugging Face expects 7860
EXPOSE 7860
CMD ["sh", "-c", "exec /app/.venv/bin/sentinel serve --host 0.0.0.0 --port ${PORT:-7860}"]
