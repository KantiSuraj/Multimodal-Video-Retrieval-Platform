# ── Indexing Service Dockerfile ───────────────────────────────────────────────
#
# Build context must be surveillance/ (the monorepo root), not this directory.
# The docker-compose context: ../../ from infra/docker/ achieves this.
#
# Build:
#   docker build -f services/indexing/Dockerfile -t indexing-service .
#
# The indexing service:
#   - Consumes EmbeddingsReadyEvent from RabbitMQ
#   - Upserts vectors into Qdrant
#   - Updates EmbeddingRecord rows in PostgreSQL
#   - Has NO GPU dependency (Qdrant I/O is CPU-bound)

FROM python:3.12-slim

# System dependencies: no ffmpeg, no torch — indexing is pure I/O
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install shared library first (changes rarely) ─────────────────────────────
COPY shared/ ./shared/
RUN pip install --no-cache-dir -e ./shared/

# ── Install service dependencies ──────────────────────────────────────────────
COPY services/indexing/pyproject.toml ./services/indexing/pyproject.toml
RUN pip install --no-cache-dir \
        fastapi>=0.111 \
        uvicorn[standard]>=0.29 \
        qdrant-client>=1.9 \
        sqlalchemy[asyncio]>=2.0 \
        asyncpg>=0.29 \
        aio-pika>=9.4 \
        pydantic>=2.7 \
        pydantic-settings>=2.2 \
        structlog>=24.1

# ── Copy service source ───────────────────────────────────────────────────────
COPY services/indexing/ ./services/indexing/

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Indexing listens on 8000 for health checks; its primary work is message
# consumption from RabbitMQ, not serving HTTP.
CMD ["uvicorn", "services.indexing.main:app", "--host", "0.0.0.0", "--port", "8000"]
