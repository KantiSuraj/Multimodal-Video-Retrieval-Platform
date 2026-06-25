# ── Search Service Dockerfile ─────────────────────────────────────────────────
#
# Build context must be surveillance/ (the monorepo root), not this directory.
# The docker-compose context: ../../ from infra/docker/ achieves this.
#
# Build:
#   docker build -f services/search/Dockerfile -t search-service .
#
# The search service:
#   - Exposes REST API: POST /api/v1/search/text, POST /api/v1/search/image
#   - Encodes queries with CLIP (CPU inference in dev, GPU in prod)
#   - Queries Qdrant for ANN retrieval
#   - Reads PostgreSQL for metadata hydration
#   - Reads/writes Redis for embedding + result caching
#   - Is request-driven (HTTP), NOT event-driven (no RabbitMQ consumer)

FROM python:3.12-slim

# System dependencies:
#   - libgl1 / libglib2.0-0: required by Pillow for image decoding
#   - curl: health-check probe
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install shared library first (changes rarely) ─────────────────────────────
COPY shared/ ./shared/
RUN pip install --no-cache-dir -e ./shared/

# ── Install service dependencies ──────────────────────────────────────────────
# transformers + torch (CPU) for CLIP query encoding.
# Use --extra-index-url for the CPU-only PyTorch wheel to avoid
# downloading the full CUDA build (~2 GB) in the container.
COPY services/search/pyproject.toml ./services/search/pyproject.toml
RUN pip install --no-cache-dir \
        fastapi>=0.111 \
        uvicorn[standard]>=0.29 \
        qdrant-client>=1.9 \
        sqlalchemy[asyncio]>=2.0 \
        asyncpg>=0.29 \
        redis>=5.0 \
        pillow>=10.0 \
        numpy>=1.26 \
        pydantic>=2.7 \
        pydantic-settings>=2.2 \
        structlog>=24.1 \
        python-multipart>=0.0.9 \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch>=2.2 \
        transformers>=4.40

# ── Copy service source ───────────────────────────────────────────────────────
COPY services/search/ ./services/search/

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# HuggingFace model cache — CLIP weights are downloaded on first startup
# and cached here. Mount a named volume in production to avoid
# re-downloading on every container restart.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "services.search.main:app", "--host", "0.0.0.0", "--port", "8000"]
