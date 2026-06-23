# Embedding Service — Testing Guide

Two parts: **Part 1** is what you do right now (no Docker, no GPU, no real
Postgres/RabbitMQ/MinIO). **Part 2** is the Docker-based path for later,
included so you have it when you're ready.

---

## Part 1 — Pure local testing (no Docker)

The unit test suite never touches a real database, queue, object store, or
GPU — every test mocks `_already_processed`, `_mark_status`,
`fetch_artifact`, etc. with `AsyncMock`/`MagicMock`. That means **Tier 1
below needs nothing installed beyond pure-Python packages** — not even
`torch`.

### Tier 1 — Run the unit test suite (start here)

This is the fastest signal and what you should run after every change.

**1. Create a virtual environment at the repo root:**

```bash
cd /path/to/repo-root        # the directory that directly contains shared/ and services/
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

**2. Install exactly what the test suite needs to *import* (not run a model):**

```bash
pip install \
  "pydantic>=2.6" \
  "pydantic-settings>=2.2" \
  "sqlalchemy[asyncio]>=2.0" \
  "asyncpg>=0.29" \
  "aio-pika>=9.4" \
  "minio>=7.2" \
  "tenacity>=8.2" \
  "structlog>=24.1" \
  "pillow>=10.2" \
  "numpy>=1.26" \
  "pytest>=8.0" \
  "pytest-asyncio>=0.23"
```

Notice **`torch` and `transformers` are not in this list**. `clip_model.py`
only imports them *inside* `load()`/`_run_sync()`, never at module level —
the same lazy-import pattern Detection uses for `grounding_dino.py` —
specifically so the test suite never needs them. `asyncpg`/`aio-pika`/`minio`
*are* needed even though no real connection happens, because creating the
SQLAlchemy engine and importing the queue/storage modules resolves those
drivers at import time (no socket is opened — `create_async_engine` is lazy).

**3. Set `PYTHONPATH` to the repo root** (so `shared.X` and `services.embedding.X`
resolve) and run pytest:

```bash
# from the repo root, with the venv active
export PYTHONPATH=$(pwd)          # Windows (PowerShell): $env:PYTHONPATH = (Get-Location).Path
pytest services/embedding/tests -v
```

You should see all tests pass — including the `TestCLIPEmbedderNormalization`
tests, which exercise `CLIPEmbedder._normalize` directly without loading any
model. No `.env` file is required for this tier; `Settings()` falls back to
its defaults (e.g. `DATABASE_URL` defaults to a localhost Postgres URL that's
never actually connected to).

> If `pytest` can't find `services` or `shared`, double check `PYTHONPATH`
> is the **parent** of both directories, not `services/embedding` itself.

### Tier 2 — Smoke-test the real CLIP model on CPU (no infra, no Docker)

This actually loads a model and generates a real embedding — useful to
confirm `CLIP_DEVICE=cpu` works end-to-end before you wire up the full
pipeline. It still doesn't need Postgres/RabbitMQ/MinIO since it talks to
`CLIPEmbedder` directly.

**1. Install the model-runtime deps on top of Tier 1:**

```bash
pip install "torch>=2.2" "transformers>=4.40"
```

This installs the standard PyPI `torch` wheel, which runs on CPU out of the
box — no CUDA toolkit or GPU driver needed locally.

**2. (Recommended) Use a smaller checkpoint for faster local iteration.**
Add a `.env` file at the repo root (or wherever your process reads it from):

```dotenv
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CLIP_DEVICE=cpu
```

`clip-vit-base-patch32` is much faster on CPU than the production
`clip-vit-large-patch14`. Swap back to the large checkpoint when you actually
care about embedding quality or are testing against production-equivalent
output.

**3. Run a tiny ad-hoc smoke script** (not part of the test suite — just a
scratch file, e.g. `scratch_smoke_test.py` at the repo root):

```python
import asyncio
from PIL import Image

from services.embedding.core.config import Settings
from services.embedding.services.clip_model import CLIPEmbedder


async def main():
    settings = Settings(CLIP_MODEL_NAME="openai/clip-vit-base-patch32", CLIP_DEVICE="cpu")
    embedder = CLIPEmbedder(settings)
    embedder.load()  # downloads the model from Hugging Face on first run

    # any local image works — swap in a real path
    Image.new("RGB", (224, 224), color="red").save("/tmp/sample.jpg")
    vector = await embedder.embed_image("/tmp/sample.jpg")

    print(f"vector length: {len(vector)}")
    print(f"L2 norm (should be ~1.0): {sum(v * v for v in vector) ** 0.5:.6f}")


asyncio.run(main())
```

```bash
python scratch_smoke_test.py
```

First run will download the model from Hugging Face (a few hundred MB for
the base checkpoint) and cache it under `~/.cache/huggingface/`. Subsequent
runs are fast. If you're on a slow connection, set `HF_HOME` to a folder you
control so you know where the cache lives:

```bash
export HF_HOME=$(pwd)/.hf-cache
```

### Tier 3 — Full service boot locally without Docker (optional, heavier)

Only do this if you specifically want to exercise the FastAPI app + RabbitMQ
consumer end-to-end without containers. It means installing Postgres,
RabbitMQ, and MinIO **natively** on your machine:

- macOS: `brew install postgresql rabbitmq minio/stable/minio`
- Linux: `apt install postgresql rabbitmq-server` + download the MinIO
  binary directly from min.io

Then:

```dotenv
# .env at repo root
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/surveillance
AMQP_URL=amqp://guest:guest@localhost/
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CLIP_DEVICE=cpu
```

```bash
uvicorn services.embedding.main:app --reload --port 8001
```

This is genuinely more setup than it's worth for day-to-day iteration —
**Tier 1 + Tier 2 cover almost everything you need locally.** Tier 3 is here
for completeness; most people skip straight to Docker (Part 2) once they
need real infra, rather than installing it natively.

---

## Part 2 — Docker-based testing (for later)

### 1. Build the image

```bash
# from the repo root
docker build -f services/embedding/Dockerfile -t embedding-service .
```

### 2. Bring up infra + the service with Docker Compose

You likely already have a `docker-compose.yml` for Detection/Ingestion —
add an `embedding` service block alongside it:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: surveillance
    ports: ["5432:5432"]

  rabbitmq:
    image: rabbitmq:3-management
    ports: ["5672:5672", "15672:15672"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]

  embedding:
    build:
      context: .
      dockerfile: services/embedding/Dockerfile
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@postgres:5432/surveillance
      AMQP_URL: amqp://guest:guest@rabbitmq/
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      MINIO_SECURE: "false"
      CLIP_MODEL_NAME: openai/clip-vit-large-patch14
      CLIP_DEVICE: cpu     # see GPU note below
    depends_on: [postgres, rabbitmq, minio]
    ports: ["8002:8000"]
```

```bash
docker compose up --build
```

### 3. Switching to GPU in Docker (cloud phase)

When you move to a cloud GPU box:

```yaml
  embedding:
    environment:
      CLIP_DEVICE: cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

(Or `docker run --gpus all ...` if you're not using Compose.) This requires
the host to have NVIDIA drivers + the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed — that's a host-level setup, not something the Dockerfile itself
needs to change for. The application code already handles the `cpu`/`cuda`
switch dynamically via `CLIP_DEVICE`, so no code or Dockerfile change is
needed between the two environments — only the env var and, in the cloud
case, the host GPU runtime.

### 4. Running the test suite inside the container (optional)

```bash
docker run --rm -e PYTHONPATH=/app embedding-service \
  sh -c "pip install pytest pytest-asyncio && pytest services/embedding/tests -v"
```

In practice, running tests in Tier 1 locally (Part 1) is faster for
day-to-day work — use this mainly to confirm the container image itself is
healthy before deploying.

---

## Quick reference

| What you want to do | Tier | Needs Docker? | Needs torch? | Needs real infra? |
|---|---|---|---|---|
| Run unit tests | 1 | No | No | No |
| Confirm CLIP works on CPU | 2 | No | Yes | No |
| Full FastAPI + consumer boot | 3 | No | Yes | Yes (native installs) |
| Production-equivalent run | Part 2 | Yes | Yes | Yes (containers) |

Start with Tier 1, move to Tier 2 once you're touching `clip_model.py`
itself, and only reach for Part 2 when you're ready to test the whole
pipeline together.