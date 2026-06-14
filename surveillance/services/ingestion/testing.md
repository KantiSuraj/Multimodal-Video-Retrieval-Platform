# Local testing guide — ingestion pipeline


Everything runs as native processes on your laptop.
Total setup time: ~15 minutes the first time.

---

## The three things you need open while working

| Terminal | What runs there |
|----------|-----------------|
| Tab 1    | Infrastructure (postgres, minio, rabbitmq, redis) |
| Tab 2    | The ingestion service (`uvicorn`) |
| Tab 3    | Your work — curl tests, pytest, code edits |

---

## STEP 1 — Install system dependencies (once per machine)

### macOS
```bash
brew install postgresql@16 redis ffmpeg libmagic
brew install rabbitmq

# MinIO is a single binary — no brew formula needed
curl -L https://dl.min.io/server/minio/release/darwin-arm64/minio -o /usr/local/bin/minio
# Intel Mac: use darwin-amd64 instead of darwin-arm64
chmod +x /usr/local/bin/minio
```

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    postgresql-16 postgresql-client-16 \
    redis-server \
    rabbitmq-server \
    ffmpeg \
    libmagic1

# MinIO binary
curl -L https://dl.min.io/server/minio/release/linux-amd64/minio -o ~/minio
chmod +x ~/minio
sudo mv ~/minio /usr/local/bin/minio
```

### Windows
Use WSL2 with Ubuntu and follow the Ubuntu instructions above.

---

## STEP 2 — Create the Python environment (once per machine)

Run this from the **`surveillance/` root folder**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows WSL: same command

pip install uv

# Install shared library + ingestion service (editable = code changes apply immediately)
uv pip install -e ./shared -e ./services/ingestion

# All runtime dependencies
uv pip install \
    fastapi "uvicorn[standard]" python-multipart \
    aiofiles aiohttp python-magic watchdog \
    "aio-pika" minio "sqlalchemy[asyncio]" asyncpg alembic \
    structlog tenacity pydantic pydantic-settings

# Test dependencies
uv pip install pytest pytest-asyncio httpx
```

Verify it worked:
```bash
python3 -c "
from shared.models import VideoRecord
from shared.events import VideoIngestedEvent
from services.ingestion.core.config import get_settings
print('OK —', get_settings().APP_NAME)
"
```
Expected: `OK — VideoIngestionService`

---

## STEP 3 — Create a .env file (once per machine)

In `surveillance/` create `.env`:

```bash
cat > .env << 'EOF'
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/surveillance
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
AMQP_URL=amqp://guest:guest@localhost/
REDIS_URL=redis://localhost:6379/0
MINIO_RAW_BUCKET=raw-videos
MINIO_QUARANTINE_BUCKET=quarantine-videos
WATCH_DIRECTORY=/tmp/video_watch
DEBUG=true
EOF
```

---

## STEP 4 — Start infrastructure (every session, Tab 1)

### macOS — start all four with one block:
```bash
# PostgreSQL
brew services start postgresql@16
createdb -U postgres surveillance 2>/dev/null || true

# Redis
brew services start redis

# RabbitMQ
brew services start rabbitmq

# MinIO (runs in foreground — leave this terminal open)
mkdir -p ~/minio-data
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
    minio server ~/minio-data --console-address ":9001"
```

### Ubuntu — start all four with one block:
```bash
sudo systemctl start postgresql redis-server rabbitmq-server
sudo -u postgres createdb surveillance 2>/dev/null || true

# MinIO (runs in foreground — leave this terminal open)
mkdir -p ~/minio-data
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
    minio server ~/minio-data --console-address ":9001"
```

Quick check — all four should respond:
```bash
pg_isready -U postgres           # should print: accepting connections
redis-cli ping                   # should print: PONG
curl -s localhost:9000/minio/health/live && echo " MinIO OK"
curl -s localhost:15672 | grep -q RabbitMQ && echo "RabbitMQ OK"
```

---

## STEP 5 — Run database migrations (once, or after model changes)

```bash
cd surveillance/
export PYTHONPATH=$PWD
source .venv/bin/activate

alembic -c infra/migrations/alembic.ini upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001_initial, Initial schema
```

Verify tables exist:
```bash
psql -U postgres -d surveillance -c "\dt"
```
You should see: `detection_results`, `embedding_records`, `video_records`

---

## STEP 6 — Start the ingestion service (Tab 2)

```bash
cd surveillance/
source .venv/bin/activate
export PYTHONPATH=$PWD

uvicorn services.ingestion.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-config null
```

Expected startup log:
```
startup_begin   service=VideoIngestionService
storage_ready   raw=raw-videos quarantine=quarantine-videos
amqp_publisher_ready
filesystem_watcher_started  directory=/tmp/video_watch
startup_complete
```

- API docs: http://localhost:8000/docs
- Health:   http://localhost:8000/health
- MinIO UI: http://localhost:9001  (minioadmin / minioadmin)

---

## STEP 7 — Run unit tests (Tab 3, no infra needed)

```bash
cd surveillance/
source .venv/bin/activate
export PYTHONPATH=$PWD

pytest services/ingestion/tests/ -v
```

Expected — 11 passed, 0 warnings:
```
PASSED  TestVideoValidator::test_invalid_extension_rejected
PASSED  TestVideoValidator::test_sha256_computed
PASSED  TestVideoValidator::test_corrupt_file_fails
PASSED  TestVideoValidator::test_ffprobe_metadata_extracted
PASSED  TestIngestionService::test_duplicate_returns_existing
PASSED  TestIngestionService::test_invalid_file_quarantined_and_raises
PASSED  TestIngestionService::test_minio_failure_raises_503
PASSED  TestIngestionService::test_db_failure_cleans_up_storage
PASSED  TestIngestionService::test_successful_ingest_publishes_event
PASSED  TestIngestionStorageService::test_upload_returns_object_path
PASSED  TestIngestionStorageService::test_quarantine_uses_quarantine_bucket
```

These tests are fully mocked — PostgreSQL / MinIO / RabbitMQ do NOT need to be running.

---

## STEP 8 — Manual API tests (service must be running)

### Get a test video
```bash
# Generate a 5-second test clip with FFmpeg
ffmpeg -f lavfi -i testsrc=duration=5:size=1280x720:rate=25 \
       -f lavfi -i sine=frequency=440:duration=5 \
       -c:v libx264 -c:a aac \
       /tmp/test_video.mp4
```

### Test 1 — Upload a video (FR-ING-01)
```bash
curl -X POST http://localhost:8000/api/v1/videos \
  -F "file=@/tmp/test_video.mp4" \
  -F 'metadata={"camera_id":"CAM-01","location":"Entrance"}'
```
Expected: **202** with `video_id` and `polling_url`

Save the video_id:
```bash
VIDEO_ID=$(curl -s -X POST http://localhost:8000/api/v1/videos \
  -F "file=@/tmp/test_video.mp4" \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('video_id',''))")
echo "video_id = $VIDEO_ID"
```

### Test 2 — Poll status (FR-ING-04)
```bash
curl http://localhost:8000/api/v1/videos/$VIDEO_ID/status | python3 -m json.tool
```
Check: `camera_id`, `duration_seconds`, `resolution_width` all populated from FFprobe.

### Test 3 — Deduplication (FR-ING-06)
Upload the same file again:
```bash
curl -X POST http://localhost:8000/api/v1/videos \
  -F "file=@/tmp/test_video.mp4"
```
Expected: **200** (not 202) with `"message": "Duplicate: video already exists"`

### Test 4 — Corrupted file / quarantine (FR-ING-05)
```bash
echo "this is not a video" > /tmp/fake.mp4
curl -X POST http://localhost:8000/api/v1/videos \
  -F "file=@/tmp/fake.mp4"
```
Expected: **422** — file moved to quarantine bucket in MinIO.

Check quarantine in MinIO UI: http://localhost:9001 → bucket `quarantine-videos`

### Test 5 — Unsupported extension (FR-ING-02)
```bash
curl -X POST http://localhost:8000/api/v1/videos \
  -F "file=@/tmp/fake.mp4;filename=malware.exe"
```
Expected: **422** `Unsupported file extension: '.exe'`

### Test 6 — Filesystem ingest (FR-ING-01)
```bash
curl -X POST http://localhost:8000/api/v1/videos/fs \
  -H "Content-Type: application/json" \
  -d "{\"file_path\":\"/tmp/test_video.mp4\",\"camera_id\":\"CAM-NAS\"}"
```

### Test 7 — Filesystem watcher (FR-ING-01)
Drop a file into the watched directory — the service picks it up automatically:
```bash
cp /tmp/test_video.mp4 /tmp/video_watch/cam_feed_$(date +%s).mp4
# Watch the service terminal — you'll see: file_detected → ingestion_accepted
```

---

## STEP 9 — Verify in the database

```bash
psql -U postgres -d surveillance << 'SQL'
SELECT
    id,
    original_filename,
    status,
    camera_id,
    duration_seconds,
    resolution_width,
    resolution_height,
    file_size_bytes,
    created_at
FROM video_records
ORDER BY created_at DESC
LIMIT 10;
SQL
```

Check quarantined files:
```bash
psql -U postgres -d surveillance -c "
SELECT id, original_filename, status, error_message
FROM video_records
WHERE status = 'QUARANTINED';"
```

---

## Daily workflow (after first setup)

```bash
# Tab 1 — infra (macOS)
brew services start postgresql@16 redis rabbitmq
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
    minio server ~/minio-data --console-address ":9001"

# Tab 2 — service
cd surveillance && source .venv/bin/activate && export PYTHONPATH=$PWD
uvicorn services.ingestion.main:app --reload --log-config null

# Tab 3 — work
cd surveillance && source .venv/bin/activate && export PYTHONPATH=$PWD
pytest services/ingestion/tests/ -v   # run after every change
```

---

## Common errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'shared'` | `export PYTHONPATH=$PWD` from `surveillance/` |
| `asyncpg: could not connect` | PostgreSQL not running — `brew services start postgresql@16` |
| `connection refused port 9000` | MinIO not running — start it in Tab 1 |
| `connection refused port 5672` | RabbitMQ not running — `brew services start rabbitmq` |
| `UndefinedTableError: video_records` | Migration not run — `alembic -c infra/migrations/alembic.ini upgrade head` |
| `FileNotFoundError: libmagic` | `brew install libmagic` or `apt-get install libmagic1` |
| `422 on valid video` | Reinstall: `uv pip install python-magic` (not `python-magic-bin`) |