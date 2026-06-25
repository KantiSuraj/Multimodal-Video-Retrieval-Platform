# Multimodal Video Retrieval Platform — Complete Architecture Review

**Audience:** Computer science student familiar with programming fundamentals, unfamiliar with backend infrastructure (Redis, RabbitMQ, MinIO, SQLAlchemy, etc.)

---

## Part 1 — Executive Summary

### What problem does this system solve?

Modern surveillance systems produce terabytes of video every day. A facility with 50 cameras at 15 fps fills roughly 250 GB daily in compressed H.264. When investigators need to find a specific person, vehicle, or event, they face two bad options: scrub through footage manually (impossible at scale), or rely on rule-based alerting that only fires on pre-programmed conditions and misses everything else.

This system solves that by turning video into a **searchable index**. After ingestion, every frame is described by a vector of numbers — an "embedding" — that captures its visual meaning. Searching for "person in red jacket near entrance" becomes a nearest-neighbour lookup in that vector space, returning timestamped results in milliseconds.

### What happens from upload to results?

```
User uploads video
     ↓
Ingestion service validates, deduplicates, stores raw file, emits event
     ↓  (RabbitMQ message)
Preprocessing service transcodes, extracts frames/clips
     ↓  (RabbitMQ message)
Detection service runs Grounding DINO, finds objects, bounding boxes
     ↓  (RabbitMQ message)
Embedding service runs CLIP/SigLIP, produces vectors for each frame/crop/clip
     ↓  (RabbitMQ message)
Indexing service upserts vectors into Qdrant
     ↓
User sends text/image query
     ↓
Search service encodes query → ANN search in Qdrant → ranked results
     ↓
Dashboard renders preview thumbnails with deep-linked timestamps
```

### Major services

|Service|Short description|
|---|---|
|Ingestion|Accept raw video, validate, store, emit event|
|Preprocessing|Transcode, segment, extract frames|
|Detection|Find objects in frames (Grounding DINO)|
|Embedding|Encode frames/crops/clips into vectors (CLIP/SigLIP)|
|Indexing|Write vectors into Qdrant vector DB|
|Search|Accept queries, return ranked results|
|Dashboard|Web UI for upload, search, playback|
|API Gateway|Auth, rate-limiting, routing|

### Why not a simple monolith?

A monolith would be faster to start but would fail at this problem for three concrete reasons.

**GPU scheduling.** Embedding generation requires a GPU and takes ~50 ms per frame. If it ran inside the same process as the REST API, every HTTP request would compete for the GPU. By separating embedding into its own service, you can run ten embedding workers on GPU machines and zero on the web servers.

**Independent failure.** If the ML detection model crashes, you do not want the upload API to also crash. With separate services, ingestion keeps accepting uploads; they queue up; detection restarts; nothing is lost.

**Independent scaling.** During peak upload hours you need more ingestion workers. During search-heavy periods you need more search replicas. A monolith forces you to scale everything together even though only one part is stressed.

---

## Part 2 — System Architecture

### API Layer — `services/ingestion/api/routes.py`

**Purpose:** Translate HTTP requests into Python function calls.

**Responsibility:** Parse multipart uploads, validate request shapes with Pydantic, call the ingestion service, and return the correct HTTP status code. It knows nothing about storage, queues, or databases — it only calls the service layer and maps errors to HTTP codes.

**Inputs:** HTTP requests (multipart/form-data, JSON bodies)

**Outputs:** HTTP responses (202 Accepted, 200 Duplicate, 422 Validation error, 503 Storage unavailable)

**Dependencies:** FastAPI router, the ingestion service singleton, the get_db dependency, Pydantic schemas

**Failure points:** If the service layer raises an `IngestionError`, routes.py catches it and returns the embedded `status_code`. Unhandled exceptions bubble to the global handler in `main.py` which returns 500.

**Why it's separate from services:** Routes are framework-specific (FastAPI). By keeping business logic in `services/ingestion.py`, you can test `IngestionService` without a running HTTP server. Routes just wire HTTP to Python.

---

### Services Layer — `services/ingestion/services/`

This folder contains the actual business logic. Four files:

#### `ingestion.py` — The orchestrator

The most important file in the service. `IngestionService._run_pipeline()` is the six-step sequence that every video upload runs through. It coordinates all the other services. If anything fails, it decides what to do:

- Validation fails → quarantine the file, raise 422
- MinIO fails → raise 503 (after retries in storage layer)
- DB fails → delete partial MinIO upload, raise 500
- Duplicate → return 200 immediately, never continue

**Entry points:** `ingest_upload()`, `ingest_rtsp()`, `ingest_filesystem()`, `get_status()`

**Key class:** `IngestionService` — stateless; the singleton is safe because nothing on the instance changes after creation.

**Key exception:** `IngestionError(message, status_code)` — carries the HTTP status code so routes.py doesn't need to know about internal logic.

#### `validator.py` — File validation

`VideoValidator.validate()` runs four checks in sequence:

1. Extension in the allow-list (`.mp4`, `.avi`, etc.)
2. MIME type detection via libmagic (reads the actual file bytes, not the name)
3. SHA-256 hash computation
4. FFprobe: launches `ffprobe` as a subprocess, parses JSON output, extracts duration, resolution, codec

Returns a `ValidationResult` dataclass. `is_valid=False` means the file should be quarantined. `is_valid=True` means all fields are populated and the pipeline can continue.

**Why FFprobe as a subprocess?** FFprobe is the industry-standard tool for inspecting video files. Python has no pure-Python equivalent that handles all formats reliably. Launching it as a subprocess is the standard pattern.

#### `storage.py` — MinIO wrapper for ingestion

`IngestionStorageService` wraps `shared.storage.ObjectStorageClient` and adds ingestion-specific knowledge: which bucket names to use, what object path format to use (`{video_id}/{filename}`), and a `quarantine()` helper.

All retry logic (3× exponential backoff) lives in the shared client, not here.

#### `queue.py` — RabbitMQ publisher for ingestion

`IngestionPublisher` subclasses `shared.queue.BasePublisher` and adds one typed method: `publish_video_ingested(event: VideoIngestedEvent)`. The connection management, channel setup, exchange declaration, and message serialisation all live in the base class.

---

### Shared Layer — `shared/`

The contract library. Every service imports from here. Nothing in `shared/` imports from any service.

#### `shared/config/base.py`

`BaseServiceSettings` is a Pydantic `BaseSettings` class. It reads values from environment variables and the `.env` file. Every infrastructure address (PostgreSQL URL, MinIO endpoint, RabbitMQ URL, Redis URL) is defined here once. Each service's `Settings` class inherits from it and adds only its own specific fields.

This means if you rename the PostgreSQL password, you change one environment variable, not eight config files.

#### `shared/db.py`

Two factory functions: `build_engine()` and `build_session_factory()`. They take settings and return SQLAlchemy objects. The service's `database.py` calls them at module-level, producing module-level singletons (`engine`, `AsyncSessionLocal`, `get_db`).

**Why factories instead of singletons?** Each service needs its own engine with its own connection pool. The shared package can't create a singleton because it doesn't know which service's settings to use.

#### `shared/models/`

ORM table definitions using SQLAlchemy's modern `DeclarativeBase` and `Mapped[]` typed columns. Three models:

- `VideoRecord` — written by ingestion, read by every service
- `DetectionResult` — written by detection, read by embedding/search
- `EmbeddingRecord` — written by embedding, read by indexing/search

All three inherit from the same `Base`, which means a single Alembic project can manage all three tables.

#### `shared/events/`

Pydantic models that define the shape of messages on RabbitMQ. Four events:

- `VideoIngestedEvent` — ingestion → preprocessing
- `FramesExtractedEvent` — preprocessing → detection
- `DetectionCompleteEvent` — detection → embedding
- `EmbeddingsReadyEvent` — embedding → indexing

**Why in shared?** The producer and consumer of each event must agree on the schema. If they defined the schema independently, they would drift. Putting it in shared means both sides import the same class. If you add a field, you change one file, and the type checker immediately tells you every consumer that needs updating.

#### `shared/queue/publisher.py` and `consumer.py`

Base classes for AMQP communication. `BasePublisher` manages one connection, one channel, one exchange. `BaseConsumer` manages consuming from a durable queue bound to the exchange.

Services subclass these and add one typed method per message type. This keeps service-level queue files at ~10 lines instead of ~60.

#### `shared/storage/client.py`

`ObjectStorageClient` wraps the synchronous MinIO SDK in async with `loop.run_in_executor()`. The retry decorator (`@retry` from tenacity) is applied to `put_object()`, handling `S3Error`, `ConnectionError`, and `TimeoutError` with exponential backoff.

Also provides `presigned_get_url()` which generates a time-limited URL that the dashboard can use to show video previews without proxying bytes through the API.

#### `shared/logging.py`

`configure_logging()` sets up structlog with two modes: JSON for production (machine-readable, compatible with ELK/Splunk), colorised console for development (`DEBUG=true`). Every service calls this once in its lifespan startup. `get_logger(name)` returns a bound logger.

---

### Database Layer — `services/ingestion/db/database.py`

```python
engine            = build_engine(settings)       # connection pool
AsyncSessionLocal = build_session_factory(engine) # session factory

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session          # FastAPI dependency injection
        await session.commit() # auto-commit on success
        # rollback on exception
```

This is a FastAPI dependency. When `routes.py` declares `db: AsyncSession = Depends(get_db)`, FastAPI calls `get_db()` before the handler and commits/rolls back after. The route handler never touches `commit()` or `rollback()` directly.

`engine` and `AsyncSessionLocal` are also imported by `fs_watcher.py`, which needs sessions outside of HTTP context.

**What breaks if you delete this file?** FastAPI can't start — all routes import `get_db`. The filesystem watcher can't create DB sessions. The entire service is dead.

---

### Queue Layer — `shared/queue/` + `services/ingestion/services/queue.py`

RabbitMQ uses a **topic exchange** named `video.events`. Topic exchanges route messages by a dot-separated routing key. Any service can bind a queue to a pattern:

- `video.ingested` → preprocessing listens
- `video.frames_extracted` → detection listens
- `video.detection_complete` → embedding listens
- `video.embeddings_ready` → indexing listens
- `video.*` → a monitoring service could listen to all

Messages are **persistent** (`delivery_mode=PERSISTENT`). This means RabbitMQ writes them to disk before acknowledging. If RabbitMQ restarts, messages survive.

Queues are **durable**. If RabbitMQ restarts, the queue itself (and its unprocessed messages) survive.

The consumer uses `message.process()` as a context manager. If `handle_message()` raises an exception, the message is **nack'd** (not acknowledged) and requeued. This implements at-least-once delivery: the same message may be processed twice if a worker crashes mid-processing. All processing steps must be idempotent to handle this safely.

---

### Storage Layer — `shared/storage/client.py` + `services/ingestion/services/storage.py`

MinIO is an S3-compatible object store. Files are organised into **buckets** (like top-level directories). Ingestion uses two:

- `raw-videos` — the uploaded video file at path `{video_id}/{filename}`
- `quarantine-videos` — corrupted or unsupported files

The MinIO SDK is synchronous (blocking I/O). Calling it directly from an async function would block the entire event loop — no other requests could be served while a 500 MB file uploads. The solution is `loop.run_in_executor()`, which runs the blocking call in a thread pool while Python's event loop remains free to handle other requests.

---

### Embedding and Retrieval Layers

These are defined by the shared events and models but their services are not yet implemented (stubs). The contracts are:

- **Embedding layer** receives `DetectionCompleteEvent`, runs CLIP/SigLIP on each frame and crop, normalises vectors to unit L2 norm, publishes `EmbeddingsReadyEvent` with the full list of `EmbeddingRecord` objects
- **Indexing layer** receives `EmbeddingsReadyEvent`, upserts each vector into Qdrant with metadata filters (camera_id, timestamp), writes an `EmbeddingRecord` row to PostgreSQL linking the Qdrant point ID back to the video
- **Search layer** receives a text or image query, encodes it with the same model, queries Qdrant with optional metadata filters, returns ranked results

---

### Testing Layer — `services/ingestion/tests/test_ingestion.py`

Covered in detail in Part 9.

---

## Part 3 — End-to-End Data Flow

**Scenario:** Investigator uploads `cam_east_2024.mp4` via the Swagger UI.

### Step 1 — HTTP request arrives

**File:** `services/ingestion/api/routes.py` **Function:** `upload_video()` **What happens:**

```
POST /api/v1/videos
Content-Type: multipart/form-data
file=<bytes>
metadata={"camera_id":"CAM-EAST","location":"Building A"}
```

FastAPI calls `get_db()` (injects a DB session), then calls `upload_video()`. The function reads the file bytes into memory, parses the JSON metadata string into a `VideoUploadMetadata` object, and calls `ingestion_service.ingest_upload()`.

**Tables touched:** none yet **Queues used:** none yet **Storage used:** none yet

---

### Step 2 — Validation

**File:** `services/ingestion/services/ingestion.py` → `validator.py` **Function:** `IngestionService._run_pipeline()` → `VideoValidator.validate()` **What happens:**

1. Check `.mp4` is in `ALLOWED_EXTENSIONS` ✓
2. Call `magic.from_buffer(data[:8192])` — libmagic reads the first 8 KB and identifies the binary format. Returns `"video/mp4"` ✓
3. Compute SHA-256 hash of the entire file → `"a3f4b..."`
4. Write file to `/tmp/`, run `ffprobe -v quiet -print_format json -show_format -show_streams /tmp/file.mp4`, parse JSON output. Result: `{duration: 120.5, width: 1920, height: 1080, codec: h264}`

Returns `ValidationResult(is_valid=True, sha256_hash="a3f4b...", ...)`

**Tables touched:** none **Queues used:** none **Storage used:** none

---

### Step 3 — Deduplication check

**File:** `services/ingestion/services/ingestion.py` **Function:** `IngestionService._find_by_hash()` **What happens:**

```python
SELECT * FROM video_records WHERE sha256_hash = 'a3f4b...'
```

Result: no row → this is a new video. Continue.

If a row existed, the function would return `DuplicateVideoResponse` with the existing `video_id` and exit the pipeline. No file would be stored again.

**Tables touched:** `video_records` (read) **Queues used:** none **Storage used:** none

---

### Step 4 — Upload to MinIO

**File:** `services/ingestion/services/storage.py` **Function:** `IngestionStorageService.upload_video()` **What happens:**

Generate `video_id = uuid4()` → `"3fa85f64-..."`

Call MinIO SDK: `put_object("raw-videos", "3fa85f64-.../cam_east_2024.mp4", data)`. The MinIO SDK wraps the bytes in an `io.BytesIO` stream and sends them over HTTP to MinIO (running on port 9000).

Returns storage path: `"3fa85f64-.../cam_east_2024.mp4"`

If MinIO is unreachable, tenacity retries 3× with exponential backoff (1s, 2s, 4s). After the third failure, it re-raises and the ingestion pipeline returns 503.

**Tables touched:** none **Queues used:** none **Storage used:** MinIO bucket `raw-videos`

---

### Step 5 — Insert database record

**File:** `services/ingestion/services/ingestion.py` **Function:** `IngestionService._run_pipeline()` step 4 **What happens:**

```python
record = VideoRecord(
    id=video_id,
    sha256_hash="a3f4b...",
    original_filename="cam_east_2024.mp4",
    mime_type="video/mp4",
    file_size_bytes=156789012,
    storage_path="3fa85f64-.../cam_east_2024.mp4",
    storage_bucket="raw-videos",
    camera_id="CAM-EAST",
    location="Building A",
    duration_seconds=120.5,
    resolution_width=1920,
    resolution_height=1080,
    status=VideoStatus.PENDING,
)
db.add(record)
await db.flush()  # sends INSERT to PostgreSQL but does not commit yet
```

`flush()` sends the SQL to PostgreSQL without committing the transaction. The commit happens when `get_db()` exits its context manager after the HTTP handler returns. This ensures that if the event publish fails, the record is still committed (event failure is non-fatal).

If `flush()` raises (e.g., DB is down), the code catches the exception, calls `storage_service.delete_object()` to remove the MinIO object (cleanup), and raises `IngestionError(500)`.

**Tables touched:** `video_records` (INSERT) **Queues used:** none **Storage used:** none (cleanup only on failure)

---

### Step 6 — Publish event

**File:** `services/ingestion/services/queue.py` **Function:** `IngestionPublisher.publish_video_ingested()` **What happens:**

Build `VideoIngestedEvent`:

```json
{
  "event_type": "VideoIngestedEvent",
  "video_id": "3fa85f64-...",
  "storage_path": "3fa85f64-.../cam_east_2024.mp4",
  "storage_bucket": "raw-videos",
  "sha256_hash": "a3f4b...",
  "original_filename": "cam_east_2024.mp4",
  "metadata": {
    "camera_id": "CAM-EAST",
    "duration_seconds": 120.5,
    ...
  }
}
```

Serialise to JSON bytes, publish to RabbitMQ exchange `video.events` with routing key `video.ingested`. Message is marked `PERSISTENT` so it survives a RabbitMQ restart.

If this publish fails, the code logs an error but does **not** raise. Why? The `VideoRecord` is already in PostgreSQL with `status=PENDING`. A replay mechanism (not yet built) can scan for PENDING records and republish events. The record being in the DB is the source of truth; the queue message is a notification.

**Tables touched:** none **Queues used:** RabbitMQ exchange `video.events`, routing key `video.ingested` **Storage used:** none

---

### Step 7 — HTTP response

**File:** `services/ingestion/api/routes.py` **What happens:**

FastAPI receives the returned `VideoIngestResponse`:

```json
{
  "video_id": "3fa85f64-...",
  "status": "PENDING",
  "polling_url": "/api/v1/videos/3fa85f64-.../status",
  "message": "Video accepted for processing"
}
```

Returns HTTP 202 Accepted. The `get_db()` context manager then commits the transaction.

---

### Step 8 — Downstream processing (future services)

The preprocessing service, which subscribes to `video.ingested`, receives the event from RabbitMQ. It fetches the raw file from MinIO, transcodes to H.264 720p via FFmpeg, extracts keyframes, generates 5-second clips, stores them back in MinIO, publishes `FramesExtractedEvent`.

Detection service receives that event, runs Grounding DINO on each keyframe, publishes `DetectionCompleteEvent` with bounding boxes.

Embedding service receives that event, runs CLIP on each frame and crop, publishes `EmbeddingsReadyEvent`.

Indexing service receives that event, upserts vectors into Qdrant, writes `EmbeddingRecord` rows to PostgreSQL, updates `VideoRecord.status = INDEXED`.

---

### Step 9 — Search

User queries `"person in red jacket"`:

1. Search service encodes query text with CLIP's text encoder → 512-dim vector
2. Queries Qdrant: `search(collection="global", vector=v, top=20, filter={camera_id: "CAM-EAST"})`
3. Qdrant performs HNSW approximate nearest-neighbour search
4. Returns top-20 matches with cosine similarity scores
5. Applies temporal deduplication (suppress results within 5s of a higher-ranked hit)
6. For each result, looks up the `EmbeddingRecord` → finds `video_id` + `timestamp_ms`
7. Returns ranked list with `video_id`, `clip_start_ms`, `score`, `preview_frame_url`

---

## Part 4 — Service Communication Map

```
Client
  │
  │ HTTP multipart
  ▼
[Ingestion Service]
  │── PostgreSQL: INSERT video_records
  │── MinIO: PUT raw-videos/{id}/file.mp4
  └── RabbitMQ: PUBLISH video.ingested
            │
            ▼
      [Preprocessing Service]
        │── MinIO: GET raw-videos/{id}/file.mp4
        │── MinIO: PUT processed-clips/...
        └── RabbitMQ: PUBLISH video.frames_extracted
                  │
                  ▼
            [Detection Service]
              │── MinIO: GET frames/...
              │── PostgreSQL: INSERT detection_results
              └── RabbitMQ: PUBLISH video.detection_complete
                        │
                        ▼
                  [Embedding Service]
                    │── MinIO: GET frames/crops/...
                    │── PostgreSQL: INSERT embedding_records
                    └── RabbitMQ: PUBLISH video.embeddings_ready
                              │
                              ▼
                        [Indexing Service]
                          │── Qdrant: UPSERT vectors
                          │── PostgreSQL: UPDATE embedding_records (qdrant_point_id)
                          └── PostgreSQL: UPDATE video_records (status=INDEXED)

[Search Service]
  │── Qdrant: SEARCH vectors
  │── PostgreSQL: SELECT embedding_records, video_records
  └── MinIO: presigned_get_url for preview frames

[Dashboard]
  └── HTTP to all services
```

### Communication choices explained

**HTTP (REST):** Used for client-facing APIs because it's universally compatible, stateless, and easy to test with curl. The dashboard calls the ingestion API via HTTP.

**RabbitMQ (AMQP):** Used between pipeline stages because it decouples producers from consumers. The ingestion service does not know that a preprocessing service exists — it just publishes a message. If preprocessing is down, the message waits in the queue. When preprocessing restarts, it processes the backlog. This is called **asynchronous decoupling**.

Alternative considered: direct HTTP calls between services. Problem: if preprocessing is slow or down, ingestion's HTTP call blocks or fails. The caller must implement retries, timeouts, circuit breakers. RabbitMQ handles all of this at the infrastructure level.

**PostgreSQL (direct access):** Used for persistent metadata. Multiple services read `video_records`. Rather than routing all DB access through one service's API, services read the DB directly. This is simpler and faster, with the trade-off that the schema is shared — changes require coordination.

**MinIO (direct access):** Video files are too large to pass through HTTP APIs between services. Instead, each service reads and writes to MinIO directly, referencing files by their path. This is shared storage, not shared state.

**Redis (planned):** Will cache search results and hot preview URLs. Not yet wired in the ingestion layer, but the connection URL is already in `BaseServiceSettings`.

---

## Part 5 — Infrastructure Technologies

### PostgreSQL

**What it is:** A relational database. Data is stored in tables with rows and columns. SQL queries retrieve it. PostgreSQL is one of the most reliable open- source databases, used by companies processing billions of rows.

**Why this project uses it:** Metadata about videos (filename, camera ID, duration, status) is structured, relational data. You need to query it with filters ("show me all videos from camera CAM-01 in the last 7 days"). You need ACID transactions — when you insert a `VideoRecord`, either the entire insert succeeds or it is completely rolled back. You do not get partial data.

**What is stored:** `video_records`, `detection_results`, `embedding_records`

**How it interacts:** SQLAlchemy is the Python bridge. Services build Python objects and SQLAlchemy translates them to SQL.

**What breaks if removed:** Every service that reads or writes metadata fails. `VideoRecord` rows no longer persist across restarts. Deduplication (the SHA-256 hash lookup) stops working — the same file can be uploaded infinitely. The status tracking system disappears.

---

### RabbitMQ

**What it is:** A message broker. Think of it as a postal system for software. Service A puts a message in a queue. Service B picks it up later. A and B never talk directly — they only talk to RabbitMQ.

**Core concept — exchange vs queue:** Messages are published to an **exchange**, not directly to a queue. The exchange routes messages to queues based on **routing keys**. This project uses a **topic exchange** (`video.events`) with routing keys like `video.ingested`. A preprocessing service creates a queue (`preprocessing.tasks`) and binds it to pattern `video.ingested`. All messages published with key `video.ingested` are delivered to that queue.

**Why this project uses it:** The pipeline has 5 stages. If stage 3 (detection) is slow, stages 1-2 should not be blocked. RabbitMQ acts as a buffer between stages. A video is ingested immediately; it sits in the queue until detection has capacity to process it.

**Messages that flow through it:**

- `VideoIngestedEvent` — ingestion → preprocessing
- `FramesExtractedEvent` — preprocessing → detection
- `DetectionCompleteEvent` — detection → embedding
- `EmbeddingsReadyEvent` — embedding → indexing

**At-least-once delivery:** A message stays in the queue until the consumer explicitly acknowledges it (`ack`). If the consumer crashes before acking, the message is re-delivered to another consumer. This means the same message might be processed twice — all consumers must handle duplicates gracefully (idempotency).

**What breaks if removed:** The pipeline becomes synchronous. Ingestion cannot signal preprocessing. The entire video processing chain breaks. Videos pile up in PostgreSQL with `status=PENDING` forever, never getting indexed.

---

### MinIO

**What it is:** S3-compatible object storage. Amazon S3 is the cloud standard for storing arbitrary files (objects). MinIO implements the same API but runs on your own server. An "object" is any sequence of bytes with a name (called a key) and a bucket (container).

**Why not PostgreSQL for video files?** PostgreSQL stores data in pages of 8 KB. A 1 GB video would be split across ~125,000 pages. PostgreSQL was not designed for this and would perform very poorly. Object stores stream bytes directly off disk, designed specifically for large binary objects.

**Why not the local filesystem?** Multiple services (ingestion writes, preprocessing reads, search reads for previews) need access to the same files. If files were on the ingestion server's local disk, preprocessing on a different server could not read them. MinIO is network-accessible from all services.

**What files are stored:**

- `raw-videos` bucket: original uploaded video files
- `quarantine-videos` bucket: corrupted or rejected files
- (Future) processed clips, extracted keyframes, detected-object crops

**How retrieval works:** Services call `get_object(bucket, key)` to download. For the dashboard, presigned URLs are generated — a time-limited URL that lets the browser download the file directly from MinIO without going through the Python application.

**What breaks if removed:** Ingestion fails immediately — nowhere to store videos. Preprocessing cannot fetch raw footage. No preview frames for search results.

---

### SQLAlchemy

**What it is:** An ORM (Object-Relational Mapper). It lets you define Python classes that correspond to database tables. Instead of writing SQL strings, you write Python objects and method calls.

**How models become tables:**

```python
class VideoRecord(Base):
    __tablename__ = "video_records"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ...
```

This class definition tells SQLAlchemy: there's a table called `video_records` with a column `id` of type UUID (primary key) and a column `sha256_hash` of type VARCHAR(64). Alembic reads this metadata and generates the SQL `CREATE TABLE` statement.

**Why async?** FastAPI is asynchronous — it handles many HTTP requests concurrently without threads. Standard SQLAlchemy uses blocking I/O (it waits for the DB to respond while blocking). `AsyncSession` uses asyncpg under the hood, which communicates with PostgreSQL asynchronously.

**asyncpg** is a PostgreSQL-specific driver written in Cython. It's much faster than the general-purpose `psycopg2` because it implements the PostgreSQL wire protocol directly, without going through libpq.

**Why this over raw SQL?** Type safety (Python type checker knows the type of every column), refactoring safety (rename a column in the class and your editor finds every usage), migration management via Alembic, automatic connection pooling.

---

### aio-pika / AMQP

**What it is:** An async Python library for talking to RabbitMQ using the AMQP protocol. `aio_pika.connect_robust()` creates a connection that automatically reconnects if RabbitMQ restarts.

**Key concepts in this codebase:**

- `connect_robust()` — reconnects automatically
- `channel.declare_exchange()` — idempotent: creates the exchange if it doesn't exist, does nothing if it already does
- `DeliveryMode.PERSISTENT` — message survives RabbitMQ restart
- `message.process()` context manager — acks on success, nacks on exception

---

### tenacity

**What it is:** A Python retry library. The `@retry` decorator wraps a function and re-calls it if it raises a specified exception.

In `shared/storage/client.py`:

```python
@retry(
    retry=retry_if_exception_type((S3Error, ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def put_object(...):
```

This means: if `put_object` raises `S3Error`, `ConnectionError`, or `TimeoutError`, wait 1 second and try again. Then wait 2 seconds. Then wait 4 seconds. After the third failure, give up and re-raise the original exception.

`wait_exponential` is the standard retry pattern for distributed systems — rapid retries make transient failures (momentary network blip) succeed quickly, while the exponential backoff prevents a thundering-herd problem where hundreds of clients all retry simultaneously and overwhelm a recovering server.

---

### structlog

**What it is:** A structured logging library. Standard Python `logging.info()` produces a plain text string. structlog produces key-value pairs:

```
{"event": "ingestion_accepted", "video_id": "3fa85f...", "filename": "cam.mp4", "size": 156789012}
```

This JSON can be indexed by log aggregation systems (ELK, Datadog, Splunk). You can query: "show me all ingestion failures in the last hour for camera CAM-EAST" without parsing free-form text.

In `DEBUG=true` mode it renders colorised human-readable output. In production it renders JSON.

---

### Watchdog

**What it is:** A Python library that uses OS-level file system events to watch a directory for changes.

On Linux, it uses `inotify` — a kernel facility that delivers events when files are created, modified, or deleted. No polling required; the OS tells you.

In this project, `VideoFileHandler.on_created()` fires when a video file appears in `/tmp/video_watch`. It uses `asyncio.run_coroutine_threadsafe()` to safely bridge the watchdog thread with the asyncio event loop, putting the file path into an async queue.

---

### pydantic-settings

**What it is:** An extension to Pydantic that reads settings from environment variables and `.env` files, with type validation.

```python
class BaseServiceSettings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://..."
    DB_POOL_SIZE: int = 10
```

When `Settings()` is instantiated, it reads `DATABASE_URL` from the environment (or `.env` file). If the value doesn't match the declared type (e.g., you set `DB_POOL_SIZE=banana`), Pydantic raises a validation error immediately at startup rather than crashing later when the value is used.

The `@lru_cache` decorator on `get_settings()` means the settings object is created once and cached. Subsequent calls return the same instance.

---

## Part 6 — Database Deep Dive

### `video_records` — `shared/models/video.py`

**Purpose:** The master record for every video in the system. Created by ingestion, updated by every downstream service.

**Lifecycle:**

- Created by ingestion with `status=PENDING`
- Updated to `status=PROCESSING` when preprocessing picks it up
- Updated to `status=INDEXED` when embedding+indexing completes
- Updated to `status=FAILED` if any stage fails unrecoverably
- Updated to `status=QUARANTINED` if validation fails

**Columns:**

|Column|Type|Purpose|
|---|---|---|
|`id`|UUID|Primary key. UUIDs are globally unique without coordinating with a counter. Safe to generate client-side.|
|`sha256_hash`|VARCHAR(64)|Content fingerprint for deduplication. UNIQUE constraint prevents two rows with the same hash.|
|`original_filename`|VARCHAR(512)|Human-readable reference. Not used for retrieval.|
|`mime_type`|VARCHAR(128)|Detected content type. Stored so downstream services know the format without re-detecting.|
|`file_size_bytes`|BIGINT|Size in bytes. BIGINT because a 10 GB file exceeds INTEGER (2.1 GB limit).|
|`storage_path`|VARCHAR(1024)|MinIO object key. Combined with `storage_bucket`, gives the full address of the file.|
|`storage_bucket`|VARCHAR(256)|MinIO bucket name. Separating bucket from path allows cross-bucket operations.|
|`camera_id`|VARCHAR(256)|Which camera produced this footage. NULL-able because filesystem/RTSP ingests may not know. Indexed for filtering.|
|`location`|VARCHAR(512)|Human-readable location of the camera.|
|`recorded_at`|TIMESTAMPTZ|When the footage was recorded (not when it was uploaded). With timezone.|
|`duration_seconds`|FLOAT|Total video length from FFprobe. NULL if FFprobe unavailable.|
|`resolution_width/height`|INTEGER|Frame dimensions from FFprobe. Used to filter searches ("only HD footage").|
|`status`|ENUM|Pipeline state machine. One of 6 values. Indexed for queries like "all PENDING videos".|
|`error_message`|TEXT|Why processing failed or why a file was quarantined. Long-form text, hence TEXT not VARCHAR.|
|`created_at`|TIMESTAMPTZ|Set by PostgreSQL server clock on insert. `server_default=func.now()` means the DB sets it, not Python.|
|`updated_at`|TIMESTAMPTZ|Set by PostgreSQL on insert and on every update. `onupdate=func.now()` is a SQLAlchemy hook.|

**Indexes:**

- `uq_video_hash` — UNIQUE on `sha256_hash`. Enables O(log n) dedup check.
- `ix_video_status` — B-tree index on `status`. Enables fast "find all PENDING videos" queries.
- `ix_video_camera_id` — B-tree on `camera_id`. Enables fast per-camera queries.
- `ix_video_created_at` — B-tree on `created_at`. Enables fast time-range queries.

**What breaks if this table disappears?** The entire system loses its source of truth. No deduplication. No status tracking. Search results cannot be linked back to video timestamps. The dashboard has nothing to display.

---

### `detection_results` — `shared/models/detection_result.py`

**Purpose:** One row per detected object per frame. If a frame has 3 people and 2 cars, that's 5 rows for that frame.

**Columns:**

|Column|Purpose|
|---|---|
|`video_id`|Foreign key → `video_records.id`. Links detection back to its source video.|
|`frame_path`|MinIO path of the keyframe image this detection came from.|
|`frame_timestamp_ms`|Position in the video in milliseconds. Used to seek to the exact moment in the player.|
|`label`|Free-text class label from Grounding DINO, e.g. "person", "red car", "laptop".|
|`confidence`|Detection confidence score 0.0–1.0.|
|`bbox_x1/y1/x2/y2`|Bounding box in normalised coordinates (0.0–1.0 relative to frame width/height). Normalised so it works regardless of resolution.|
|`crop_path`|MinIO path of the cropped sub-image (the bounding box region). Used by embedding service to generate crop embeddings.|

**Relationships:** Many-to-one with `VideoRecord` (`video_id` FK). One video → many detection results.

**What breaks if this table disappears?** Crop-level embeddings cannot be linked back to their source frames. Search results showing "object at timestamp X" become impossible. The detection service cannot write its output.

---

### `embedding_records` — `shared/models/embedding_record.py`

**Purpose:** One row per embedding vector. Tracks where each vector came from and where it lives in Qdrant.

**Key columns:**

|Column|Purpose|
|---|---|
|`kind`|"frame" / "crop" / "clip" — distinguishes three types of embedding|
|`source_path`|MinIO path of the image/clip that was embedded|
|`model_name`|e.g. `"openai/clip-vit-large-patch14"`. Stored so the system knows if embeddings need to be regenerated after a model swap.|
|`timestamp_ms`|Position in video. Used to deep-link search results to the right moment.|
|`label`|Detection label (for crop embeddings only)|
|`qdrant_point_id`|The ID of the corresponding point in Qdrant. Used to delete or update the vector later.|
|`qdrant_collection`|Which Qdrant collection holds this vector (e.g., `"camera_CAM01_2024_06"`)|
|`vector_dim`|Sanity check. If you switch models and dimension changes from 512 to 768, this helps identify stale embeddings.|

**What breaks if this table disappears?** Cannot map Qdrant search results back to video timestamps. Cannot detect model drift (same video embedded with different models). Re-indexing after Qdrant failure becomes impossible without re-running embedding.

---

## Part 7 — Code Flow Analysis

### Complete call chain for a video upload

```
HTTP POST /api/v1/videos
    │
    ▼
routes.upload_video()                              [api/routes.py]
    │  reads file bytes
    │  parses metadata JSON
    │  calls get_db() → AsyncSession
    │
    ▼
ingestion_service.ingest_upload(db, data, filename, meta)
    │                                              [services/ingestion.py]
    │
    ▼
IngestionService._run_pipeline()
    │
    ├─── Step 1: video_validator.validate(data, filename)
    │                                              [services/validator.py]
    │        VideoValidator.validate()
    │            _detect_mime() → magic.from_buffer()
    │            _compute_hash() → hashlib.sha256()
    │            _ffprobe() → asyncio.create_subprocess_exec("ffprobe ...")
    │        → ValidationResult
    │
    ├─── Step 2: _find_by_hash(db, sha256)
    │        AsyncSession.execute(SELECT WHERE sha256_hash=?)
    │        → None (new) or VideoRecord (duplicate)
    │
    ├─── Step 3: storage_service.upload_video(video_id, data, filename, mime)
    │                                              [services/storage.py]
    │        IngestionStorageService.upload_video()
    │            ObjectStorageClient.put_object(bucket, key, data)
    │                                              [shared/storage/client.py]
    │                loop.run_in_executor(minio_client.put_object)
    │                [tenacity retry × 3 on S3Error/ConnectionError/TimeoutError]
    │        → storage_path string
    │
    ├─── Step 4: db.add(VideoRecord(...)); db.flush()
    │        SQLAlchemy → asyncpg → PostgreSQL
    │        INSERT INTO video_records ...
    │
    ├─── Step 5: mq_publisher.publish_video_ingested(event)
    │                                              [services/queue.py]
    │        IngestionPublisher.publish_video_ingested()
    │            BasePublisher.publish(event, routing_key="video.ingested")
    │                                              [shared/queue/publisher.py]
    │                event.model_dump_json() → bytes
    │                exchange.publish(Message(body=bytes, PERSISTENT))
    │
    └─── Step 6: return VideoIngestResponse(video_id, status, polling_url)

routes.upload_video() returns result
FastAPI serialises to JSON, sends HTTP 202
get_db() context manager: session.commit()
PostgreSQL COMMIT → row is permanently written
```

---

## Part 8 — Shared Folder Analysis

### Why does `shared/` exist at all?

Without it, you would face a choice: duplicate code, or import across services.

**Duplication:** Both ingestion and preprocessing define `VideoIngestedEvent`. They drift. Ingestion adds a `metadata.codec` field. Preprocessing still expects the old schema. Events start failing silently.

**Cross-service imports:** `from services.ingestion.models import VideoRecord`. Now preprocessing depends on ingestion. If ingestion is restructured, preprocessing breaks. You've accidentally coupled two services that should be independent.

`shared/` is the third option: a separate, independently installable library that all services depend on. Changes to shared are versioned. All consumers update together.

### File-by-file analysis

**`shared/config/base.py`** _Consumers:_ Every service's `Settings` class inherits from it. _Impact of deletion:_ Every service's settings class breaks. All env-var reading fails. Every service crashes at import time.

**`shared/db.py`** _Consumers:_ `services/ingestion/db/database.py`. Future: all other services. _Impact of deletion:_ The ingestion service cannot set up its database connection. FastAPI startup fails with ImportError.

**`shared/models/video.py`** _Consumers:_ Ingestion service (writes), all other services (read), Alembic (generates migrations). _Impact of deletion:_ No ORM model for `video_records`. Alembic has no metadata to compare against. All DB queries for video records fail.

**`shared/models/detection_result.py`** _Consumers:_ Detection service (writes), embedding service (reads), search service (reads). _Impact of deletion:_ Detection cannot write its output. Search cannot return frame-level timestamps.

**`shared/models/embedding_record.py`** _Consumers:_ Embedding service (writes), indexing service (reads and updates), search service (reads). _Impact of deletion:_ Cannot track which vectors are in Qdrant. Re-indexing after Qdrant failure becomes impossible.

**`shared/events/video_ingested.py`** _Consumers:_ Ingestion (publishes), preprocessing (consumes). _Impact of deletion:_ Ingestion cannot build the event message. Preprocessing cannot parse received messages. The first pipeline hop breaks.

**`shared/events/frames_extracted.py`** _Consumers:_ Preprocessing (publishes), detection (consumes). _Impact of deletion:_ Second hop breaks. Detection never receives work.

**`shared/events/detection_complete.py`** _Consumers:_ Detection (publishes), embedding (consumes). _Impact of deletion:_ Third hop breaks. No embeddings generated.

**`shared/events/embeddings_ready.py`** _Consumers:_ Embedding (publishes), indexing (consumes). _Impact of deletion:_ Fourth hop breaks. Vectors never enter Qdrant.

**`shared/queue/publisher.py`** _Consumers:_ `services/ingestion/services/queue.py`, all other services' queue files. _Impact of deletion:_ All publisher classes lose their base. Every service stops publishing events. The whole pipeline goes silent.

**`shared/queue/consumer.py`** _Consumers:_ Preprocessing, detection, embedding, indexing (when built). _Impact of deletion:_ No service can consume from RabbitMQ. The pipeline can publish but never process.

**`shared/storage/client.py`** _Consumers:_ `services/ingestion/services/storage.py`, preprocessing, search. _Impact of deletion:_ No service can read or write files. Ingestion fails at step 3. Preprocessing cannot fetch raw footage.

**`shared/logging.py`** _Consumers:_ All services via their `core/logging.py` re-export. _Impact of deletion:_ `configure_logging()` and `get_logger()` cannot be imported. All services fail at startup with ImportError.

---

## Part 9 — Test Analysis

**File:** `services/ingestion/tests/test_ingestion.py`

All tests use `AsyncMock` (for async functions) and `MagicMock` (for sync functions) to replace real infrastructure with fakes. No database, MinIO, or RabbitMQ is needed to run them. This is the **unit testing** style.

---

### `TestVideoValidator`

#### `test_invalid_extension_rejected`

**Scenario:** User uploads a file named `video.exe`. **What it tests:** The extension check in `VideoValidator.validate()`. **Why it matters:** Without this, attackers could upload executables. The extension check is the first line of defence. **What fails if removed:** The extension guard could be silently bypassed by code refactors. A developer changing the allowed extensions set might accidentally remove `.mp4` or add `.exe` without a test catching it. **Production protection:** Prevents executable uploads, PHP files, or anything that isn't a video container format.

---

#### `test_sha256_computed`

**Scenario:** Valid MP4 bytes pass MIME check and FFprobe. Verify hash is computed correctly. **What it tests:** `_compute_hash()` produces the correct SHA-256. The test independently computes `hashlib.sha256(data).hexdigest()` and compares. **Why it matters:** The hash is the deduplication key. If the hash is wrong (e.g., only hashes part of the file), two different files with the same partial hash would be falsely detected as duplicates. **What fails if removed:** A bug in `_compute_hash()` (e.g., only hashing the first 65536 bytes) would silently cause false deduplication. Files would be rejected as duplicates when they aren't.

---

#### `test_corrupt_file_fails`

**Scenario:** MIME check passes (magic returns "video/mp4"), but FFprobe returns `None` (file is not a valid video despite the extension and MIME). **What it tests:** The FFprobe failure path. When `_ffprobe()` returns `None`, `is_valid` must be `False` and `error_reason` must mention "corrupt". **Why it matters:** A file can have a `.mp4` extension and even fake MP4 MIME bytes in its header, but still be corrupt. FFprobe is the definitive check. **Production protection:** Without this, corrupt files would make it into the pipeline. Preprocessing would fail trying to transcode them, producing obscure errors deep in the pipeline rather than a clean 422 at upload time.

---

#### `test_ffprobe_metadata_extracted`

**Scenario:** FFprobe returns rich metadata (duration, resolution, codec). Verify all fields are correctly parsed into `ValidationResult`. **What it tests:** The JSON parsing logic in `_ffprobe()`. Verifies that `format.duration` → `duration_seconds`, stream `width`/`height` → `resolution_width`/`resolution_height`. **Why it matters:** These fields are written to `VideoRecord`. If parsing is wrong, the dashboard shows incorrect video metadata. The embedding service might use resolution to calculate crop coordinates — wrong resolution = wrong crops. **Production protection:** Catches any change to the FFprobe output format or the parsing logic that would silently corrupt metadata.

---

### `TestIngestionService`

#### `test_duplicate_returns_existing`

**Scenario:** `_find_by_hash()` returns an existing `VideoRecord`. Verify the service returns `DuplicateVideoResponse` with the existing `video_id` and does not call storage or the DB again. **What it tests:** The deduplication branch in `_run_pipeline()`. **Why it matters:** Without deduplication, every re-upload stores a new copy in MinIO (wasting storage), inserts a new DB row (wasting space), and produces a new event (causing double-processing of the same video). **Production protection:** Protects against cameras that re-upload footage after a network interruption, and against users who upload the same file twice accidentally.

---

#### `test_invalid_file_quarantined_and_raises`

**Scenario:** Validation returns `is_valid=False`. Verify two things: (1) the `_quarantine()` method is called, and (2) `IngestionError(422)` is raised. **What it tests:** The quarantine path. Both actions must happen — neither alone is sufficient. **Why it matters:** If quarantine is skipped, corrupt files disappear with no record of why they were rejected. If the exception is not raised, the pipeline continues processing an invalid file. **Production protection:** Ensures audit trail of all rejected files. The quarantine bucket becomes evidence for debugging upload issues.

---

#### `test_minio_failure_raises_503`

**Scenario:** `storage_service.upload_video()` raises `ConnectionError`. Verify the service raises `IngestionError(503)`. **What it tests:** The MinIO failure branch. 503 Service Unavailable is the correct HTTP status for "please try again later, infrastructure is down". **Why it matters:** Without this, a MinIO outage would produce an unhandled exception and a generic 500 Internal Server Error, hiding the root cause. A 503 tells the client "retry later". **Production protection:** Correct error codes allow load balancers and retry logic to automatically retry on 503 and not on 500.

---

#### `test_db_failure_cleans_up_storage`

**Scenario:** MinIO upload succeeds, but `db.flush()` raises an exception. Verify: (1) `storage_service.delete_object()` is called with the correct path, (2) `IngestionError(500)` is raised. **What it tests:** The cleanup path. This is the most critical failure test because it verifies **consistency**: if the DB write fails, the MinIO upload must be undone. Otherwise the system is in an inconsistent state — there's a file in MinIO with no corresponding DB record. **Why it matters:** Without cleanup, orphaned files accumulate in MinIO indefinitely. Storage costs grow. There is no way to identify which files have records and which are orphans. **Production protection:** Maintains the invariant: every file in MinIO has a corresponding `VideoRecord`. Prevents storage leaks.

---

#### `test_successful_ingest_publishes_event`

**Scenario:** Happy path — all steps succeed. Verify: (1) result is `VideoIngestResponse` with `status=PENDING`, (2) `publish_video_ingested()` is called exactly once. **What it tests:** The complete success path and the event publish. **Why it matters:** The event is what triggers downstream processing. If it's not published, the pipeline stops silently — the video is ingested and stored, but never processed, never indexed, never searchable. **Production protection:** Catches any code path where the event publish is skipped (e.g., an `if` branch introduced by a refactor that exits before publishing).

---

### `TestIngestionStorageService`

#### `test_upload_returns_object_path`

**Scenario:** `put_object()` is mocked to succeed. Verify the returned path contains both the video_id and the filename. **What it tests:** The object naming convention in `upload_video()`. **Why it matters:** The path `{video_id}/{filename}` is stored in `VideoRecord` and used by all downstream services to fetch the file. If the path format changes, downstream services break. This test locks the format. **Production protection:** Catches any change to the path format that would cause a mismatch between what's stored in `video_records.storage_path` and what MinIO actually contains.

---

#### `test_quarantine_uses_quarantine_bucket`

**Scenario:** A quarantine upload is triggered. Verify it goes to the `MINIO_QUARANTINE_BUCKET`, not `MINIO_RAW_BUCKET`. **What it tests:** The bucket routing in `quarantine()`. **Why it matters:** If corrupt files go to the raw bucket, they mix with valid footage. Downstream services trying to process them would fail cryptically. The quarantine bucket is separate so operators can review rejected files without them polluting the main pipeline. **Production protection:** Ensures that rejected files are isolated. A corrupt file in `raw-videos` could cause preprocessing to crash repeatedly as it tries to transcode something that isn't a valid video.

---

## Part 10 — Author Mindset Reconstruction

### What the author was optimising for

**Reliability over simplicity.** Every failure mode is handled explicitly. MinIO retries 3× before failing. DB failures clean up storage. Queue publish failures are non-fatal (the DB record survives). This reflects a mindset where "what goes wrong" is as important as "what goes right".

**Extensibility over DRY within services.** The `shared/` pattern means more files, not fewer. But it means adding the 6th service is copy-paste of the 5th with 20 lines changed. The author chose more boilerplate now for lower marginal cost per new service.

**Correctness of data over performance.** SHA-256 is computed for every upload. FFprobe is run as a subprocess. These are slow. The author accepted the latency hit because the cost of incorrect dedup or corrupted pipeline entries is higher than the cost of a 200ms validation delay.

### Why RabbitMQ over alternatives

**Alternative: direct HTTP calls between services.** Problem: synchronous coupling. If preprocessing is slow, ingestion blocks. Error handling becomes each service's responsibility.

**Alternative: Apache Kafka.** More powerful but more complex. Kafka is designed for very high throughput (millions of messages/second) and replay. RabbitMQ is simpler to operate for message-passing between microservices at this scale.

**Alternative: Celery + Redis.** Celery is a Python-specific task queue. It works well for Python-only systems. This project's pipeline might eventually use services in different languages (a Go service for preprocessing, a C++ service for detection). AMQP/RabbitMQ is language-agnostic.

### Why MinIO over alternatives

**Alternative: Local filesystem.** Doesn't work across multiple servers. **Alternative: Amazon S3.** Works, but costs money in development and creates cloud vendor dependency. MinIO is drop-in S3-compatible — switching to S3 in production means changing one environment variable (`MINIO_ENDPOINT`). **Alternative: PostgreSQL Large Objects.** PostgreSQL supports storing binary blobs but is not designed for streaming large files.

### Why SQLAlchemy over raw SQL

**Alternative: raw asyncpg queries.** Faster, but no type safety. Column names are strings. Renames break queries silently. **Alternative: tortoise-orm.** Less mature, smaller community. **SQLAlchemy** is the industry standard. The typed `Mapped[]` annotations (introduced in SQLAlchemy 2.0) give full IDE completion and type-checker coverage on database columns.

### Scalability goals visible in the code

- Connection pooling (`DB_POOL_SIZE=10`, `DB_MAX_OVERFLOW=20`) — the engine can serve 30 concurrent DB operations without creating 30 connections.
- Stateless services — `IngestionService()` has no instance state. Multiple processes can run it without sharing memory.
- Pre-created exchange, durable queues — RabbitMQ configuration survives restarts, no data loss on service crash.
- `@lru_cache` on `get_settings()` — settings parsed once, not per-request.

### Future extensibility visible in the code

- `BaseConsumer.handle_message()` is abstract — adding a new consumer is subclassing and implementing one method.
- `BasePublisher.publish()` takes any Pydantic model — adding a new event type is defining a new Pydantic class and calling `publish()`.
- `BaseServiceSettings` has `REDIS_URL` already — Redis caching layer is designed in, just not wired yet.
- Events carry full metadata (camera_id, timestamp, codec) in their payload — consumers don't need to make additional DB lookups to decide how to process.

---

## Part 11 — State and Ownership Analysis

### Where does state live?

|State type|Primary location|Secondary|Lifecycle|
|---|---|---|---|
|Raw video files|MinIO `raw-videos`|Quarantine bucket|Created on ingest, never deleted unless explicitly purged|
|Video metadata|PostgreSQL `video_records`|RabbitMQ event (transient)|Created on ingest, updated through pipeline stages|
|Detection results|PostgreSQL `detection_results`|MinIO (crop images)|Created by detection, read by embedding/search|
|Embedding vectors|Qdrant|PostgreSQL `embedding_records` (metadata)|Created by embedding, updated by indexing (adds Qdrant ID)|
|Pipeline events|RabbitMQ|None|Created by each stage, consumed by next stage, ACK'd and deleted|
|Search cache|Redis (planned)|None|Created on first search, TTL-expired|
|Application config|Environment variables / `.env`|None|Loaded once at startup|
|Processing status|PostgreSQL `video_records.status`|None|Updated at each pipeline stage|

### Source of truth analysis

**For "does this video exist?"** → `video_records.sha256_hash` (PostgreSQL). The SHA-256 is computed from file content, independent of filename or upload source.

**For "is this video searchable?"** → `video_records.status = INDEXED`.

**For "where is the raw file?"** → `video_records.storage_path` + `.storage_bucket`.

**For "which vectors are in Qdrant?"** → `embedding_records.qdrant_point_id`.

**For "what was detected at timestamp X?"** → `detection_results` filtered by `video_id` and `frame_timestamp_ms`.

---

## Part 12 — Destruction Analysis

### What happens if you delete each component?

**`services/ingestion/api/routes.py`** Immediate: FastAPI has no routes registered. Every HTTP request returns 404. Downstream: Nothing — no videos are ingested, but existing data is unaffected. Recovery: Restore the file and restart.

**`services/ingestion/services/ingestion.py`** Immediate: `routes.py` cannot import `ingestion_service`. Service fails to start. Downstream: Same as above — no new ingestions. Recovery: Restore file, restart.

**`services/ingestion/db/database.py`** Immediate: Import error at startup. FastAPI cannot start (routes imports `get_db`). Filesystem watcher cannot create sessions. Recovery: Restore file, restart.

**`shared/models/video.py`** Immediate: Import error everywhere that imports `VideoRecord`. Ingestion cannot write records. Alembic cannot generate migrations. Recovery: Restore file. If table was also dropped from PostgreSQL, run migrations to recreate.

**PostgreSQL `video_records` table** Immediate: All `SELECT` and `INSERT` queries fail with "relation does not exist". Ingestion step 2 (dedup check) fails. Step 4 (insert record) fails. The pipeline stops after MinIO upload but before DB insert — files pile up in MinIO with no DB record. Recovery: Run `alembic upgrade head` to recreate the table. Orphaned MinIO files need manual reconciliation.

**PostgreSQL `detection_results` table** Immediate: Detection service cannot write. Embedding service cannot read bounding boxes for crop embeddings. Recovery: Run migrations to recreate. Re-run detection on all INDEXED videos.

**MinIO `raw-videos` bucket** Immediate: New uploads succeed (MinIO creates the bucket on first `put_object` due to `ensure_bucket()`). Preprocessing cannot fetch existing videos — they are gone. Recovery: Restore from backup. Re-run the entire pipeline from ingestion for all affected videos (set their status back to PENDING and republish events).

**MinIO `quarantine-videos` bucket** Immediate: New corrupt files cannot be quarantined. `quarantine()` call raises `S3Error`. Ingestion logs the error and continues — the `VideoRecord` is still inserted with `status=QUARANTINED` even if the file wasn't stored. Recovery: Ensure bucket exists (`ensure_bucket()` at startup handles this).

**RabbitMQ** Immediate: `mq_publisher.startup()` fails. Service fails to start. If RabbitMQ goes down while service is running: `publish_video_ingested()` raises an exception, which is caught and logged (non-fatal). Videos are ingested and stored in PostgreSQL and MinIO but events are lost — they will never be preprocessed unless replayed. Recovery: Restart RabbitMQ. Implement replay: scan `video_records` for `status=PENDING` older than N minutes and republish events.

**Redis** Immediate: No impact yet — Redis is not actively used in the current codebase. Future impact: Search result cache misses, every search hits PostgreSQL/Qdrant. Recovery: Redis is stateless cache — just restart it. No data to restore.

**`shared/queue/publisher.py`** Immediate: `IngestionPublisher` loses its base class. Import error. Service fails to start. Recovery: Restore file, restart.

**`shared/events/video_ingested.py`** Immediate: `services/ingestion/services/queue.py` cannot import `VideoIngestedEvent`. Service fails to start. Recovery: Restore file, restart.

**`infra/migrations/versions/0001_initial.py`** Immediate: No immediate impact — migration has already run. Future impact: Cannot roll back migration. Cannot regenerate the initial schema on a fresh database. Recovery: Restore file. If the table exists, no harm done. If you need a fresh database, you cannot run `alembic upgrade head` without it.

**`services/ingestion/workers/fs_watcher.py`** Immediate: Import error in `main.py` (it imports `start_filesystem_watcher`). Service fails to start. Recovery: Restore file, restart. Files dropped into the watch directory during the outage are not automatically processed — they sit there until manually ingested via the `/fs` endpoint.

**`services/ingestion/tests/test_ingestion.py`** Immediate: No production impact. CI/CD pipeline fails to run tests. Future regressions go undetected. Recovery: Restore file. But any bugs introduced while the tests were missing may have already reached production.

**`.env` file** Immediate: `pydantic-settings` falls back to default values. Service connects to `localhost` for PostgreSQL, MinIO, RabbitMQ. In production (where infra is on different hosts), all connections fail. Recovery: Recreate the file with correct values, restart.

---

## Summary: The Five Questions

**What does this system do?** Accepts video, validates and stores it, runs an ML pipeline (transcoding → object detection → embedding → vector indexing), and makes the footage semantically searchable by natural language or image query.

**Why was it designed this way?** Pipeline stages have very different compute needs (GPU vs CPU) and different failure modes. Separating them lets each scale and fail independently. Shared infrastructure code prevents the seven services from drifting apart.

**Where does state live?** Raw files in MinIO. Metadata and status in PostgreSQL. Vectors in Qdrant. Events in-flight in RabbitMQ. Config in environment variables.

**Where does feedback live?** `VideoRecord.status` and `VideoRecord.error_message` are the feedback channel. Quarantined files have their rejection reason stored. The webhook fires for human notification. Future: the search ranking scores feed back into retrieval quality.

**How do services communicate?** Client→service: HTTP REST. Service→service (pipeline): RabbitMQ async events. Service→storage: MinIO S3 API. Service→metadata: PostgreSQL direct. Service→ cache: Redis (planned).

**What happens if component X disappears?** PostgreSQL: data integrity fails, dedup fails, status tracking fails. MinIO: files cannot be stored or retrieved. RabbitMQ: pipeline stages cannot signal each other, processing stops. shared/: import errors, services cannot start. Tests: no immediate production impact, but regressions accumulate undetected.

---

## Completion — Missing Sections

The sections below complete the architecture review, filling gaps identified after the initial document was written.

---

## Part 2 (continued) — Remaining Folders

### `core/` — `services/ingestion/core/`

**Why it exists:** Every service needs two things before it can do anything useful: configuration (where is the database? what are the bucket names?) and logging (how do I write structured output?). Putting these in `core/` separates cross-cutting concerns from business logic.

**Files:**

`config.py` — Defines `Settings(BaseServiceSettings)`. This class inherits all shared infrastructure config and adds ingestion-specific fields: bucket names, allowed extensions, MIME types, the routing key, the watch directory path, and the quarantine webhook URL. The `@lru_cache` on `get_settings()` means `Settings()` is only instantiated once — Pydantic doesn't re-parse environment variables on every function call.

`logging.py` — A three-line re-export of `shared.logging`. It exists so all code inside `services/ingestion/` can write `from services.ingestion.core.logging import get_logger` rather than reaching into `shared` directly. If the shared logging implementation ever moves, only this file needs updating.

**What would happen if deleted:**

- `config.py` deleted: every other file in the service fails to import — they all start with `from services.ingestion.core.config import get_settings`. Service cannot start.
- `logging.py` deleted: every file that calls `get_logger` fails to import. Service cannot start.

**Why separated from `services/`:** `services/` contains stateful business logic that calls external systems. `core/` contains pure configuration and utilities with no external calls. This distinction makes `core/` trivially testable and makes `services/` clearly responsible for side effects.

---

### `models/` — `services/ingestion/models/`

**Why it exists:** Contains Pydantic schemas that are specific to the ingestion service's HTTP interface. These are not ORM models (those live in `shared/models/`). These are request body shapes and response body shapes.

**The key distinction:**

- `shared/models/video.py` → SQLAlchemy ORM → becomes a PostgreSQL table
- `services/ingestion/models/schemas.py` → Pydantic → validates HTTP request JSON, serialises HTTP response JSON

**What lives here:** `schemas.py` — covered in full in Part 6 below.

**What would happen if deleted:** `api/routes.py` cannot import `VideoIngestResponse`, `VideoUploadMetadata`, `RTSPIngestRequest` etc. FastAPI cannot parse request bodies or serialise responses. Service fails to start.

**Why separated from `shared/`:** These schemas are HTTP-specific. The preprocessing service consuming a `VideoIngestedEvent` from RabbitMQ has no need for `VideoUploadMetadata` (which is an HTTP upload form shape). Keeping HTTP schemas in the service prevents them polluting the shared contract layer.

---

### `workers/` — `services/ingestion/workers/`

**Why it exists:** Contains long-running background processes that are not HTTP request handlers. The filesystem watcher runs as an `asyncio.Task` for the entire lifetime of the service — it never returns unless cancelled.

**What lives here:** `fs_watcher.py` — analysed in detail in Part 7.

**What would happen if deleted:** `main.py` imports `start_filesystem_watcher` from here. Service fails to start with `ImportError`. The filesystem drop functionality (FR-ING-01) is completely lost — files placed in the watch directory are never ingested.

**Why separated from `api/`:** API handlers are short-lived (they handle one request and return). Workers are long-lived (they run forever). Mixing them would make the codebase confusing — a developer looking in `api/` expects to find request handlers, not daemon processes.

**Why separated from `services/`:** `services/` contains the business logic (`ingestion.py`, `validator.py`). `workers/` contains the runtime harness that calls the business logic on a schedule or trigger. The filesystem watcher is a trigger mechanism, not business logic. `_process_queue()` in `fs_watcher.py` is literally one function that calls `ingestion_service.ingest_filesystem()`.

---

### `infra/migrations/` — Alembic migration project

**Why it exists:** When you change a SQLAlchemy model (add a column, add an index), the Python class changes but the PostgreSQL table does not. You need a way to apply the change to the database without dropping and recreating it (which would destroy all data). Alembic is the standard migration tool for SQLAlchemy projects.

**Files:**

`alembic.ini` — Alembic's own configuration file. Specifies the database URL and the location of migration scripts.

`env.py` — The bridge between Alembic and your models. The critical line is:

```python
from shared.models.video import Base
target_metadata = Base.metadata
```

This tells Alembic: compare the current database schema against the Python `Base.metadata` object (which knows about every `Mapped` class defined across all three model files). Generate migration scripts that bring the database in line with the Python definitions.

`versions/0001_initial.py` — The first (and currently only) migration. Running `alembic upgrade head` executes the `upgrade()` function, which creates `video_records`, `detection_results`, and `embedding_records` with all their columns and indexes. Running `alembic downgrade base` executes `downgrade()`, which drops them.

**Why a single migration project for all services?** All three model files (`video.py`, `detection_result.py`, `embedding_record.py`) inherit from the same `Base`. They share one PostgreSQL database (`surveillance`). Managing them with separate Alembic projects would create coordination problems — who runs migrations first? Does ingestion's migration conflict with detection's? One project, one database, one migration history.

**What would happen if this folder were deleted:**

- Immediate: No impact — the database tables already exist from a previous migration run.
- Next fresh database: `alembic upgrade head` fails with "no such revision". The only way to create the database schema is to run raw SQL manually — error-prone and undocumented.
- After a model change: No way to generate a migration. The developer must write raw `ALTER TABLE` SQL. The change cannot be reviewed, versioned, or rolled back systematically.

---

## Part 5 (continued) — Redis

### Redis

**What it is:** Redis is an in-memory data store. Unlike PostgreSQL (which stores data on disk and reads it into memory on demand), Redis keeps all data in RAM. This makes it extremely fast — reads and writes complete in under a millisecond.

Redis supports many data structures: strings, lists, sets, sorted sets, and hash maps. This makes it suitable for many use cases beyond simple key-value caching.

**Why this project will use it:** Three planned use cases:

1. **Search result caching.** The query `"person in red jacket near entrance"` might be submitted 50 times by different investigators during a shift. The first query hits Qdrant (20-50ms). The result can be stored in Redis with a key like `search:hash(query+filters)` and a TTL (time-to-live) of 5 minutes. The next 49 queries return in <1ms from Redis.
    
2. **Preview frame URL caching.** MinIO presigned URLs (generated by `presigned_get_url()`) are time-limited. Generating one requires calling MinIO. If the same frame is requested 100 times in the dashboard, 100 MinIO calls could be avoided by caching the URL in Redis until it expires.
    
3. **Rate limiting.** `FR-API-05` requires per-API-key rate limiting. Redis's atomic `INCR` and `EXPIRE` commands are the standard implementation: `INCR ratelimit:{api_key}:{minute}` increments a counter, `EXPIRE` sets it to auto-delete after 60 seconds. If the counter exceeds the limit, return 429.
    

**What data will be stored:** (planned)

- `search:{query_hash}` → JSON array of search results, TTL 5 min
- `url:{video_id}:{frame_path}` → presigned URL string, TTL 1 hour
- `ratelimit:{api_key}:{minute}` → integer counter, TTL 60 seconds

**How it interacts with other services:** The search service writes and reads search cache. The API gateway reads rate limit counters. The dashboard reads preview URL cache. None of these are implemented yet — but `REDIS_URL` is already in `BaseServiceSettings` so adding Redis requires no config changes.

**What breaks if Redis is removed:**

- Currently: nothing — Redis is not actively used in the ingestion pipeline.
- After search caching is implemented: every search hits Qdrant directly. Performance degrades at high query volume. No data is lost — just slower.
- After rate limiting is implemented: rate limiting stops working. The API becomes unprotected against abuse. This is a security concern, not just a performance concern.

**Why Redis instead of alternatives:**

_Alternative: PostgreSQL for caching._ Possible, but PostgreSQL stores data on disk and is 100× slower for cache-style reads. Adding cache columns to `video_records` would also violate single-responsibility.

_Alternative: In-memory Python dict._ Works for a single process, but breaks with multiple instances — each process has its own cache with no sharing. Redis is shared across all instances.

_Alternative: Memcached._ Simpler than Redis but only supports strings. Cannot do atomic rate limiting counters or sorted sets for leaderboards.

---

## Part 6 (continued) — `schemas.py` Analysis

**File:** `services/ingestion/models/schemas.py`

This file contains the HTTP layer's data contracts. Every class here is a Pydantic model — it defines what JSON shapes the API accepts and returns. None of these become database tables.

---

### Request bodies

**`VideoUploadMetadata`**

```python
class VideoUploadMetadata(BaseModel):
    camera_id:   str | None = Field(None, max_length=256)
    location:    str | None = Field(None, max_length=512)
    recorded_at: datetime | None = None
```

This is the optional JSON the client sends alongside the video file in the multipart upload. All three fields are `None`-able — a valid upload has no metadata at all. `max_length` constraints are enforced by Pydantic before the service layer sees the data, so `IngestionService` never receives a `camera_id` longer than 256 characters.

**Why a separate class for metadata rather than putting fields on the form directly?** Because multipart forms don't support nested JSON natively. The client sends one `metadata` form field as a JSON string. `routes.py` parses that string into `VideoUploadMetadata`. This pattern lets you add metadata fields without changing the multipart form structure.

**`RTSPIngestRequest`**

```python
class RTSPIngestRequest(BaseModel):
    rtsp_url:         str
    camera_id:        str | None = None
    location:         str | None = None
    duration_seconds: int = Field(default=60, ge=5, le=3600)
```

JSON body for `POST /api/v1/videos/rtsp`. `ge=5` (greater-or-equal) and `le=3600` are Pydantic validators. If a client sends `duration_seconds: 0`, Pydantic rejects it with a 422 before the service is called. The service never needs to validate this itself.

**`FilesystemIngestRequest`**

```python
class FilesystemIngestRequest(BaseModel):
    file_path:   str
    camera_id:   str | None = None
    location:    str | None = None
    recorded_at: datetime | None = None
```

JSON body for `POST /api/v1/videos/fs`. Note `file_path` is a raw string with no path validation at the Pydantic level — the service layer handles the `FileNotFoundError` if the path doesn't exist.

---

### Response bodies

**`VideoIngestResponse`**

```python
class VideoIngestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    video_id:    uuid.UUID
    status:      str
    polling_url: str
    message:     str = "Video accepted for processing"
```

Returned as HTTP 202. `from_attributes=True` (previously called `orm_mode`) means Pydantic can create this from a SQLAlchemy ORM object directly, not just from a dict. This allows `VideoStatusResponse.model_validate(record)` where `record` is a `VideoRecord` ORM object.

`polling_url` gives the client a ready-made URL to check status, following the REST principle of hypermedia — the API tells you what you can do next, you don't have to construct URLs yourself.

**`DuplicateVideoResponse`**

```python
class DuplicateVideoResponse(BaseModel):
    video_id:              uuid.UUID
    status:                str
    message:               str = "Duplicate: video already exists"
    existing_storage_path: str | None
```

Returned as HTTP 200 (not 202) when the SHA-256 hash matches an existing video. Giving the client the `existing_storage_path` means they can verify which file it matched without making a second request.

**Why 200 for duplicates and not 409 Conflict?** 409 would mean "something went wrong, the request cannot be processed". But a duplicate is not an error — the video is already in the system. The client's goal (have this video available for search) is already achieved. 200 communicates: "your request was understood and the desired state already exists".

**`VideoStatusResponse`**

The full metadata view returned by `GET /api/v1/videos/{id}/status`. Contains every field from `VideoRecord` except the internal `sha256_hash` (not useful to clients) and `storage_bucket` (an implementation detail). This is the polling endpoint the client uses to know when `status` transitions from `PENDING` to `INDEXED`.

**`ErrorResponse`**

```python
class ErrorResponse(BaseModel):
    detail: str
    code:   str | None = None
```

Used in OpenAPI documentation to describe error response shapes. FastAPI automatically uses this in the generated Swagger UI when you declare it in `responses={}`. It matches FastAPI's default error format so clients see a consistent structure for both framework errors (422 validation) and application errors (503 storage unavailable).

---

### Why Pydantic for HTTP schemas?

**Validation is declarative.** `duration_seconds: int = Field(ge=5, le=3600)` — one line replaces five lines of `if/raise` validation code. Pydantic generates clear error messages automatically: `{"detail": [{"loc": ["body", "duration_seconds"], "msg": "Input should be greater than or equal to 5"}]}`.

**Type coercion.** If the client sends `"recorded_at": "2024-06-01T08:30:00Z"`, Pydantic converts the string to a Python `datetime` object automatically. The service receives a `datetime`, not a string to parse.

**OpenAPI generation.** FastAPI reads the Pydantic model and generates the Swagger UI documentation automatically. The `examples=["CAM-EAST-01"]` in `Field()` appears in the Swagger UI so developers know what format to use.

---

## Part 7 (continued) — Additional Call Chains

### RTSP ingest call chain

```
POST /api/v1/videos/rtsp
  {"rtsp_url": "rtsp://192.168.1.100:554/stream", "duration_seconds": 60}
    │
    ▼
routes.ingest_rtsp(body: RTSPIngestRequest, db)      [api/routes.py]
    │
    ▼
ingestion_service.ingest_rtsp(db, rtsp_url, duration, camera_id, location)
    │                                                 [services/ingestion.py]
    │
    ▼
IngestionService._capture_rtsp(url, duration)
    │  asyncio.create_subprocess_exec(
    │      "ffmpeg", "-rtsp_transport", "tcp",
    │      "-i", rtsp_url, "-t", "60", "-c", "copy", "/tmp/uuid.mp4"
    │  )
    │  wait_for(proc.communicate(), timeout=duration+30)
    │  aiofiles.open(tmp_path, "rb") → bytes
    │  os.unlink(tmp_path)
    │  → (bytes, "rtsp_capture_a3f4b.mp4")
    │
    ▼
IngestionService._run_pipeline(db, file_data, filename, camera_id, ...)
    │  [same 6-step pipeline as HTTP upload]
    └─ Steps 1-6 identical to HTTP upload chain
```

**Key difference from HTTP upload:** `_capture_rtsp()` runs FFmpeg as a subprocess with a timeout of `duration + 30` seconds. The `+30` gives FFmpeg time to finalise the output file after the capture duration expires. If FFmpeg hangs, `asyncio.wait_for()` kills it after the timeout.

**Failure mode unique to RTSP:** If the camera is offline, FFmpeg exits with non-zero return code and `stderr` contains the error. `_capture_rtsp()` raises `IngestionError(422)` with the FFmpeg stderr as the message.

---

### Filesystem ingest call chain

```
POST /api/v1/videos/fs
  {"file_path": "/mnt/nas/cam01/footage.mp4", "camera_id": "CAM-NAS"}
    │
    ▼
routes.ingest_filesystem(body: FilesystemIngestRequest, db)
    │
    ▼
ingestion_service.ingest_filesystem(db, file_path, camera_id, location, recorded_at)
    │
    ▼
IngestionService.ingest_filesystem()
    │  aiofiles.open(file_path, "rb") → bytes
    │  [reads entire file into memory]
    │
    ▼
IngestionService._run_pipeline(...)
    └─ Steps 1-6 identical
```

**Filesystem watcher variant of the same chain:**

```
File appears in /tmp/video_watch/cam_feed_1234.mp4
    │
    ▼ (OS inotify event via watchdog)
VideoFileHandler.on_created(FileCreatedEvent)
    │  checks: is extension in ALLOWED_EXTENSIONS?
    │  asyncio.run_coroutine_threadsafe(queue.put(path), loop)
    │  [bridges watchdog thread → asyncio event loop]
    │
    ▼
_process_queue(queue)  [running as asyncio.Task]
    │  path = await queue.get()
    │  async with AsyncSessionLocal() as db:  [own session, not from HTTP]
    │
    ▼
ingestion_service.ingest_filesystem(db, path, camera_id=None, ...)
    └─ same _run_pipeline() chain
```

**Key difference:** The filesystem watcher creates its own `AsyncSessionLocal()` session directly — it has no FastAPI `Depends(get_db)` because it's not inside a request handler. The session commit is explicit: `await db.commit()` after the ingest call.

**Important subtlety:** `asyncio.run_coroutine_threadsafe()` is necessary because watchdog's event callbacks run in a separate OS thread (not the asyncio event loop thread). Directly calling `await queue.put()` from a thread would crash. `run_coroutine_threadsafe` safely schedules a coroutine to run on the event loop from a thread.

---

### Status polling call chain

```
GET /api/v1/videos/3fa85f64-.../status
    │
    ▼
routes.get_video_status(video_id: uuid.UUID, db)     [api/routes.py]
    │
    ▼
ingestion_service.get_status(db, video_id)           [services/ingestion.py]
    │
    ▼
db.execute(SELECT * FROM video_records WHERE id = video_id)
    │  asyncpg → PostgreSQL
    │  → VideoRecord ORM object (or None)
    │
    ▼  (if None)
raise HTTPException(404, "Video not found")

    ▼  (if found)
VideoStatusResponse.model_validate(record)
    │  Pydantic reads attributes from ORM object
    │  → VideoStatusResponse dict
    │
    ▼
FastAPI serialises to JSON → HTTP 200
```

**This is a read-only chain.** No writes. No events. No MinIO calls. The session opened by `get_db()` is committed with no changes.

**Polling pattern rationale:** The client calls this endpoint repeatedly (every 2-5 seconds) until `status` is `INDEXED`. This is simpler than WebSockets for a team of two — no persistent connection to manage. The downside is extra HTTP requests, but at one request per 5 seconds per video, the load is negligible.

---

### Quarantine call chain

```
Validation fails (corrupt file / bad MIME / bad extension)
    │
    ▼
IngestionService._run_pipeline()
    │  validation = await video_validator.validate(data, filename)
    │  validation.is_valid == False
    │
    ▼
IngestionService._quarantine(db, video_id, data, filename, reason)
    │
    ├─── storage_service.quarantine(video_id, data, filename)
    │        ObjectStorageClient.put_object(
    │            "quarantine-videos",
    │            "{video_id}/{filename}",
    │            data,
    │            "application/octet-stream"   ← not video/mp4; we don't trust the MIME
    │        )
    │
    ├─── db.add(VideoRecord(
    │        id=video_id,
    │        sha256_hash="",           ← may not have been computed yet
    │        status=QUARANTINED,
    │        error_message=reason,
    │        storage_bucket="quarantine-videos"
    │    ))
    │    db.flush()
    │
    └─── _fire_quarantine_webhook(video_id, filename, reason)
             [if QUARANTINE_WEBHOOK_URL is set]
             aiohttp.ClientSession.post(url, json={...}, timeout=5s)

    ▼
raise IngestionError("File validation failed: ...", 422)
```

**Why insert a `VideoRecord` even for quarantined files?** Audit trail. An operator can query `WHERE status = 'QUARANTINED'` to see all rejected uploads, when they came in, and why. Without this record, there's no visibility into what was rejected and why.

**Why `sha256_hash=""` for quarantined files?** If the file failed MIME detection before the hash was computed, the hash is unknown. An empty string is used rather than `NULL` because the column has a UNIQUE constraint — two quarantined files with `NULL` would violate uniqueness in some DB configurations. An empty string is a known-bad value that makes the uniqueness constraint behave predictably.

**Why `application/octet-stream` in the quarantine upload?** The file failed MIME validation. Storing it as `video/mp4` would be a lie. `octet-stream` is the generic "unknown binary" MIME type, honest about what we know.

---

### Validation failure vs quarantine distinction

Not all validation failures reach `_quarantine()`. The extension check fails _before_ the file is even stored:

```
Extension not in ALLOWED_EXTENSIONS
    → ValidationResult(is_valid=False)
    → _quarantine() IS called (the bytes are still stored for audit)
    → raise IngestionError(422)

MIME type not in ALLOWED_MIME_TYPES
    → ValidationResult(is_valid=False)
    → _quarantine() IS called
    → raise IngestionError(422)

FFprobe returns None (corrupt container)
    → ValidationResult(is_valid=False)
    → _quarantine() IS called
    → raise IngestionError(422)
```

All three invalid cases quarantine the file. The only validation that does NOT quarantine is the size check — it happens in `routes.py` before bytes are passed to the service:

```python
if len(data) > settings.MAX_UPLOAD_SIZE_BYTES:
    raise HTTPException(413)
    # No quarantine — we haven't read the full file, just detected it's too big
```

This is the right design: an oversized file is not "corrupt", it's just too big. There's no reason to store it.

---

## Cross-Cutting Concerns

### Idempotency

Every pipeline step is designed to be safe to run twice. This matters because RabbitMQ guarantees at-least-once delivery — a message might arrive twice if the consumer crashes after processing but before acknowledging.

- **Ingestion deduplication:** SHA-256 check means re-uploading the same file is always safe. Returns the existing record.
- **MinIO `put_object`:** If the same object is uploaded twice, MinIO overwrites it (same result, no error).
- **PostgreSQL INSERT:** The `UNIQUE` constraint on `sha256_hash` would prevent a duplicate row. In `_run_pipeline()`, the dedup check happens before the INSERT, so the UNIQUE constraint is a safety net, not the primary guard.
- **Future: embedding service:** Upserting into Qdrant is idempotent — if you upsert the same point twice, the second upsert overwrites the first with identical data.

### Async everywhere

The entire service is async (using Python's `asyncio`). This means:

```
One OS thread handles many requests concurrently.
While waiting for PostgreSQL to respond, the same thread handles another request.
While waiting for MinIO to accept bytes, another request's validation runs.
```

The alternative (synchronous + thread-per-request) would require one thread per concurrent request. At 100 concurrent camera streams, that's 100 threads, each with ~8 MB stack memory = ~800 MB RAM just for thread stacks. Async handles the same load with one thread and negligible overhead.

The exception: MinIO SDK calls are synchronous. They're wrapped in `loop.run_in_executor()`, which runs them in a thread pool. The thread pool (default: min(32, cpu_count + 4) threads) is managed by Python's `ThreadPoolExecutor`. So MinIO calls do use threads — but only for the I/O, not for the request handling logic.

### Configuration hierarchy

```
Environment variables  (highest priority)
    ↑ override
.env file
    ↑ override
Default values in BaseServiceSettings / Settings
    (lowest priority, always present as fallback)
```

`pydantic-settings` reads in this order. In production (Kubernetes), you set environment variables directly. In local development, you use `.env`. The defaults in the Python class ensure the service starts with sensible values even if nothing is configured.

### The `@lru_cache` pattern on settings

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`lru_cache` with no arguments caches the first return value forever. `Settings()` is expensive: it reads files, parses environment variables, runs Pydantic validation. Calling it 1000 times per second (once per request) would be wasteful. With `@lru_cache`, it's called once at startup.

**In tests:** Tests that need to override settings call `get_settings.cache_clear()` then mock environment variables. This is the standard pattern for testing settings-dependent code without restarting the process.

---

## Final Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Client / Dashboard                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP REST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    services/ingestion/api/routes.py                     │
│   POST /videos  POST /videos/rtsp  POST /videos/fs  GET /{id}/status   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ Python function calls
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              services/ingestion/services/ingestion.py                   │
│                     IngestionService._run_pipeline()                    │
│                                                                         │
│  validate → dedup → upload → db insert → publish event → respond       │
└────┬──────────────┬────────────────┬────────────────────────────────────┘
     │              │                │
     ▼              ▼                ▼
┌─────────┐  ┌──────────┐  ┌────────────────┐
│validator│  │ storage  │  │    queue       │
│.py      │  │ .py      │  │    .py         │
│         │  │          │  │                │
│ffprobe  │  │ MinIO    │  │ RabbitMQ       │
│libmagic │  │ put_obj  │  │ publish event  │
│sha256   │  │ quarant. │  │                │
└─────────┘  └──────────┘  └────────────────┘
     │              │                │
     ▼              ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         shared/ library                                 │
│                                                                         │
│  config/base.py    models/          events/          queue/             │
│  db.py             video.py         VideoIngested    publisher.py       │
│  logging.py        detection.py     FramesExtracted  consumer.py       │
│  storage/          embedding.py     DetectionDone                       │
│    client.py                        EmbedReady                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌───────────┐  ┌──────────────┐
   │ PostgreSQL  │  │   MinIO   │  │  RabbitMQ    │
   │             │  │           │  │              │
   │video_records│  │raw-videos │  │video.events  │
   │detection_   │  │quarantine │  │  exchange    │
   │  results    │  │  -videos  │  │              │
   │embedding_   │  │           │  │video.ingested│
   │  records    │  │           │  │  → preproc   │
   └─────────────┘  └───────────┘  └──────────────┘

   ┌─────────────┐
   │    Redis    │
   │  (planned)  │
   │search cache │
   │rate limits  │
   │preview URLs │
   └─────────────┘
```