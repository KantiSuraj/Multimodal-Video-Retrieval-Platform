"""Redis client for the Search Service.

Provides two distinct caches as specified in the architecture:

1. Query Embedding Cache
   key:  "search:emb:<sha256(query_text | image_bytes_hex)>"
   TTL:  SEARCH_EMBEDDING_CACHE_TTL_SECONDS (default 1 hour)
   value: JSON-serialised list[float] (the embedding vector)

2. Search Result Cache
   key:  "search:result:<sha256(query_text | image_bytes_hex + filter_json)>"
   TTL:  SEARCH_RESULT_CACHE_TTL_SECONDS (default 5 minutes)
   value: JSON-serialised list[HydratedResult]

Redis failures are recoverable — a cache miss is not an error, and a
cache write failure is logged but never re-raised.  The cache is a
performance optimisation, not a correctness requirement.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as redis

from services.search.core.config import Settings
from services.search.core.logging import get_logger
from services.search.models.schemas import HydratedResult

logger = get_logger(__name__)


class SearchCacheClient:
    """Async Redis wrapper for embedding and result caching."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: redis.Redis | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        self._client = redis.from_url(
            self._settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("search_cache_connected", url=self._settings.REDIS_URL)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("search_cache_closed")

    # ── Embedding cache ───────────────────────────────────────────────────────

    async def get_embedding(self, cache_key: str) -> list[float] | None:
        """Return a cached embedding vector or None on miss/error."""
        client = self._client
        if client is None:
            return None
        try:
            raw = await client.get(f"search:emb:{cache_key}")
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_embedding_cache_read_error", reason=str(exc))
            return None

    async def set_embedding(self, cache_key: str, vector: list[float]) -> None:
        """Store an embedding vector; log and swallow Redis errors."""
        client = self._client
        if client is None:
            return
        try:
            await client.setex(
                f"search:emb:{cache_key}",
                self._settings.SEARCH_EMBEDDING_CACHE_TTL_SECONDS,
                json.dumps(vector),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_embedding_cache_write_error", reason=str(exc))

    # ── Result cache ──────────────────────────────────────────────────────────

    async def get_results(self, cache_key: str) -> list[dict] | None:
        """Return cached search results (as dicts) or None on miss/error."""
        client = self._client
        if client is None:
            return None
        try:
            raw = await client.get(f"search:result:{cache_key}")
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_result_cache_read_error", reason=str(exc))
            return None

    async def set_results(self, cache_key: str, results: list[HydratedResult]) -> None:
        """Serialise and store search results; log and swallow Redis errors."""
        client = self._client
        if client is None:
            return
        try:
            payload = json.dumps([_result_to_dict(r) for r in results])
            await client.setex(
                f"search:result:{cache_key}",
                self._settings.SEARCH_RESULT_CACHE_TTL_SECONDS,
                payload,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_result_cache_write_error", reason=str(exc))

    # ── Search history ────────────────────────────────────────────────────────

    async def append_history(self, entry: dict[str, Any], user_key: str = "default") -> None:
        """Append a search history entry.  Keeps the most recent 1000 per user."""
        client = self._client
        if client is None:
            return
        try:
            list_key = f"search:history:{user_key}"
            await client.lpush(list_key, json.dumps(entry))
            await client.ltrim(list_key, 0, 999)  # keep last 1000
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_history_write_error", reason=str(exc))

    async def get_history(
        self,
        user_key: str = "default",
        cursor: int = 0,
        page_size: int = 20,
    ) -> tuple[list[dict], int | None]:
        """Return a page of history entries and the next cursor (or None)."""
        client = self._client
        if client is None:
            return [], None
        try:
            raw_entries = await client.lrange(
                f"search:history:{user_key}", cursor, cursor + page_size - 1
            )
            entries = [json.loads(e) for e in raw_entries]
            next_cursor = cursor + page_size if len(raw_entries) == page_size else None
            return entries, next_cursor
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_history_read_error", reason=str(exc))
            return [], None


# ── Cache key helpers ─────────────────────────────────────────────────────────


def make_text_embedding_key(text: str) -> str:
    """Stable cache key for a text query embedding."""
    return hashlib.sha256(text.encode()).hexdigest()


def make_image_embedding_key(image_bytes: bytes) -> str:
    """Stable cache key for an image query embedding."""
    return hashlib.sha256(image_bytes).hexdigest()


def make_result_cache_key(embedding_key: str, filters_dict: dict[str, Any]) -> str:
    """Stable cache key for a (query + filters) result set."""
    filter_blob = json.dumps(filters_dict, sort_keys=True)
    combined = f"{embedding_key}:{filter_blob}"
    return hashlib.sha256(combined.encode()).hexdigest()


# ── Internal serialisation ────────────────────────────────────────────────────


def _result_to_dict(result: HydratedResult) -> dict[str, Any]:
    return {
        "video_id": result.video_id,
        "clip_id": result.clip_id,
        "camera_name": result.camera_name,
        "camera_location": result.camera_location,
        "timestamp_start": result.timestamp_start,
        "timestamp_end": result.timestamp_end,
        "similarity_score": result.similarity_score,
        "thumbnail_url": result.thumbnail_url,
        "detected_labels": result.detected_labels,
        "video_start_epoch": result.video_start_epoch,
    }


def dict_to_result(d: dict[str, Any]) -> HydratedResult:
    return HydratedResult(
        video_id=d["video_id"],
        clip_id=d["clip_id"],
        camera_name=d.get("camera_name"),
        camera_location=d.get("camera_location"),
        timestamp_start=d["timestamp_start"],
        timestamp_end=d["timestamp_end"],
        similarity_score=d["similarity_score"],
        thumbnail_url=d.get("thumbnail_url"),
        detected_labels=d.get("detected_labels", []),
        video_start_epoch=d.get("video_start_epoch"),
    )
