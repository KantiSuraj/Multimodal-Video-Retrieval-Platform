"""HTTP API for the Search Service.

Implements the three endpoints specified in the architecture:

    POST /api/v1/search/text        — text-to-video retrieval
    POST /api/v1/search/image       — image-to-video retrieval
    GET  /api/v1/search/history     — paginated search history

Routes are framework-specific (FastAPI).  All business logic lives in
SearchService — routes only translate HTTP to Python and map SearchErrors
to HTTP status codes.

Pattern mirrors ingestion/api/routes.py: thin HTTP boundary, no
retrieval logic.
"""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from services.search.models.schemas import MetadataFilters, SearchError
from services.search.services.search import SearchService

router = APIRouter(prefix="/api/v1/search", tags=["search"])


# ── Request / response schemas ────────────────────────────────────────────────


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    camera_ids: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None
    labels: list[str] = Field(default_factory=list)
    page_size: int | None = Field(default=None, ge=1, le=100)
    page_token: str | None = None


class SearchResultItem(BaseModel):
    video_id: str
    clip_id: str
    camera_name: str | None
    camera_location: str | None
    timestamp_start: int
    timestamp_end: int
    similarity_score: float
    thumbnail_url: str | None
    detected_labels: list[str]
    video_start_epoch: int | None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    next_page_token: str | None
    total_returned: int


class HistoryEntry(BaseModel):
    id: str
    query_type: str
    query_text: str | None
    filters: dict
    result_count: int
    created_at: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
    next_cursor: int | None


# ── Dependency ────────────────────────────────────────────────────────────────

# search_service is set by main.py after construction.  This avoids
# creating a module-level singleton and keeps dependency injection
# explicit — the same pattern used by the Embedding Service.
_search_service: SearchService | None = None


def set_search_service(service: SearchService) -> None:
    global _search_service
    _search_service = service


def get_search_service() -> SearchService:
    if _search_service is None:
        raise RuntimeError("SearchService not initialised — call set_search_service() first")
    return _search_service


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/text", response_model=SearchResponse)
async def text_search(
    request: TextSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute text-to-video retrieval."""
    filters = MetadataFilters(
        camera_ids=request.camera_ids,
        locations=request.locations,
        start_ms=request.start_ms,
        end_ms=request.end_ms,
        labels=request.labels,
    )

    try:
        results, next_token = await service.execute_text_search(
            query_text=request.query,
            filters=filters,
            page_size=request.page_size,
            page_token=request.page_token,
        )
    except SearchError as exc:
        status_code = 503 if exc.recoverable else 422
        raise HTTPException(status_code=status_code, detail=exc.message) from exc

    return SearchResponse(
        results=[_to_item(r) for r in results],
        next_page_token=next_token,
        total_returned=len(results),
    )


@router.post("/image", response_model=SearchResponse)
async def image_search(
    images: list[UploadFile] = File(...),
    camera_ids: str = Form(default="[]"),
    locations: str = Form(default="[]"),
    start_ms: int | None = Form(default=None),
    end_ms: int | None = Form(default=None),
    labels: str = Form(default="[]"),
    page_size: int | None = Form(default=None),
    page_token: str | None = Form(default=None),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute image-to-video retrieval.

    Accepts one or more image files via multipart/form-data.
    Multiple images are averaged before ANN retrieval.
    """
    if not images:
        raise HTTPException(status_code=422, detail="At least one image file is required")

    try:
        camera_ids_list: list[str] = json.loads(camera_ids)
        locations_list: list[str] = json.loads(locations)
        labels_list: list[str] = json.loads(labels)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in filter fields: {exc}") from exc

    filters = MetadataFilters(
        camera_ids=camera_ids_list,
        locations=locations_list,
        start_ms=start_ms,
        end_ms=end_ms,
        labels=labels_list,
    )

    image_bytes_list: list[bytes] = []
    for upload in images:
        try:
            data = await upload.read()
            if not data:
                raise HTTPException(status_code=422, detail="Uploaded image file is empty")
            image_bytes_list.append(data)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Failed to read uploaded file: {exc}") from exc

    try:
        results, next_token = await service.execute_image_search(
            image_bytes_list=image_bytes_list,
            filters=filters,
            page_size=page_size,
            page_token=page_token,
        )
    except SearchError as exc:
        status_code = 503 if exc.recoverable else 422
        raise HTTPException(status_code=status_code, detail=exc.message) from exc

    return SearchResponse(
        results=[_to_item(r) for r in results],
        next_page_token=next_token,
        total_returned=len(results),
    )


@router.get("/history", response_model=HistoryResponse)
async def search_history(
    query_type: str | None = Query(default=None, description="Filter by query type: text | image | multi_image"),
    cursor: int = Query(default=0, ge=0),
    page_size: int = Query(default=20, ge=1, le=100),
    service: SearchService = Depends(get_search_service),
) -> HistoryResponse:
    """Return paginated search history with optional query_type filter."""
    entries, next_cursor = await service.get_search_history(
        query_type=query_type,
        cursor=cursor,
        page_size=page_size,
    )
    return HistoryResponse(
        entries=[HistoryEntry(**e) for e in entries],
        next_cursor=next_cursor,
    )


# ── Serialisation helper ──────────────────────────────────────────────────────


def _to_item(r) -> SearchResultItem:
    return SearchResultItem(
        video_id=r.video_id,
        clip_id=r.clip_id,
        camera_name=r.camera_name,
        camera_location=r.camera_location,
        timestamp_start=r.timestamp_start,
        timestamp_end=r.timestamp_end,
        similarity_score=r.similarity_score,
        thumbnail_url=r.thumbnail_url,
        detected_labels=r.detected_labels,
        video_start_epoch=r.video_start_epoch,
    )
