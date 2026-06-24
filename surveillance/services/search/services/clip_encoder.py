"""CLIP wrapper for query encoding — the only file allowed to know about
model weights, devices, tensor shapes, or normalisation math.

Search encodes text queries and image queries using the same CLIP model
family used during indexing.  Vectors produced here are guaranteed
compatible with the indexed vectors because both sides use the same
model checkpoint.

Concurrency note: a single CLIPQueryEncoder instance backs one loaded
model on one device.  All encode calls are serialised through a
semaphore — the same guard used by the Embedding Service.

Vector normalisation: embeddings are normalised to unit L2 before they
are handed to the search orchestrator.  The orchestrator never touches
tensors or numpy arrays.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from services.search.core.config import Settings
from services.search.core.logging import get_logger
from services.search.models.schemas import SearchError, SearchStage

logger = get_logger(__name__)


@dataclass
class _LoadedModel:
    processor: object
    model: object
    device: str


class CLIPQueryEncoder:
    """Encodes text and image queries into unit-L2-normalised vectors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loaded: _LoadedModel | None = None
        # Serialises all inference calls — same rationale as CLIPEmbedder.
        self._inference_lock = asyncio.Semaphore(1)

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = self._settings.CLIP_MODEL_NAME
        device = self._settings.CLIP_DEVICE
        logger.info("clip_query_encoder_loading", model_name=model_name, device=device)

        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        self._loaded = _LoadedModel(processor=processor, model=model, device=device)
        logger.info("clip_query_encoder_loaded", model_name=model_name)

    # ── Text encoding ─────────────────────────────────────────────────────────

    async def encode_text(self, text: str) -> list[float]:
        """Encode a text query into a unit-L2-normalised vector."""
        if self._loaded is None:
            raise SearchError(
                message="CLIP model not loaded",
                stage=SearchStage.ENCODE,
                recoverable=False,
            )

        loop = asyncio.get_running_loop()
        async with self._inference_lock:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self._encode_text_sync, text),
                    timeout=self._settings.SEARCH_INFERENCE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise SearchError(
                    message=f"CLIP text encoding timed out for query: {text!r}",
                    stage=SearchStage.ENCODE,
                    recoverable=True,
                ) from exc
            except SearchError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise SearchError(
                    message=f"CLIP text encoding failed: {exc}",
                    stage=SearchStage.ENCODE,
                    recoverable=False,
                ) from exc

    def _encode_text_sync(self, text: str) -> list[float]:
        import torch

        assert self._loaded is not None
        processor, model, device = (
            self._loaded.processor,
            self._loaded.model,
            self._loaded.device,
        )

        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(
            device
        )

        with torch.no_grad():
            features = model.get_text_features(**inputs)

        vector = features[0].cpu().numpy().astype(np.float64)
        return self._normalize(vector).tolist()

    # ── Image encoding ────────────────────────────────────────────────────────

    async def encode_image_bytes(self, image_bytes: bytes) -> list[float]:
        """Encode raw image bytes into a unit-L2-normalised vector."""
        if self._loaded is None:
            raise SearchError(
                message="CLIP model not loaded",
                stage=SearchStage.ENCODE,
                recoverable=False,
            )

        loop = asyncio.get_running_loop()
        async with self._inference_lock:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self._encode_image_sync, image_bytes),
                    timeout=self._settings.SEARCH_INFERENCE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise SearchError(
                    message="CLIP image encoding timed out",
                    stage=SearchStage.ENCODE,
                    recoverable=True,
                ) from exc
            except SearchError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise SearchError(
                    message=f"CLIP image encoding failed: {exc}",
                    stage=SearchStage.ENCODE,
                    recoverable=False,
                ) from exc

    def _encode_image_sync(self, image_bytes: bytes) -> list[float]:
        import torch

        assert self._loaded is not None
        processor, model, device = (
            self._loaded.processor,
            self._loaded.model,
            self._loaded.device,
        )

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise SearchError(
                message=f"Failed to decode image: {exc}",
                stage=SearchStage.ENCODE,
                recoverable=False,
            ) from exc

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        vector = features[0].cpu().numpy().astype(np.float64)
        return self._normalize(vector).tolist()

    # ── Multi-image averaging ─────────────────────────────────────────────────

    @staticmethod
    def average_embeddings(embeddings: list[list[float]]) -> list[float]:
        """Average multiple image embeddings then re-normalise.

        Per the architecture specification:
            query_embedding = mean(image_embeddings)

        Re-normalising after averaging ensures the result is unit-L2,
        matching the norm of indexed vectors.
        """
        if not embeddings:
            raise SearchError(
                message="Cannot average zero embeddings",
                stage=SearchStage.ENCODE,
                recoverable=False,
            )
        arr = np.mean(np.array(embeddings, dtype=np.float64), axis=0)
        return CLIPQueryEncoder._normalize(arr).tolist()

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Unit-L2-normalise.  Pulled out as a pure, torch-free function
        so it can be unit-tested without a loaded model or GPU."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
