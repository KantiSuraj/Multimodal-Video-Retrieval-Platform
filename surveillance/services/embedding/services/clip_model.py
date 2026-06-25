"""CLIP wrapper — the only file allowed to know about model weights,
devices, tensor shapes, or normalisation math.

Concurrency note: a single CLIPEmbedder instance backs one loaded model on
one device. PyTorch inference is not guaranteed safe under unsynchronized
concurrent calls from multiple threads against one model instance/CUDA
context. Since embedding's prefetch can hand a worker several unacked
messages at once, every call to embed_image() is serialised through an
internal semaphore — this is the model's own concurrency guard, not the
orchestrator's, so it holds regardless of how many call sites exist.

Vector normalisation: the architecture requires embeddings normalised to
unit L2 norm before they're handed to the orchestrator. That happens here,
not in services/embedding.py — the orchestrator never touches a tensor or
a numpy array.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from PIL import Image

from services.embedding.core.config import Settings
from services.embedding.core.logging import get_logger
from services.embedding.models.schemas import EmbeddingError, EmbeddingStage

logger = get_logger(__name__)


@dataclass
class _LoadedModel:
    processor: object
    model: object
    device: str


class CLIPEmbedder:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._loaded: _LoadedModel | None = None
        # Serialises all inference calls against the one model instance.
        self._inference_lock = asyncio.Semaphore(1)

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = self._settings.CLIP_MODEL_NAME
        device = self._settings.CLIP_DEVICE
        logger.info("clip_loading", model_name=model_name, device=device)

        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        self._loaded = _LoadedModel(processor=processor, model=model, device=device)
        logger.info("clip_loaded", model_name=model_name)

    async def embed_image(self, local_image_path: str) -> list[float]:
        if self._loaded is None:
            raise EmbeddingError(
                message="CLIP model not loaded",
                stage=EmbeddingStage.MODEL_INFERENCE,
                recoverable=False,
            )

        loop = asyncio.get_running_loop()
        async with self._inference_lock:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self._run_sync, local_image_path),
                    timeout=self._settings.EMBEDDING_INFERENCE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise EmbeddingError(
                    message=f"CLIP inference timed out on {local_image_path}",
                    stage=EmbeddingStage.MODEL_INFERENCE,
                    recoverable=True,
                ) from exc
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    message=f"CLIP inference failed: {exc}",
                    stage=EmbeddingStage.MODEL_INFERENCE,
                    recoverable=False,
                ) from exc

    def _run_sync(self, local_image_path: str) -> list[float]:
        import torch

        assert self._loaded is not None
        processor, model, device = (
            self._loaded.processor,
            self._loaded.model,
            self._loaded.device,
        )

        image = Image.open(local_image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            # Do NOT use model.get_image_features() — in this transformers
            # version it returns last_hidden_state (1, seq_len, hidden_dim)
            # instead of the projected pooled output (1, projection_dim).
            #
            # Spell out the three steps that get_image_features() is
            # supposed to perform internally:
            #
            #   1. vision_model → BaseModelOutputWithPooling
            #      .last_hidden_state : (1, seq_len, hidden_dim)  ← NOT this
            #      .pooler_output     : (1, hidden_dim)            ← this
            #
            #   2. visual_projection(pooler_output)
            #      → (1, projection_dim)  e.g. 512 for base/32
            #
            #   3. [0] collapses the batch dimension → (projection_dim,)
            #
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output          # (1, hidden_dim)
            projected = model.visual_projection(pooled)    # (1, projection_dim)

        vector = projected[0].cpu().numpy().astype(np.float64)

        if vector.ndim != 1:
            raise EmbeddingError(
                message=(
                    f"Expected 1-D vector after visual_projection()[0], "
                    f"got shape {vector.shape}."
                ),
                stage=EmbeddingStage.MODEL_INFERENCE,
                recoverable=False,
            )

        vector = self._normalize(vector)
        return vector.tolist()


    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Unit-L2-normalise a vector. Pulled out as a pure, torch-free
        function so it can be unit tested without a loaded model or GPU."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
