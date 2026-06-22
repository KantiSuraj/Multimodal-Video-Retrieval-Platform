"""Grounding DINO wrapper — the only file allowed to know about model
weights, devices, or tensor shapes.

Concurrency note: a single GroundingDINODetector instance backs one loaded
model on one device. PyTorch inference is not guaranteed safe under
unsynchronized concurrent calls from multiple threads against one model
instance/CUDA context. Since detection's prefetch can hand a worker several
unacked messages at once, every call to detect() is serialised through an
internal semaphore — this is the model's own concurrency guard, not the
orchestrator's, so it holds regardless of how many call sites exist.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from PIL import Image

from services.detection.core.config import Settings
from services.detection.core.logging import get_logger
from services.detection.models.schemas import DetectionError, DetectionStage, RawDetection

logger = get_logger(__name__)


@dataclass
class _LoadedModel:
    processor: object
    model: object
    device: str


class GroundingDINODetector:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._loaded: _LoadedModel | None = None
        # Serialises all inference calls against the one model instance.
        self._inference_lock = asyncio.Semaphore(1)

    def load(self) -> None:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        model_name = self._settings.GROUNDING_DINO_MODEL_NAME
        device = self._settings.GROUNDING_DINO_DEVICE

        
        # Validate / normalize device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "cuda_unavailable_falling_back_to_cpu"
            )
            device = "cpu"

        logger.info(
            "grounding_dino_loading",
            model_name=model_name,
            device=device,
        )

        logger.info(
            "grounding_dino_loading",
            model_name=model_name,
            device=device,
        )

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)

        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA requested but unavailable. device={device}"
                )

        model = model.to(device)
        model.eval()

        self._loaded = _LoadedModel(
            processor=processor,
            model=model,
            device=device,
        )

        logger.info(
            "grounding_dino_loaded",
            model_name=model_name,
        )

    async def detect(self, local_image_path: str) -> list[RawDetection]:
        if self._loaded is None:
            raise DetectionError(
                message="Grounding DINO model not loaded",
                stage=DetectionStage.MODEL_INFERENCE,
                recoverable=False,
            )

        loop = asyncio.get_running_loop()
        async with self._inference_lock:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self._run_sync, local_image_path),
                    timeout=self._settings.DETECTION_INFERENCE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise DetectionError(
                    message=f"Grounding DINO inference timed out on {local_image_path}",
                    stage=DetectionStage.MODEL_INFERENCE,
                    recoverable=True,
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise DetectionError(
                    message=f"Grounding DINO inference failed: {exc}",
                    stage=DetectionStage.MODEL_INFERENCE,
                    recoverable=False,
                ) from exc

    def _run_sync(self, local_image_path: str) -> list[RawDetection]:
        import torch

        assert self._loaded is not None
        processor, model, device = (
            self._loaded.processor,
            self._loaded.model,
            self._loaded.device,
        )

        image = Image.open(local_image_path).convert("RGB")
        width, height = image.size

        inputs = processor(
            images=image,
            text=self._settings.GROUNDING_DINO_TEXT_PROMPT,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._settings.GROUNDING_DINO_BOX_THRESHOLD,
            text_threshold=self._settings.GROUNDING_DINO_TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]

        detections: list[RawDetection] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            detections.append(
                RawDetection(
                    label=str(label),
                    confidence=float(score),
                    bbox_x1=max(0.0, x1 / width),
                    bbox_y1=max(0.0, y1 / height),
                    bbox_x2=min(1.0, x2 / width),
                    bbox_y2=min(1.0, y2 / height),
                )
            )
        return detections