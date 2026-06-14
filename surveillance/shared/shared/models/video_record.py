"""
Convenience re-export so other services can import from either path:

    from shared.models.video        import VideoRecord   # canonical
    from shared.models.video_record import VideoRecord   # alias (same object)

The actual definition lives in shared/models/video.py.
This file exists so the models folder has a predictable one-file-per-model
layout that matches detection_result.py and embedding_record.py.
"""
from shared.models.video import Base, VideoRecord, VideoStatus  # noqa: F401

__all__ = ["Base", "VideoRecord", "VideoStatus"]