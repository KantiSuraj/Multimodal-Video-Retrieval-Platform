FROM python:3.11-slim

# ffmpeg/ffprobe are required by transcoder.py, frame_extractor.py, and
# clip_generator.py. libgl1/libglib2.0-0 are OpenCV's runtime dependencies
# for headless image decode (cv2.imread/imwrite, CLAHE, histogram compare).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/preprocessing/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

COPY shared ./shared
COPY services/preprocessing ./services/preprocessing

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.preprocessing.main:app", "--host", "0.0.0.0", "--port", "8000"]
