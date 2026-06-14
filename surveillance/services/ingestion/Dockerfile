FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy workspace definition + both packages
COPY pyproject.toml .
COPY shared/ shared/
COPY services/ingestion/ services/ingestion/

# Install shared (editable) then ingestion service
RUN uv pip install --system -e ./shared -e ./services/ingestion

RUN mkdir -p /tmp/video_watch

EXPOSE 8000

CMD ["uvicorn", "services.ingestion.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-config", "null"]