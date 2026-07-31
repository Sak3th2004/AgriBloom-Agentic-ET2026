# AgriBloom V2 backend — deployable to HuggingFace Spaces (Docker SDK, free
# tier) or any container host. CPU-only (free hosting has no GPU); the code
# already degrades gracefully to CPU everywhere (see get_device() /
# DINOv2KNN / DenseRetriever), so no code changes are needed to run here.
#
# Build from the REPO ROOT (not this directory) so the whole codebase is in
# context:
#   docker build -f deployment/Dockerfile -t agribloom-backend .
#   docker run -p 7860:7860 --env-file .env agribloom-backend
#
# For HuggingFace Spaces (Docker SDK): the Space's git repo root must contain
# this Dockerfile — see deployment/README.md for the exact push steps.

FROM python:3.11-slim

# ffmpeg: pydub/voice processing. libsndfile1: soundfile. libgl1: opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY deployment/requirements-cpu.txt ./requirements-cpu.txt
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Now the application code.
COPY . .

# HF Spaces containers run as a non-root user; create one and own /app,
# including the runtime output dirs the backend writes to (voice mp3s, PDFs).
RUN useradd -m -u 1000 agribloom \
    && mkdir -p models/outputs \
    && chown -R agribloom:agribloom /app
USER agribloom

ENV PYTHONUNBUFFERED=1 \
    ENVIRONMENT=prod \
    PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
