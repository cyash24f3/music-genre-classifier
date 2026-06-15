# ─── Music Genre Classifier ──────────────────────────────────────────────────
# Optimised for Hugging Face Spaces (Docker SDK, port 7860)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps for audio processing (libsndfile, ffmpeg) & matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Writable dirs for HF cache & matplotlib
RUN mkdir -p /app/.cache /app/.config && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.config/matplotlib \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
