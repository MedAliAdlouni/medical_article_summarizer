FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --prefer-binary --no-build-isolation -r requirements.txt

# Copy application code
COPY src ./src
COPY scripts ./scripts

# Copy local model cache (if present) to avoid HF downloads at runtime
COPY model ./model

# Skip model warmup during build to keep builds fast; download happens on first use if needed
# To force warmup, build with: --build-arg DO_WARMUP=1
ARG DO_WARMUP=0
RUN if [ "$DO_WARMUP" = "1" ]; then python -m scripts.warmup || true; fi

EXPOSE 8000

CMD ["python", "-m", "src.api.server"]


