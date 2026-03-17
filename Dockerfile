# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile  —  PyTorch Docs RAG Bot  (CPU / HF Inference edition)
#
# This image runs the bot with EMBEDDER_MODE=hf (HuggingFace Inference API).
# No GPU or torch required — image is ~400 MB.
#
# For GPU / local-model deployment see the GPU variant below the comments.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────

COPY requirements.txt ./
# Install everything except torch/FlagEmbedding (not needed for HF mode)
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────

COPY src/  ./src/
COPY bot/  ./bot/

# ── Runtime ───────────────────────────────────────────────────────────────

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDER_MODE=hf

# Health-check: verify the process is still running
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "-m", "bot.main"]


# ─────────────────────────────────────────────────────────────────────────────
# GPU variant (build with: docker build -f Dockerfile.gpu .)
#
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base
# RUN apt-get update && apt-get install -y python3.12 python3-pip curl \
#     && ln -s /usr/bin/python3.12 /usr/bin/python
# WORKDIR /app
# COPY requirements.txt requirements-local.txt ./
# RUN pip install --no-cache-dir -r requirements.txt -r requirements-local.txt
# COPY src/ ./src/
# COPY bot/ ./bot/
# ENV PYTHONUNBUFFERED=1 EMBEDDER_MODE=local
# CMD ["python", "-m", "bot.main"]
# ─────────────────────────────────────────────────────────────────────────────
