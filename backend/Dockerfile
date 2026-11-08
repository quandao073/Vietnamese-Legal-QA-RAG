# syntax=docker/dockerfile:1.7
# CPU by default. GPU: docker-compose.gpu.yml sets BASE_IMAGE to a pytorch/cuda image and
# TORCH_INDEX="" so torch is taken from the base image instead of the CPU wheel index.
ARG BASE_IMAGE=python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# ---------- builder ----------
FROM ${BASE_IMAGE} AS builder
ARG TORCH_INDEX
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Dependencies first (cached layer). The lock already pins the CPU torch wheels; when
# TORCH_INDEX is empty (GPU image) we drop torch from the sync and reuse the base image's torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_INDEX" ]; then \
      uv sync --frozen --no-dev --no-install-project; \
    else \
      uv export --frozen --no-dev --no-hashes --no-emit-project | grep -viE '^(torch|nvidia-|triton)' > /tmp/req.txt \
      && uv venv /opt/venv --system-site-packages && uv pip install --python /opt/venv -r /tmp/req.txt; \
    fi
COPY legalrag ./legalrag
COPY alembic ./alembic
COPY alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_INDEX" ]; then uv sync --frozen --no-dev; else uv pip install --python /opt/venv --no-deps -e .; fi

# ---------- runtime ----------
FROM ${BASE_IMAGE} AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
    PATH="/opt/venv/bin:$PATH" HF_HOME=/data/models DATA_DIR=/data \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app && mkdir -p /data /app/docs && chown -R app:app /data /app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/legalrag ./legalrag
COPY --from=builder /app/alembic ./alembic
COPY --from=builder /app/alembic.ini ./
USER app
EXPOSE 8000
CMD ["uvicorn", "legalrag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
