# Base image with Python 3.13
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Builder stage: install Python dependencies
FROM base AS builder
COPY pyproject.toml uv.lock* ./
RUN pip install --prefix=/install --no-cache-dir uv
RUN uv pip install --prefix=/install --no-cache-dir .

# Runtime stage: copy only what's needed
FROM base AS runtime
COPY --from=builder /install /usr/local
COPY api ./api
COPY mylib ./mylib
COPY templates ./templates
COPY best_model.onnx ./best_model.onnx
COPY labels.json ./labels.json

EXPOSE 8000
CMD ["uvicorn", "api.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]
