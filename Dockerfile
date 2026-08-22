# The service image. CPU-only and ONNX by default: the model is 163 MB of
# int8 weights and a CPU box serves it at single-digit milliseconds, so paying
# for a GPU to run inference here would be paying for idle silicon.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools are needed for some wheels and are dropped from the final layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# The CPU torch wheel on purpose: the CUDA build is 2.5 GB and this image
# never touches a GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -e .

COPY config.yaml ./

# The model is mounted rather than baked in, so the image does not need
# rebuilding for a retrain and does not carry model weights into a registry.
ENV MODEL_DIR=/models/current \
    BACKEND=onnx

# Runs as a non-root user; the service needs no privileges and writes only to
# its own log directory.
RUN useradd --create-home --uid 10001 serve \
    && mkdir -p /app/logs && chown -R serve:serve /app
USER serve

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "arabic_nlp.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
