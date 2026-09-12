# Stage 1: build the frontend into ateflow/static
FROM node:24-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci --no-fund --no-audit
COPY frontend/ frontend/
RUN cd frontend && npm run build

# Stage 2: the app
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY ateflow/ ateflow/
COPY examples/ examples/
RUN pip install --no-cache-dir ".[server]" \
    && python examples/make_data.py
COPY --from=frontend /build/ateflow/static ateflow/static

# Saved analyses live here: mount a volume to keep them across deploys
# (without one the directory is ephemeral and saves last until the next restart).
ENV ATEFLOW_DB=/data/ateflow.db
VOLUME /data

# Some platforms (Render) inject the port to listen on via $PORT.
EXPOSE 8080
CMD uvicorn ateflow.server:app --host 0.0.0.0 --port ${PORT:-8080}
