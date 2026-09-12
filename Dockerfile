# Self-hosted ateflow: the same static app as the public demo (the engine runs in
# the browser), served by the FastAPI server, which also exposes the HTTP API.

# Stage 1: the synthetic example datasets
FROM python:3.12-slim AS examples
WORKDIR /build
RUN pip install --no-cache-dir numpy pandas
COPY examples/ examples/
RUN python examples/make_data.py

# Stage 2: the frontend, with the Python engine and examples copied in for the browser
FROM node:24-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci --no-fund --no-audit
COPY ateflow/ ateflow/
COPY --from=examples /build/examples/ examples/
COPY frontend/ frontend/
RUN cd frontend && npm run build

# Stage 3: the app
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY ateflow/ ateflow/
COPY --from=examples /build/examples/ examples/
RUN pip install --no-cache-dir ".[server]"
COPY --from=frontend /build/ateflow/static ateflow/static

# Analyses saved through the HTTP API live here: mount a volume to keep them.
ENV ATEFLOW_DB=/data/ateflow.db
VOLUME /data

EXPOSE 8080
CMD uvicorn ateflow.server:app --host 0.0.0.0 --port ${PORT:-8080}
