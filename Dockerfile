# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build tuned for both local docker validation and Hugging Face
# Docker Spaces. The app itself still serves OpenEnv on port 8000, but the
# container exposes the Space-facing port 7860 and forwards traffic there.

FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    rm -rf /var/lib/apt/lists/*

# Install only the server/runtime dependencies so the deployment image stays
# lean and does not pull the full training stack.
COPY server/requirements.txt /app/server-requirements.txt

RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -r /app/server-requirements.txt

COPY . /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Validate the environment during build so bad deploys fail early.
RUN cd /app/env && openenv validate


FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends socat curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health || exit 1

# Hugging Face Docker Spaces expects the public app port to be the README
# `app_port` (7860 here). OpenEnv itself stays on 8000 internally.
CMD ["sh", "-c", "SPACE_PORT=${PORT:-7860}; cd /app/env && socat TCP-LISTEN:${SPACE_PORT},fork,reuseaddr TCP:127.0.0.1:8000 & exec uvicorn server.app:app --host 0.0.0.0 --port 8000"]
