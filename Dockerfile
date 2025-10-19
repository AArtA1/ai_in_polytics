FROM python:3.13-slim AS base

# Environment setup
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy 

# Install uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first for better caching
COPY pyproject.toml uv.lock .python-version ./

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install dependencies using uv (honor lockfile, no dev deps)
RUN uv sync --locked --no-dev

# Copy application source
COPY src ./src

COPY .env ./.env

# Copy Qdrant database (needed for RAG retrieval)
COPY qdrant_store ./qdrant_store

EXPOSE 8000

# Run the app via uv + uvicorn
CMD ["uv", "run", "uvicorn", "src.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


