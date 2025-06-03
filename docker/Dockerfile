# Build stage for compiling dependencies
FROM python:3.12-slim as builder

# Install system build dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_ROOT_USER_ACTION=ignore

# Install dependencies
WORKDIR /install
COPY requirements.txt ./
COPY docker/requirements-docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --retries 10 --timeout 120 pip==25.1.1 && \
    python -m pip wheel --retries 10 --timeout 120 --wheel-dir=/install/wheels -r requirements.txt -r requirements-docker.txt

# Final stage
FROM python:3.12-slim

# Create non-root user
RUN groupadd --gid 1000 developer && \
    useradd --uid 1000 --gid developer --shell /bin/bash --create-home developer

# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cdo \
    curl \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_ROOT_USER_ACTION=ignore

# Copy wheels from builder and install
COPY --from=builder /install/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Set working directory
WORKDIR /workspaces/dss-pollution-extraction

# Switch to non-root user for security
USER developer

# Copy application code
COPY --chown=developer:developer . .
