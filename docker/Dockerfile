FROM python:3.12-slim

# Add non-root user
RUN groupadd -g 1000 developer && \
    useradd -u 1000 -g developer -m -s /bin/bash developer

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    cdo \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    git \
    build-essential \
    libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv and move it to a global location
# Install uv and move it to a global location
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -Dm755 /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace/dss-pollution-extraction

# Switch to developer user
USER developer
