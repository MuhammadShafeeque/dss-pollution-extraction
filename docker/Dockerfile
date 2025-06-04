FROM python:3.12-slim

# Add non-root user
RUN groupadd -g 1000 developer && useradd -u 1000 -g developer -m -s /bin/bash developer

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
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.cargo/bin/uv /usr/local/bin/uv


# Setup work directory
WORKDIR /workspaces/dss-pollution-extraction
COPY . .

# Set permissions
RUN chown -R developer:developer /workspaces

# Switch to non-root user
USER developer
