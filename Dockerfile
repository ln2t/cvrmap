# CVRmap Docker Image v4.3.0
# Single-stage build for simplicity

FROM python:3.12.11-slim

# Build argument for version
ARG CVRMAP_VERSION=4.3.1

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_TRUSTED_HOST="pypi.org pypi.python.org files.pythonhosted.org"

# Create non-root user for security
RUN groupadd --gid 1000 cvrmap && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash cvrmap

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/cvrmap

# Copy application files
COPY setup.py pyproject.toml ./
COPY cvrmap/ ./cvrmap/
COPY data/ ./data/

# Install CVRmap and dependencies
RUN pip install -U --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org . && \
    apt-get purge -y build-essential gcc g++ gfortran && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create directories for data processing
RUN mkdir -p /data/input /data/output /data/work && \
    chown -R cvrmap:cvrmap /data /opt/cvrmap

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Set default values\n\
INPUT_DIR=${INPUT_DIR:-/data/input}\n\
OUTPUT_DIR=${OUTPUT_DIR:-/data/output}\n\
WORK_DIR=${WORK_DIR:-/data/work}\n\
\n\
# Ensure directories exist and are writable\n\
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$WORK_DIR"\n\
\n\
# If no arguments provided, show help\n\
if [ $# -eq 0 ]; then\n\
    echo "CVRmap Docker Container v${CVRMAP_VERSION}"\n\
    echo ""\n\
    echo "Usage:"\n\
    echo "  docker run --rm -v /path/to/data:/data/input -v /path/to/output:/data/output arovai/cvrmap:${CVRMAP_VERSION} [cvrmap arguments]"\n\
    echo ""\n\
    echo "Example:"\n\
    echo "  docker run --rm -v /home/user/bids:/data/input -v /home/user/output:/data/output arovai/cvrmap:${CVRMAP_VERSION} /data/input /data/output participant --task gas --derivatives fmriprep=/data/input/derivatives/fmriprep"\n\
    echo ""\n\
    echo "Environment variables:"\n\
    echo "  INPUT_DIR   - Input data directory (default: /data/input)"\n\
    echo "  OUTPUT_DIR  - Output directory (default: /data/output)"\n\
    echo "  WORK_DIR    - Working directory (default: /data/work)"\n\
    echo ""\n\
    cvrmap --help\n\
    exit 0\n\
fi\n\
\n\
# Execute cvrmap with provided arguments\n\
exec cvrmap "$@"\n\
' > /usr/local/bin/cvrmap-entrypoint && \
    chmod +x /usr/local/bin/cvrmap-entrypoint

# Switch to non-root user
USER cvrmap

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/cvrmap-entrypoint"]

# Add labels for metadata
LABEL maintainer="CVRMap Development Team" \
      version="${CVRMAP_VERSION}" \
      description="CVRmap - Cerebrovascular Reactivity Mapping Pipeline" \
      org.opencontainers.image.title="CVRmap" \
      org.opencontainers.image.description="A Python CLI application for cerebrovascular reactivity mapping using BIDS-compatible physiological and BOLD fMRI data" \
      org.opencontainers.image.version="${CVRMAP_VERSION}" \
      org.opencontainers.image.authors="CVRMap Development Team" \
      org.opencontainers.image.source="https://github.com/arovai/cvrmap" \
      org.opencontainers.image.documentation="https://cvrmap.readthedocs.io/" \
      org.opencontainers.image.licenses="MIT"
