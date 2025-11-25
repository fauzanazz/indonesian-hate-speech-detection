# Multi-stage Dockerfile for Indonesian Toxicity Detection API
# Optimized for production deployment with minimal image size

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as builder

WORKDIR /build

# Copy dependency files
COPY pyproject.toml setup.py ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-warn-script-location -r requirements.txt

# Stage 3: Production image
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY pyproject.toml setup.py ./

# Install package in editable mode
RUN pip install -e . --no-deps

# Create logs directory
RUN mkdir -p logs

# Expose API port
EXPOSE 8000

# Run the unified API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

