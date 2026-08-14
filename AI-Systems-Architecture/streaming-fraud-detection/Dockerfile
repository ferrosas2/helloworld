# =============================================================================
# Multi-Stage Dockerfile: Streaming Fraud Detection API
# =============================================================================
# Stage 1: Build (install dependencies, compile if needed)
# Stage 2: Runtime (minimal image with only runtime deps)
#
# Security best practices:
#   - Non-root user execution
#   - No cache pip install (reduce image size)
#   - Multi-stage build (no build tools in final image)
#   - Pinned base image version
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Build argument for version tracking
ARG BUILD_SHA=unknown
ENV BUILD_SHA=${BUILD_SHA}

# Security: create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/
COPY pipeline/ ./pipeline/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Cloud Run expects port 8080
EXPOSE 8080

# Environment defaults (overridden by Cloud Run env vars)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check for local Docker testing
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the FastAPI application via Uvicorn
ENTRYPOINT ["python", "-m", "uvicorn", "api.main:app", \
            "--host", "0.0.0.0", \
            "--port", "8080", \
            "--workers", "1", \
            "--log-level", "info"]
