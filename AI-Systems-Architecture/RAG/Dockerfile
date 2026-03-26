# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables for non-interactive installs and unbuffered Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies (required for FAISS and other build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ./src ./src
COPY ./api ./api

# Set ownership of the application to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the designated port
EXPOSE 8080

# Run Uvicorn serving the FastAPI app
ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]