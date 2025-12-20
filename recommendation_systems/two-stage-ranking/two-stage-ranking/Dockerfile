# Two-Stage Ranking System - Production Container
# 
# This Dockerfile packages the XGBoost Learning-to-Rank model for deployment
# on AWS SageMaker or as a standalone service.
#
# Build: docker build -t two-stage-ranking:latest .
# Run:   docker run -p 8080:8080 two-stage-ranking:latest

FROM python:3.10-slim

# Metadata
LABEL maintainer="github.com/ferrosas2"
LABEL description="XGBoost Learning-to-Rank model for high-scale auction recommendations"
LABEL version="1.0.0"

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/opt/ml/code"

# SageMaker uses these environment variables
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Create model directory (SageMaker standard)
RUN mkdir -p /opt/ml/model /opt/ml/input /opt/ml/output

# Default entrypoint for training
# For inference, SageMaker will override this with the inference handler
ENTRYPOINT ["python", "src/train.py"]

# Default command (can be overridden)
CMD ["--bucket", "ltr-models-frp", "--key", "data/ltr_training_data.csv"]
