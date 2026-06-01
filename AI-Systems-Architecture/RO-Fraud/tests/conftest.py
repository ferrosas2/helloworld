"""
Shared pytest fixtures and test environment setup.

Sets dummy GCP environment variables BEFORE any application module is imported,
so `src.config.Settings` validates successfully without real credentials.
"""
import os

# Populate required settings with harmless test values prior to imports.
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_REGION", "us-central1")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")
os.environ.setdefault("VERTEX_INDEX_ID", "test-index")
os.environ.setdefault("VERTEX_ENDPOINT_ID", "test-endpoint")
