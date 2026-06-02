import os
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        GCP_PROJECT_ID: str = Field(..., description="Google Cloud Project ID")
        GCP_REGION: str = Field(default="us-central1", description="Google Cloud Region")
        GCS_BUCKET_NAME: str = Field(..., description="Google Cloud Storage Bucket Name")
        VERTEX_INDEX_ID: str = Field(..., description="Vertex AI Vector Search Index ID")
        VERTEX_ENDPOINT_ID: str = Field(..., description="Vertex AI Vector Search Index Endpoint ID")
        GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Vertex AI Gemini model name")

        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "extra": "ignore"
        }
except ImportError:
    # Try importing directly from pydantic (Pydantic v1 fallback)
    from pydantic import BaseSettings
    class Settings(BaseSettings):
        GCP_PROJECT_ID: str = Field(..., description="Google Cloud Project ID")
        GCP_REGION: str = Field(default="us-central1", description="Google Cloud Region")
        GCS_BUCKET_NAME: str = Field(..., description="Google Cloud Storage Bucket Name")
        VERTEX_INDEX_ID: str = Field(..., description="Vertex AI Vector Search Index ID")
        VERTEX_ENDPOINT_ID: str = Field(..., description="Vertex AI Vector Search Index Endpoint ID")
        GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Vertex AI Gemini model name")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"
settings = Settings()
