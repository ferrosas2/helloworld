"""
Application Configuration (Pydantic BaseSettings).

Environment-based configuration for all GCP service connections.
Secrets and resource IDs are never committed to source control —
they flow in via environment variables set by Terraform/Cloud Run.

Usage:
    from src.config import settings
    print(settings.GCP_PROJECT_ID)
"""

from pydantic import Field

try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        """Application settings loaded from environment variables or .env file."""

        # GCP Core
        GCP_PROJECT_ID: str = Field(..., description="Google Cloud Project ID")
        GCP_REGION: str = Field(default="us-central1", description="Google Cloud Region")

        # Pub/Sub
        PUBSUB_TOPIC_TRANSACTIONS: str = Field(
            default="pos-transactions",
            description="Pub/Sub topic for incoming POS transaction events"
        )
        PUBSUB_SUBSCRIPTION_TRANSACTIONS: str = Field(
            default="pos-transactions-sub",
            description="Pub/Sub subscription for the Dataflow pipeline"
        )
        PUBSUB_TOPIC_FLAGGED: str = Field(
            default="flagged-transactions",
            description="Pub/Sub topic for high-risk flagged transactions"
        )

        # BigQuery
        BQ_DATASET: str = Field(
            default="fraud_detection",
            description="BigQuery dataset name for fraud detection tables"
        )
        BQ_TABLE_TRANSACTIONS: str = Field(
            default="enriched_transactions",
            description="BigQuery table for enriched scored transactions"
        )

        # Vertex AI Online Prediction
        VERTEX_PREDICTION_ENDPOINT_ID: str = Field(
            ..., description="Vertex AI endpoint ID for the fraud classifier model"
        )
        VERTEX_SCORING_TIMEOUT: float = Field(
            default=5.0, description="Timeout in seconds for Vertex AI prediction calls"
        )

        # Vertex AI Vector Search (for deep analysis RAG path)
        VERTEX_INDEX_ID: str = Field(
            default="", description="Vertex AI Vector Search Index ID"
        )
        VERTEX_ENDPOINT_ID: str = Field(
            default="", description="Vertex AI Vector Search Index Endpoint ID"
        )

        # Cloud Storage
        GCS_BUCKET_NAME: str = Field(
            ..., description="GCS bucket for embeddings, documents, and model artifacts"
        )

        # LLM Configuration (Deep Analysis Path)
        GEMINI_MODEL: str = Field(
            default="gemini-2.5-flash",
            description="Vertex AI Gemini model for explainable risk analysis"
        )
        LLM_TEMPERATURE: float = Field(
            default=0.0,
            description="LLM temperature (0.0 = deterministic, cost-effective)"
        )
        LLM_MAX_OUTPUT_TOKENS: int = Field(
            default=1024,
            description="Maximum output tokens for risk explanation generation"
        )

        # Scoring Thresholds
        HIGH_RISK_THRESHOLD: float = Field(
            default=0.7,
            description="Score threshold for flagging high-risk transactions"
        )
        MEDIUM_RISK_THRESHOLD: float = Field(
            default=0.3,
            description="Score threshold for medium-risk review queue"
        )

        # Dataflow Pipeline
        DATAFLOW_TEMP_LOCATION: str = Field(
            default="", description="GCS path for Dataflow temp files"
        )
        DATAFLOW_MAX_WORKERS: int = Field(
            default=20, description="Maximum Dataflow worker count for autoscaling"
        )
        VELOCITY_WINDOW_SECONDS: int = Field(
            default=300, description="Sliding window size for velocity features (5 min)"
        )

        # Service Configuration
        SERVICE_NAME: str = Field(
            default="streaming-fraud-detection",
            description="Service name for logging and monitoring"
        )
        LOG_LEVEL: str = Field(
            default="INFO",
            description="Logging level (DEBUG, INFO, WARNING, ERROR)"
        )

        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "extra": "ignore",
        }

except ImportError:
    # Pydantic v1 fallback
    from pydantic import BaseSettings

    class Settings(BaseSettings):
        GCP_PROJECT_ID: str = Field(..., description="Google Cloud Project ID")
        GCP_REGION: str = Field(default="us-central1")
        PUBSUB_TOPIC_TRANSACTIONS: str = Field(default="pos-transactions")
        PUBSUB_SUBSCRIPTION_TRANSACTIONS: str = Field(default="pos-transactions-sub")
        PUBSUB_TOPIC_FLAGGED: str = Field(default="flagged-transactions")
        BQ_DATASET: str = Field(default="fraud_detection")
        BQ_TABLE_TRANSACTIONS: str = Field(default="enriched_transactions")
        VERTEX_PREDICTION_ENDPOINT_ID: str = Field(...)
        VERTEX_SCORING_TIMEOUT: float = Field(default=5.0)
        VERTEX_INDEX_ID: str = Field(default="")
        VERTEX_ENDPOINT_ID: str = Field(default="")
        GCS_BUCKET_NAME: str = Field(...)
        GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
        LLM_TEMPERATURE: float = Field(default=0.0)
        LLM_MAX_OUTPUT_TOKENS: int = Field(default=1024)
        HIGH_RISK_THRESHOLD: float = Field(default=0.7)
        MEDIUM_RISK_THRESHOLD: float = Field(default=0.3)
        DATAFLOW_TEMP_LOCATION: str = Field(default="")
        DATAFLOW_MAX_WORKERS: int = Field(default=20)
        VELOCITY_WINDOW_SECONDS: int = Field(default=300)
        SERVICE_NAME: str = Field(default="streaming-fraud-detection")
        LOG_LEVEL: str = Field(default="INFO")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"


settings = Settings()
