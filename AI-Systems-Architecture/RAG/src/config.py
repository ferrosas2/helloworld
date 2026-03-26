"""
Configuration settings management for the RAG architecture.
Utilizes Pydantic BaseSettings for environment variable validation and parsing.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.
    """
    project_id: str = Field(..., description="Google Cloud Project ID")
    region: str = Field(default="us-central1", description="Google Cloud Region")
    llm_model_name: str = Field(
        default="gemini-1.5-pro", 
        description="Vertex AI Gemini model name for text generation"
    )
    embedding_model: str = Field(
        default="textembedding-gecko", 
        description="Vertex AI embedding model name"
    )
    vector_store_path: str = Field(
        default="./vector_store/faiss_index", 
        description="Local path to persist the FAISS index"
    )
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()