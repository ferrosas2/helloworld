"""
Pydantic Request/Response Models for the Fraud Detection API.

Defines strict schemas for:
- Incoming transaction analysis requests
- Structured fraud risk responses
- Batch scoring payloads
- Health/readiness probe responses
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator


# =============================================================================
# Request Models
# =============================================================================

class TransactionRequest(BaseModel):
    """Single transaction submitted for real-time fraud analysis."""

    transaction_id: str = Field(
        ..., description="Unique transaction identifier",
        examples=["txn_a1b2c3d4-e5f6-7890"]
    )
    card_token: str = Field(
        ..., description="Tokenized card identifier (never raw PAN)",
        examples=["card_token_abc123def456"]
    )
    merchant_id: str = Field(
        ..., description="Merchant identifier",
        examples=["merchant_m7x9k2"]
    )
    merchant_category: str = Field(
        ..., description="Merchant category code",
        examples=["electronics", "grocery", "jewelry"]
    )
    terminal_id: Optional[str] = Field(
        None, description="POS terminal identifier"
    )
    terminal_type: str = Field(
        default="chip",
        description="Terminal type: chip, contactless, swipe, online",
        examples=["chip", "contactless", "swipe", "online"]
    )
    amount: float = Field(
        ..., gt=0, description="Transaction amount in the specified currency",
        examples=[149.99, 5000.00]
    )
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    region: str = Field(
        default="us-central-1", description="Geographic region of the transaction"
    )
    is_international: bool = Field(
        default=False, description="Whether the transaction is cross-border"
    )
    pin_entered: bool = Field(
        default=True, description="Whether the cardholder entered a PIN"
    )
    recurring: bool = Field(
        default=False, description="Whether this is a recurring/subscription charge"
    )
    timestamp: Optional[str] = Field(
        None, description="ISO 8601 timestamp of the transaction"
    )

    @validator("merchant_category")
    def validate_category(cls, v):
        valid_categories = [
            "grocery", "electronics", "gas_station", "restaurant",
            "online_retail", "jewelry", "travel", "pharmacy",
            "convenience_store", "department_store", "other"
        ]
        if v not in valid_categories:
            v = "other"
        return v


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for bulk scoring."""
    transactions: List[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000,
        description="List of transactions to score (max 1000 per batch)"
    )


class DeepAnalysisRequest(BaseModel):
    """Request for RAG-based deep fraud analysis (high-risk path)."""
    transaction_id: str = Field(..., description="Transaction to analyze")
    transaction_text: str = Field(
        ..., description="Formatted transaction details for context retrieval",
        examples=["$5,000 jewelry purchase, online terminal, no PIN, international, night transaction"]
    )
    fraud_probability_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Initial fraud score from the fast path model"
    )
    context_metadata: Optional[dict] = Field(
        None, description="Additional context (merchant history, card profile)"
    )


# =============================================================================
# Response Models
# =============================================================================

class FraudScoreResponse(BaseModel):
    """Response from real-time fraud scoring (fast path)."""

    transaction_id: str = Field(..., description="Original transaction ID")
    fraud_probability_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraud probability (0.0 = legitimate, 1.0 = certain fraud)"
    )
    risk_decision: str = Field(
        ..., description="Routing decision: APPROVE | REVIEW | FLAG"
    )
    risk_tier: str = Field(
        ..., description="Risk tier: low | medium | high"
    )
    scoring_latency_ms: float = Field(
        ..., description="End-to-end scoring latency in milliseconds"
    )
    scoring_method: str = Field(
        ..., description="Scoring method used: vertex_ai | fallback_heuristic"
    )
    model_version: str = Field(
        ..., description="Model version used for scoring"
    )


class RiskFactor(BaseModel):
    """Individual risk factor identified in deep analysis."""
    factor: str = Field(..., description="Risk factor description")
    severity: str = Field(..., description="Severity: low | medium | high | critical")
    evidence: str = Field(..., description="Evidence supporting this risk factor")


class DeepAnalysisResponse(BaseModel):
    """Response from RAG-based deep fraud analysis (explainable)."""

    transaction_id: str = Field(..., description="Analyzed transaction ID")
    fraud_probability_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Refined fraud probability after deep analysis"
    )
    risk_factors: List[RiskFactor] = Field(
        ..., description="Identified risk factors with evidence"
    )
    executive_summary: str = Field(
        ..., description="Human-readable risk explanation for analysts"
    )
    similar_historical_cases: List[str] = Field(
        default_factory=list,
        description="IDs of similar historical fraud cases found"
    )
    recommended_action: str = Field(
        ..., description="Recommended action: BLOCK | HOLD_FOR_REVIEW | ESCALATE"
    )
    analysis_latency_ms: float = Field(
        ..., description="Deep analysis latency in milliseconds"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence in the analysis"
    )


class BatchScoreResponse(BaseModel):
    """Response for batch scoring request."""
    results: List[FraudScoreResponse] = Field(
        ..., description="Scoring results in same order as input"
    )
    batch_size: int = Field(..., description="Number of transactions scored")
    total_latency_ms: float = Field(..., description="Total batch processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy | degraded")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Current server timestamp")
    version: str = Field(default="1.0.0", description="API version")


class ReadinessResponse(BaseModel):
    """Readiness probe response with dependency checks."""
    status: str = Field(..., description="Ready status: ready | not_ready")
    vertex_ai_prediction: str = Field(..., description="Vertex AI endpoint status")
    vertex_ai_vector_search: str = Field(..., description="Vector Search status")
    gemini_llm: str = Field(..., description="Gemini LLM status")
    pubsub: str = Field(..., description="Pub/Sub connectivity")
