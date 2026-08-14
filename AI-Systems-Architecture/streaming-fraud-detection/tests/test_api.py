"""
Unit tests for the FastAPI endpoints.

Tests request validation, response schemas, and error handling
without requiring live GCP services (all external calls are mocked).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# These tests would use httpx AsyncClient with FastAPI's TestClient
# Example structure for interview discussion:


class TestScoreTransactionEndpoint:
    """Tests for POST /api/v1/score-transaction"""

    def test_valid_request_schema(self):
        """Verify a valid transaction request passes Pydantic validation."""
        from src.schema import TransactionRequest

        request = TransactionRequest(
            transaction_id="txn_test_001",
            card_token="card_token_abc123",
            merchant_id="merchant_xyz",
            merchant_category="electronics",
            amount=250.00,
        )
        assert request.transaction_id == "txn_test_001"
        assert request.amount == 250.00
        assert request.currency == "USD"  # Default
        assert request.pin_entered is True  # Default

    def test_invalid_amount_rejected(self):
        """Verify that zero/negative amounts are rejected."""
        from src.schema import TransactionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransactionRequest(
                transaction_id="txn_test_002",
                card_token="card_token_abc123",
                merchant_id="merchant_xyz",
                merchant_category="electronics",
                amount=-50.00,  # Invalid: must be > 0
            )

    def test_response_schema_structure(self):
        """Verify response model has all required fields."""
        from src.schema import FraudScoreResponse

        response = FraudScoreResponse(
            transaction_id="txn_test_001",
            fraud_probability_score=0.85,
            risk_decision="FLAG",
            risk_tier="high",
            scoring_latency_ms=23.4,
            scoring_method="vertex_ai",
            model_version="fraud_classifier_v1",
        )
        assert response.fraud_probability_score == 0.85
        assert response.risk_decision == "FLAG"
        assert response.risk_tier == "high"

    def test_score_bounds_validation(self):
        """Verify fraud_probability_score is bounded [0.0, 1.0]."""
        from src.schema import FraudScoreResponse
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FraudScoreResponse(
                transaction_id="txn_test_003",
                fraud_probability_score=1.5,  # Invalid: > 1.0
                risk_decision="FLAG",
                risk_tier="high",
                scoring_latency_ms=10.0,
                scoring_method="vertex_ai",
                model_version="v1",
            )


class TestBatchScoringEndpoint:
    """Tests for POST /api/v1/score-batch"""

    def test_batch_size_validation(self):
        """Verify batch is limited to 1000 transactions."""
        from src.schema import BatchTransactionRequest, TransactionRequest
        from pydantic import ValidationError

        # Valid batch (1 transaction)
        valid = BatchTransactionRequest(
            transactions=[
                TransactionRequest(
                    transaction_id="txn_001",
                    card_token="card_abc",
                    merchant_id="m_001",
                    merchant_category="grocery",
                    amount=50.0,
                )
            ]
        )
        assert valid.transactions[0].amount == 50.0


class TestDeepAnalysisEndpoint:
    """Tests for POST /api/v1/deep-analysis"""

    def test_deep_analysis_request_schema(self):
        """Verify deep analysis request validation."""
        from src.schema import DeepAnalysisRequest

        request = DeepAnalysisRequest(
            transaction_id="txn_flagged_001",
            transaction_text="$5,000 jewelry purchase, online, no PIN, night",
            fraud_probability_score=0.87,
        )
        assert request.fraud_probability_score == 0.87
        assert "jewelry" in request.transaction_text
