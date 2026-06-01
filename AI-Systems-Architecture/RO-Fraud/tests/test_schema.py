"""
Unit tests for Pydantic schemas.
"""
import pytest
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import ClaimRequest, RiskSummaryResponse

def test_claim_request_valid():
    """Test valid ClaimRequest creation."""
    claim = ClaimRequest(
        claim_id="C-001",
        customer_id="CUST-001",
        claim_text="Test claim description",
        claim_amount=1000.0
    )
    assert claim.claim_id == "C-001"
    assert claim.customer_id == "CUST-001"
    assert claim.claim_amount == 1000.0

def test_claim_request_missing_fields():
    """Test ClaimRequest with missing required fields."""
    with pytest.raises(ValidationError):
        ClaimRequest(
            claim_id="C-001",
            # Missing other required fields
        )

def test_claim_request_invalid_amount():
    """Test ClaimRequest with invalid amount type."""
    with pytest.raises(ValidationError):
        ClaimRequest(
            claim_id="C-001",
            customer_id="CUST-001",
            claim_text="Test",
            claim_amount="invalid"  # Should be float
        )

def test_risk_summary_response_valid():
    """Test valid RiskSummaryResponse creation."""
    response = RiskSummaryResponse(
        fraud_probability_score=0.75,
        risk_factors=["Factor 1", "Factor 2"],
        executive_summary="Test summary"
    )
    assert response.fraud_probability_score == 0.75
    assert len(response.risk_factors) == 2
    assert response.executive_summary == "Test summary"

def test_risk_summary_response_score_bounds():
    """Test RiskSummaryResponse accepts valid score range."""
    # Valid scores
    RiskSummaryResponse(
        fraud_probability_score=0.0,
        risk_factors=[],
        executive_summary="Low risk"
    )
    RiskSummaryResponse(
        fraud_probability_score=1.0,
        risk_factors=["High risk"],
        executive_summary="High risk"
    )
    RiskSummaryResponse(
        fraud_probability_score=0.5,
        risk_factors=["Medium risk"],
        executive_summary="Medium risk"
    )

def test_risk_summary_response_empty_factors():
    """Test RiskSummaryResponse with empty risk factors."""
    response = RiskSummaryResponse(
        fraud_probability_score=0.1,
        risk_factors=[],
        executive_summary="No significant risk factors identified"
    )
    assert len(response.risk_factors) == 0

def test_claim_request_json_serialization():
    """Test ClaimRequest JSON serialization."""
    claim = ClaimRequest(
        claim_id="C-001",
        customer_id="CUST-001",
        claim_text="Test claim",
        claim_amount=1500.50
    )
    json_data = claim.model_dump()
    assert json_data["claim_id"] == "C-001"
    assert json_data["claim_amount"] == 1500.50

def test_risk_summary_response_json_serialization():
    """Test RiskSummaryResponse JSON serialization."""
    response = RiskSummaryResponse(
        fraud_probability_score=0.85,
        risk_factors=["Factor A", "Factor B"],
        executive_summary="High risk detected"
    )
    json_data = response.model_dump()
    assert json_data["fraud_probability_score"] == 0.85
    assert len(json_data["risk_factors"]) == 2
