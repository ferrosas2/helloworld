"""
Unit tests for FastAPI endpoints.

The app initializes its GCP-backed clients in a lifespan handler, so these tests
patch the retriever/analyzer classes and drive the app through the TestClient
context manager (which runs startup/shutdown) to inject mocks.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_retriever():
    """A mock FraudPatternRetriever instance."""
    instance = Mock()
    instance.get_similar_historical_claims.return_value = [
        "Historical fraud case 1: Suspicious claim pattern detected",
        "Historical fraud case 2: Multiple claims from same location",
    ]
    # readiness probe calls vector_store.similarity_search
    instance.vector_store.similarity_search.return_value = []
    return instance


@pytest.fixture
def mock_analyzer():
    """A mock RiskAnalyzerLLM instance."""
    instance = Mock()
    instance.analyze_claim.return_value = {
        "fraud_probability_score": 0.85,
        "risk_factors": ["Inconsistent timeline", "Unusual claim amount"],
        "executive_summary": "High fraud risk detected based on historical patterns",
    }
    return instance


@pytest.fixture
def client(mock_retriever, mock_analyzer):
    """Test client with GCP clients mocked via the lifespan startup."""
    with patch("api.main.FraudPatternRetriever", return_value=mock_retriever), \
         patch("api.main.RiskAnalyzerLLM", return_value=mock_analyzer):
        from api.main import app
        # Entering the context manager runs the lifespan startup, which assigns
        # the mocked instances to the module-level globals.
        with TestClient(app) as test_client:
            yield test_client


def test_health_check(client):
    """Health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "ro-fraud-api"}


def test_readiness_check(client):
    """Readiness endpoint returns ready when Vector Search is reachable."""
    response = client.get("/readiness")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_analyze_claim_success(client):
    """Successful claim analysis returns a structured risk summary."""
    request_data = {
        "claim_id": "C-TEST-001",
        "customer_id": "CUST-001",
        "claim_text": "My car was stolen from the parking lot",
        "claim_amount": 25000.0,
    }
    response = client.post("/api/v1/analyze-claim", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fraud_probability_score"] <= 1.0
    assert isinstance(data["risk_factors"], list)
    assert data["executive_summary"]


def test_analyze_claim_invalid_request(client):
    """Missing required fields returns a 422 validation error."""
    response = client.post("/api/v1/analyze-claim", json={"claim_id": "C-TEST-001"})
    assert response.status_code == 422


def test_analyze_claim_retriever_error(client, mock_retriever):
    """A retriever failure surfaces as a 500."""
    mock_retriever.get_similar_historical_claims.side_effect = Exception("Vector search failed")
    request_data = {
        "claim_id": "C-TEST-002",
        "customer_id": "CUST-002",
        "claim_text": "Test claim",
        "claim_amount": 1000.0,
    }
    response = client.post("/api/v1/analyze-claim", json=request_data)
    assert response.status_code == 500


def test_analyze_claim_analyzer_error(client, mock_analyzer):
    """An LLM failure surfaces as a 500."""
    mock_analyzer.analyze_claim.side_effect = Exception("LLM generation failed")
    request_data = {
        "claim_id": "C-TEST-003",
        "customer_id": "CUST-003",
        "claim_text": "Test claim",
        "claim_amount": 1000.0,
    }
    response = client.post("/api/v1/analyze-claim", json=request_data)
    assert response.status_code == 500


def test_openapi_docs(client):
    """OpenAPI docs are served."""
    assert client.get("/api/docs").status_code == 200
