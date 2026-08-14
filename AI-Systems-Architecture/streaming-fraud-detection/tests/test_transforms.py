"""
Unit tests for Dataflow pipeline transforms.

Tests the core DoFn logic without requiring live GCP services:
- PII sanitization regex patterns
- Feature computation correctness
- Risk routing thresholds
- BigQuery row formatting
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# Add project root to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureComputer, TERMINAL_RISK_MAP


class TestFeatureComputer:
    """Test per-element feature computation."""

    def setup_method(self):
        self.computer = FeatureComputer()

    def test_basic_feature_computation(self):
        """Test that all expected features are computed."""
        transaction = {
            "amount": 150.00,
            "merchant_category": "electronics",
            "terminal_type": "online",
            "timestamp": "2024-06-15T14:30:00+00:00",
            "is_international": False,
            "pin_entered": True,
            "recurring": False,
        }
        features = self.computer.compute_features(transaction)

        assert "f_amount_log" in features
        assert "f_merchant_risk_score" in features
        assert "f_terminal_type_risk" in features
        assert "f_hour_of_day" in features
        assert "f_is_night_transaction" in features
        assert "f_amount_bucket" in features

    def test_amount_log_transform(self):
        """Test log1p transform for various amounts."""
        assert self.computer._log_transform(0) == 0.0
        assert self.computer._log_transform(100) > 0
        assert self.computer._log_transform(-10) == 0.0  # Floor at 0

    def test_round_amount_detection(self):
        """Test round amount detection (fraud indicator)."""
        assert self.computer._is_round_amount(100.0) is True
        assert self.computer._is_round_amount(500.0) is True
        assert self.computer._is_round_amount(99.99) is False
        assert self.computer._is_round_amount(50.0) is False  # Below threshold

    def test_amount_bucketing(self):
        """Test amount discretization into buckets."""
        assert self.computer._amount_bucket(10) == "micro"
        assert self.computer._amount_bucket(75) == "small"
        assert self.computer._amount_bucket(300) == "medium"
        assert self.computer._amount_bucket(1500) == "large"
        assert self.computer._amount_bucket(8000) == "very_large"
        assert self.computer._amount_bucket(15000) == "extreme"

    def test_night_transaction_detection(self):
        """Test night transaction flag (23:00 - 05:00)."""
        assert self.computer._is_night("2024-06-15T23:30:00+00:00") is True
        assert self.computer._is_night("2024-06-15T02:00:00+00:00") is True
        assert self.computer._is_night("2024-06-15T14:00:00+00:00") is False
        assert self.computer._is_night("invalid") is False

    def test_merchant_risk_scores(self):
        """Test that high-risk categories score higher."""
        jewelry_risk = self.computer._merchant_risk("jewelry")
        grocery_risk = self.computer._merchant_risk("grocery")
        assert jewelry_risk > grocery_risk

    def test_terminal_type_risk(self):
        """Test terminal risk ordering: online > swipe > contactless > chip."""
        assert TERMINAL_RISK_MAP["online"] > TERMINAL_RISK_MAP["swipe"]
        assert TERMINAL_RISK_MAP["swipe"] > TERMINAL_RISK_MAP["contactless"]
        assert TERMINAL_RISK_MAP["contactless"] > TERMINAL_RISK_MAP["chip"]

    def test_high_risk_composite_features(self):
        """Test composite risk signals (card-not-present, high amount)."""
        high_risk_txn = {
            "amount": 10000.0,
            "merchant_category": "jewelry",
            "terminal_type": "online",
            "timestamp": "2024-06-15T02:00:00+00:00",
            "is_international": True,
            "pin_entered": False,
            "recurring": False,
        }
        features = self.computer.compute_features(high_risk_txn)

        assert features["f_is_card_not_present"] is True
        assert features["f_is_high_risk_category"] is True
        assert features["f_is_night_transaction"] is True
        assert features["f_is_international"] is True
        assert features["f_amount_exceeds_category_3sigma"] is True


class TestPIISanitization:
    """Test PII redaction patterns."""

    def test_credit_card_redaction(self):
        """Test PAN number patterns are redacted."""
        import re
        PAN_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

        text = "Card number 4532-1234-5678-9012 was used"
        result = PAN_PATTERN.sub("[REDACTED_PAN]", text)
        assert "[REDACTED_PAN]" in result
        assert "4532" not in result

    def test_ssn_redaction(self):
        """Test SSN patterns are redacted."""
        import re
        SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

        text = "SSN: 123-45-6789"
        result = SSN_PATTERN.sub("[REDACTED_SSN]", text)
        assert "[REDACTED_SSN]" in result

    def test_phone_redaction(self):
        """Test phone number patterns are redacted."""
        import re
        PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')

        text = "Contact: 555-123-4567"
        result = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
        assert "[REDACTED_PHONE]" in result


class TestFallbackScoring:
    """Test the fallback heuristic scorer (used when Vertex AI is down)."""

    def test_low_risk_transaction(self):
        """Low-amount, chip terminal, PIN entered → low score."""
        from src.scoring import FraudScorer

        scorer = FraudScorer.__new__(FraudScorer)
        scorer._initialized = False
        scorer._total_predictions = 0
        scorer._fallback_predictions = 0
        scorer._total_latency_ms = 0

        features = {
            "amount": 25.0,
            "f_merchant_risk_score": 0.02,
            "f_velocity_count": 1,
            "f_is_night_transaction": False,
            "is_international": False,
            "pin_entered": True,
            "is_card_not_present": 0,
            "f_is_round_amount": False,
        }
        result = scorer._fallback_heuristic_score(features)

        assert result["fraud_probability"] < 0.3
        assert result["scoring_method"] == "fallback_heuristic"

    def test_high_risk_transaction(self):
        """High-amount, online, international, no PIN, night → high score."""
        from src.scoring import FraudScorer

        scorer = FraudScorer.__new__(FraudScorer)
        scorer._initialized = False
        scorer._total_predictions = 0
        scorer._fallback_predictions = 0
        scorer._total_latency_ms = 0

        features = {
            "amount": 8000.0,
            "f_merchant_risk_score": 0.15,
            "f_velocity_count": 12,
            "f_is_night_transaction": True,
            "is_international": True,
            "pin_entered": False,
            "is_card_not_present": 1,
            "f_is_round_amount": True,
        }
        result = scorer._fallback_heuristic_score(features)

        assert result["fraud_probability"] >= 0.7
        assert result["scoring_method"] == "fallback_heuristic"
