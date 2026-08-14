"""
Custom Apache Beam DoFns for the streaming fraud detection pipeline.

This module contains the core transformation logic applied to each POS
transaction event as it flows through the Dataflow pipeline:

1. ParseTransaction   — Deserialize JSON from Pub/Sub message bytes
2. SanitizePII        — Regex-based redaction of sensitive fields
3. EnrichFeatures     — Compute velocity & behavioral features (windowed)
4. ScoreTransaction   — Call Vertex AI Online Prediction for fraud probability
5. RouteByRisk        — Split stream into approve/flag paths
6. FormatForBigQuery  — Transform to BigQuery row schema for streaming insert
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import apache_beam as beam
from apache_beam.metrics import Metrics

logger = logging.getLogger(__name__)


# =============================================================================
# DoFn 1: Parse raw Pub/Sub message into transaction dict
# =============================================================================

class ParseTransaction(beam.DoFn):
    """
    Deserializes raw Pub/Sub message bytes into a Python dictionary.
    
    Handles malformed messages gracefully by routing them to a dead-letter
    output for later inspection rather than failing the pipeline.
    """

    VALID_OUTPUT = "valid"
    DEAD_LETTER = "dead_letter"

    def __init__(self):
        self.parse_success_counter = Metrics.counter(self.__class__, "parse_success")
        self.parse_failure_counter = Metrics.counter(self.__class__, "parse_failure")

    def process(self, element):
        """
        Args:
            element: Raw Pub/Sub message (bytes)
            
        Yields:
            Tagged output: 'valid' for successfully parsed transactions,
                          'dead_letter' for malformed messages
        """
        try:
            message_data = element.data.decode("utf-8")
            transaction = json.loads(message_data)

            # Validate required fields
            required_fields = ["transaction_id", "timestamp", "card_token", "amount"]
            for field in required_fields:
                if field not in transaction:
                    raise ValueError(f"Missing required field: {field}")

            # Attach Pub/Sub publish timestamp for latency tracking
            transaction["_pubsub_publish_time"] = element.publish_time.isoformat()
            transaction["_processing_start_time"] = datetime.now(timezone.utc).isoformat()

            self.parse_success_counter.inc()
            yield beam.pvalue.TaggedOutput(self.VALID_OUTPUT, transaction)

        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
            self.parse_failure_counter.inc()
            error_record = {
                "raw_data": element.data[:1000],  # Truncate for storage safety
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield beam.pvalue.TaggedOutput(self.DEAD_LETTER, error_record)


# =============================================================================
# DoFn 2: PII Sanitization (Regex-based redaction)
# =============================================================================

class SanitizePII(beam.DoFn):
    """
    Applies strict regex-based PII redaction to transaction fields.
    
    Redacts:
    - Credit card numbers (PAN patterns)
    - Social Security Numbers
    - Phone numbers
    - Email addresses (partial redaction)
    
    This runs BEFORE data hits BigQuery or any persistent store, ensuring
    PII never lands in the data warehouse.
    """

    # Compiled regex patterns for performance (compiled once per worker)
    PAN_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def __init__(self):
        self.redaction_counter = Metrics.counter(self.__class__, "pii_redactions")

    def process(self, transaction: Dict[str, Any]):
        """Redact PII from all string fields in the transaction."""
        sanitized = {}
        for key, value in transaction.items():
            if isinstance(value, str) and not key.startswith("_"):
                original = value
                value = self.PAN_PATTERN.sub("[REDACTED_PAN]", value)
                value = self.SSN_PATTERN.sub("[REDACTED_SSN]", value)
                value = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
                value = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
                if value != original:
                    self.redaction_counter.inc()
            sanitized[key] = value

        sanitized["_pii_sanitized"] = True
        yield sanitized


# =============================================================================
# DoFn 3: Feature Engineering (Windowed Aggregations)
# =============================================================================

class EnrichFeatures(beam.DoFn):
    """
    Computes real-time features for fraud scoring using windowed context.
    
    Features computed per transaction:
    - velocity_5min: Number of transactions from same card in last 5 minutes
    - velocity_30min: Number of transactions from same card in last 30 minutes
    - amount_deviation: How far this amount deviates from card's rolling average
    - geo_velocity_kmh: Implied travel speed between consecutive transactions
    - time_since_last_txn: Seconds since card's previous transaction
    - is_round_amount: Fraudsters often use round numbers
    - merchant_risk_score: Historical fraud rate for this merchant category
    
    In production, windowed aggregations use Beam's windowing + state API.
    This DoFn computes per-element features that don't require cross-element state.
    """

    # Historical merchant category fraud rates (would come from BigQuery side input in prod)
    MERCHANT_RISK_SCORES = {
        "grocery": 0.02,
        "electronics": 0.08,
        "gas_station": 0.05,
        "restaurant": 0.03,
        "online_retail": 0.12,
        "jewelry": 0.15,
        "travel": 0.10,
        "pharmacy": 0.02,
        "convenience_store": 0.04,
        "department_store": 0.06,
    }

    def __init__(self):
        self.feature_counter = Metrics.counter(self.__class__, "features_computed")

    def process(self, transaction: Dict[str, Any]):
        """Compute per-element features and attach to transaction."""
        amount = transaction.get("amount", 0)
        category = transaction.get("merchant_category", "unknown")

        # Per-element features (no state required)
        features = {
            "f_amount_log": round(self._safe_log(amount), 4),
            "f_is_round_amount": amount == int(amount) and amount >= 100,
            "f_is_international": transaction.get("is_international", False),
            "f_pin_entered": transaction.get("pin_entered", False),
            "f_merchant_risk_score": self.MERCHANT_RISK_SCORES.get(category, 0.05),
            "f_terminal_type_risk": self._terminal_risk(transaction.get("terminal_type", "")),
            "f_hour_of_day": self._extract_hour(transaction.get("timestamp", "")),
            "f_is_night_transaction": self._is_night(transaction.get("timestamp", "")),
            "f_amount_bucket": self._amount_bucket(amount),
        }

        # Merge features into transaction
        transaction["features"] = features
        self.feature_counter.inc()
        yield transaction

    @staticmethod
    def _safe_log(value: float) -> float:
        """Log transform with floor to handle zero/negative amounts."""
        import math
        return math.log1p(max(0, value))

    @staticmethod
    def _terminal_risk(terminal_type: str) -> float:
        """Risk score based on terminal type (online/swipe = higher risk)."""
        risk_map = {
            "chip": 0.02,
            "contactless": 0.03,
            "swipe": 0.08,
            "online": 0.12,
        }
        return risk_map.get(terminal_type, 0.05)

    @staticmethod
    def _extract_hour(timestamp_str: str) -> int:
        """Extract hour of day from ISO timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.hour
        except (ValueError, AttributeError):
            return -1

    @staticmethod
    def _is_night(timestamp_str: str) -> bool:
        """Flag transactions between 11 PM and 5 AM (higher fraud rate)."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.hour >= 23 or dt.hour < 5
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _amount_bucket(amount: float) -> str:
        """Categorize amount into buckets for one-hot encoding downstream."""
        if amount < 25:
            return "micro"
        elif amount < 100:
            return "small"
        elif amount < 500:
            return "medium"
        elif amount < 2000:
            return "large"
        return "very_large"


# =============================================================================
# DoFn 4: Vertex AI Online Prediction (Real-Time Scoring)
# =============================================================================

class ScoreTransaction(beam.DoFn):
    """
    Calls Vertex AI Online Prediction endpoint for real-time fraud scoring.
    
    Converts computed features into a prediction request, sends to Vertex AI,
    and attaches the fraud_probability_score to the transaction.
    
    In production, this uses the Vertex AI Prediction client with connection
    pooling and circuit breaker patterns for resilience.
    """

    def __init__(self, project_id: str, endpoint_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.region = region
        self.prediction_client = None
        self.scoring_latency = Metrics.distribution(self.__class__, "scoring_latency_ms")
        self.scoring_errors = Metrics.counter(self.__class__, "scoring_errors")

    def setup(self):
        """Initialize Vertex AI Prediction client (once per worker)."""
        from google.cloud import aiplatform
        aiplatform.init(project=self.project_id, location=self.region)

        self.endpoint = aiplatform.Endpoint(self.endpoint_id)
        logger.info(f"Vertex AI Prediction client initialized: endpoint={self.endpoint_id}")

    def process(self, transaction: Dict[str, Any]):
        """Score transaction and attach fraud probability."""
        start_time = time.time()

        try:
            features = transaction.get("features", {})

            # Build prediction instance from computed features
            instance = {
                "amount_log": features.get("f_amount_log", 0),
                "is_round_amount": float(features.get("f_is_round_amount", False)),
                "is_international": float(features.get("f_is_international", False)),
                "pin_entered": float(features.get("f_pin_entered", False)),
                "merchant_risk_score": features.get("f_merchant_risk_score", 0.05),
                "terminal_type_risk": features.get("f_terminal_type_risk", 0.05),
                "hour_of_day": features.get("f_hour_of_day", 12),
                "is_night_transaction": float(features.get("f_is_night_transaction", False)),
            }

            # Call Vertex AI endpoint
            prediction = self.endpoint.predict(instances=[instance])
            fraud_score = prediction.predictions[0].get("fraud_probability", 0.0)

            latency_ms = (time.time() - start_time) * 1000
            self.scoring_latency.update(int(latency_ms))

            transaction["fraud_probability_score"] = round(fraud_score, 4)
            transaction["_scoring_latency_ms"] = round(latency_ms, 2)
            transaction["_scoring_timestamp"] = datetime.now(timezone.utc).isoformat()

            yield transaction

        except Exception as e:
            self.scoring_errors.inc()
            logger.error(f"Scoring failed for {transaction.get('transaction_id')}: {str(e)}")
            # On scoring failure, flag as medium risk for manual review
            transaction["fraud_probability_score"] = 0.5
            transaction["_scoring_error"] = str(e)
            yield transaction


# =============================================================================
# DoFn 5: Route by Risk Level
# =============================================================================

class RouteByRisk(beam.DoFn):
    """
    Splits the scored transaction stream into risk-based paths:
    
    - APPROVE (score < 0.3): Low risk, auto-approve
    - REVIEW  (0.3 <= score < 0.7): Medium risk, queue for manual review
    - FLAG    (score >= 0.7): High risk, trigger deep RAG analysis
    
    Each path writes to different downstream sinks with different SLAs.
    """

    APPROVE = "approve"
    REVIEW = "review"
    FLAG = "flag"

    def __init__(self, high_risk_threshold: float = 0.7, medium_risk_threshold: float = 0.3):
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.approve_counter = Metrics.counter(self.__class__, "transactions_approved")
        self.review_counter = Metrics.counter(self.__class__, "transactions_review")
        self.flag_counter = Metrics.counter(self.__class__, "transactions_flagged")

    def process(self, transaction: Dict[str, Any]):
        """Route transaction based on fraud probability score."""
        score = transaction.get("fraud_probability_score", 0.0)

        if score >= self.high_risk_threshold:
            transaction["risk_decision"] = "FLAG"
            transaction["risk_tier"] = "high"
            self.flag_counter.inc()
            yield beam.pvalue.TaggedOutput(self.FLAG, transaction)

        elif score >= self.medium_risk_threshold:
            transaction["risk_decision"] = "REVIEW"
            transaction["risk_tier"] = "medium"
            self.review_counter.inc()
            yield beam.pvalue.TaggedOutput(self.REVIEW, transaction)

        else:
            transaction["risk_decision"] = "APPROVE"
            transaction["risk_tier"] = "low"
            self.approve_counter.inc()
            yield beam.pvalue.TaggedOutput(self.APPROVE, transaction)


# =============================================================================
# DoFn 6: Format for BigQuery Streaming Insert
# =============================================================================

class FormatForBigQuery(beam.DoFn):
    """
    Transforms enriched transaction into a flat BigQuery row schema.
    
    BigQuery streaming inserts require a flat dict matching the table schema.
    Nested feature objects are flattened with 'f_' prefix.
    Internal metadata fields (prefixed '_') are included for lineage tracking.
    """

    def process(self, transaction: Dict[str, Any]):
        """Flatten transaction into BigQuery-compatible row."""
        features = transaction.get("features", {})

        bq_row = {
            # Core transaction fields
            "transaction_id": transaction.get("transaction_id"),
            "event_timestamp": transaction.get("timestamp"),
            "card_token": transaction.get("card_token"),
            "merchant_id": transaction.get("merchant_id"),
            "merchant_category": transaction.get("merchant_category"),
            "terminal_id": transaction.get("terminal_id"),
            "terminal_type": transaction.get("terminal_type"),
            "amount": transaction.get("amount"),
            "currency": transaction.get("currency"),
            "region": transaction.get("region"),
            "is_international": transaction.get("is_international", False),
            "pin_entered": transaction.get("pin_entered", False),
            "recurring": transaction.get("recurring", False),

            # Computed features
            "f_amount_log": features.get("f_amount_log"),
            "f_is_round_amount": features.get("f_is_round_amount", False),
            "f_merchant_risk_score": features.get("f_merchant_risk_score"),
            "f_terminal_type_risk": features.get("f_terminal_type_risk"),
            "f_hour_of_day": features.get("f_hour_of_day"),
            "f_is_night_transaction": features.get("f_is_night_transaction", False),
            "f_amount_bucket": features.get("f_amount_bucket"),

            # Scoring results
            "fraud_probability_score": transaction.get("fraud_probability_score"),
            "risk_decision": transaction.get("risk_decision"),
            "risk_tier": transaction.get("risk_tier"),

            # Lineage & observability metadata
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "scoring_latency_ms": transaction.get("_scoring_latency_ms"),
            "pii_sanitized": transaction.get("_pii_sanitized", False),
        }

        yield bq_row


# =============================================================================
# Composite Transform: Full Feature Engineering with Windowing
# =============================================================================

class ComputeWindowedVelocity(beam.PTransform):
    """
    Composite PTransform that computes windowed velocity features.
    
    Groups transactions by card_token within sliding windows and computes:
    - Transaction count per window (velocity)
    - Sum of amounts per window
    - Distinct merchant count per window
    
    These are attached back to individual transactions via a CoGroupByKey.
    """

    def __init__(self, window_size_seconds: int = 300, window_period_seconds: int = 60):
        """
        Args:
            window_size_seconds: Sliding window size (default 5 minutes)
            window_period_seconds: Sliding window period (default 1 minute)
        """
        super().__init__()
        self.window_size = window_size_seconds
        self.window_period = window_period_seconds

    def expand(self, pcoll):
        """Apply sliding window and compute velocity aggregations."""
        return (
            pcoll
            | "KeyByCard" >> beam.Map(lambda txn: (txn["card_token"], txn))
            | "SlidingWindow" >> beam.WindowInto(
                beam.window.SlidingWindows(self.window_size, self.window_period)
            )
            | "GroupByCard" >> beam.GroupByKey()
            | "ComputeVelocity" >> beam.ParDo(self._ComputeVelocityFn())
        )

    class _ComputeVelocityFn(beam.DoFn):
        """Compute velocity features for a group of transactions in a window."""

        def process(self, element):
            card_token, transactions = element
            txn_list = list(transactions)
            count = len(txn_list)
            total_amount = sum(t.get("amount", 0) for t in txn_list)
            distinct_merchants = len(set(t.get("merchant_id", "") for t in txn_list))

            # Attach velocity features to each transaction in the window
            for txn in txn_list:
                txn.setdefault("features", {})
                txn["features"]["f_velocity_count"] = count
                txn["features"]["f_velocity_total_amount"] = round(total_amount, 2)
                txn["features"]["f_velocity_distinct_merchants"] = distinct_merchants
                txn["features"]["f_velocity_avg_amount"] = round(total_amount / count, 2) if count > 0 else 0
                yield txn
