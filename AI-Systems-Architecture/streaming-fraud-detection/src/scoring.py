"""
Vertex AI Online Prediction Client for Real-Time Fraud Scoring.

This module handles communication with the Vertex AI endpoint that serves
the trained BOOSTED_TREE_CLASSIFIER model. It provides:

1. Low-latency prediction (<50ms p99 target)
2. Connection pooling and warm-up for cold-start avoidance
3. Circuit breaker pattern for endpoint unavailability
4. Fallback scoring when Vertex AI is degraded
5. Batch prediction support for bulk re-scoring

The model is trained in BigQuery ML (see sql/02_train_fraud_classifier.sql)
and exported to Vertex AI for online serving via:
    bq extract --model fraud_detection.fraud_classifier_v1 gs://bucket/model/
    gcloud ai models upload --artifact-uri=gs://bucket/model/ ...

Usage:
    from src.scoring import FraudScorer
    
    scorer = FraudScorer(project_id="my-project", endpoint_id="123456")
    result = await scorer.score_transaction(features)
    # result = {"fraud_probability": 0.87, "latency_ms": 23.4, "model_version": "v1"}
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict as predict_schema
from google.api_core import exceptions as gcp_exceptions
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration & Constants
# =============================================================================

# Feature names expected by the model (must match training schema)
MODEL_FEATURE_NAMES = [
    "amount",
    "f_amount_log",
    "f_is_round_amount",
    "f_merchant_risk_score",
    "f_terminal_type_risk",
    "f_hour_of_day",
    "f_is_night_transaction",
    "is_international",
    "pin_entered",
    "recurring",
    "f_velocity_count",
    "f_velocity_total_amount",
    "f_velocity_distinct_merchants",
    "is_high_risk_category",
    "is_card_not_present",
    "velocity_alert_flag",
]

# Default feature values when data is missing (safe defaults = low risk)
DEFAULT_FEATURE_VALUES = {
    "amount": 0.0,
    "f_amount_log": 0.0,
    "f_is_round_amount": 0.0,
    "f_merchant_risk_score": 0.05,
    "f_terminal_type_risk": 0.03,
    "f_hour_of_day": 12,
    "f_is_night_transaction": 0.0,
    "is_international": 0.0,
    "pin_entered": 1.0,
    "recurring": 0.0,
    "f_velocity_count": 1,
    "f_velocity_total_amount": 50.0,
    "f_velocity_distinct_merchants": 1,
    "is_high_risk_category": 0,
    "is_card_not_present": 0,
    "velocity_alert_flag": 0,
}


# =============================================================================
# Circuit Breaker (resilience pattern)
# =============================================================================

@dataclass
class CircuitBreakerState:
    """
    Tracks Vertex AI endpoint health for circuit breaker pattern.
    
    States:
        CLOSED  → normal operation, requests flow through
        OPEN    → endpoint unhealthy, use fallback scoring
        HALF    → testing recovery, allow limited requests
    """
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3

    def record_success(self):
        """Record a successful prediction call."""
        self.success_count += 1
        if self.state == "HALF_OPEN" and self.success_count >= self.half_open_max_calls:
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info("Circuit breaker CLOSED — Vertex AI endpoint recovered.")

    def record_failure(self):
        """Record a failed prediction call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN — {self.failure_count} consecutive failures.")

    def should_allow_request(self) -> bool:
        """Determine if a request should be sent to Vertex AI."""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            elapsed = time.time() - (self.last_failure_time or 0)
            if elapsed >= self.recovery_timeout_seconds:
                self.state = "HALF_OPEN"
                self.success_count = 0
                logger.info("Circuit breaker HALF_OPEN — testing Vertex AI recovery.")
                return True
            return False
        # HALF_OPEN: allow limited requests
        return True


# =============================================================================
# Fraud Scorer (Main Class)
# =============================================================================

class FraudScorer:
    """
    Production-grade Vertex AI Online Prediction client for fraud scoring.
    
    Implements:
    - Connection pooling via gRPC (handled by google-cloud-aiplatform SDK)
    - Circuit breaker pattern for graceful degradation
    - Fallback heuristic scoring when Vertex AI is unavailable
    - Latency tracking for SLO monitoring
    - Batch scoring for bulk re-processing
    
    Architecture:
        Dataflow Worker → FraudScorer → Vertex AI Endpoint → Prediction Response
                                    ↓ (on failure)
                              Fallback Heuristic Scorer
    """

    def __init__(
        self,
        project_id: str,
        endpoint_id: str,
        region: str = "us-central1",
        timeout_seconds: float = 5.0,
    ):
        """
        Initialize the Vertex AI prediction client.
        
        Args:
            project_id: GCP project ID hosting the endpoint
            endpoint_id: Vertex AI endpoint resource ID
            region: GCP region where the endpoint is deployed
            timeout_seconds: Maximum time to wait for prediction response
        """
        self.project_id = project_id
        self.endpoint_id = endpoint_id
        self.region = region
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker = CircuitBreakerState()

        # Lazily initialized on first call
        self._endpoint = None
        self._initialized = False

        # Metrics
        self._total_predictions = 0
        self._fallback_predictions = 0
        self._total_latency_ms = 0.0

        logger.info(
            f"FraudScorer initialized: project={project_id}, "
            f"endpoint={endpoint_id}, region={region}, timeout={timeout_seconds}s"
        )

    def initialize(self):
        """
        Eagerly initialize the Vertex AI client and endpoint connection.
        Called once per worker in Dataflow DoFn.setup() or on Cloud Run startup.
        """
        if self._initialized:
            return

        try:
            aiplatform.init(project=self.project_id, location=self.region)
            self._endpoint = aiplatform.Endpoint(self.endpoint_id)
            self._initialized = True
            logger.info(f"Vertex AI endpoint connected: {self._endpoint.resource_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI endpoint: {str(e)}")
            raise

    def warm_up(self, num_requests: int = 3):
        """
        Send dummy prediction requests to warm up the endpoint connection.
        Reduces cold-start latency on first real request.
        """
        logger.info(f"Warming up Vertex AI endpoint with {num_requests} dummy requests...")
        dummy_features = DEFAULT_FEATURE_VALUES.copy()

        for i in range(num_requests):
            try:
                self._predict_single(dummy_features)
                logger.info(f"Warm-up request {i + 1}/{num_requests} successful.")
            except Exception as e:
                logger.warning(f"Warm-up request {i + 1} failed (non-critical): {str(e)}")

    def score_transaction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single transaction for fraud probability.
        
        Args:
            features: Dict of computed features matching MODEL_FEATURE_NAMES
            
        Returns:
            Dict with keys:
                - fraud_probability: float (0.0 - 1.0)
                - latency_ms: scoring latency in milliseconds
                - model_version: model serving version
                - scoring_method: 'vertex_ai' or 'fallback_heuristic'
        """
        if not self._initialized:
            self.initialize()

        self._total_predictions += 1

        # Circuit breaker check
        if not self.circuit_breaker.should_allow_request():
            self._fallback_predictions += 1
            return self._fallback_heuristic_score(features)

        # Attempt Vertex AI prediction
        start_time = time.time()
        try:
            prediction = self._predict_single(features)
            latency_ms = (time.time() - start_time) * 1000
            self._total_latency_ms += latency_ms

            self.circuit_breaker.record_success()

            return {
                "fraud_probability": prediction,
                "latency_ms": round(latency_ms, 2),
                "model_version": "fraud_classifier_v1",
                "scoring_method": "vertex_ai",
            }

        except (gcp_exceptions.DeadlineExceeded, gcp_exceptions.ServiceUnavailable) as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"Vertex AI prediction timeout/unavailable: {str(e)}")
            self._fallback_predictions += 1
            return self._fallback_heuristic_score(features)

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Vertex AI prediction error: {str(e)}")
            self._fallback_predictions += 1
            return self._fallback_heuristic_score(features)

    def score_batch(self, feature_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a batch of transactions in a single Vertex AI call.
        
        More efficient than individual calls for bulk re-scoring.
        Vertex AI endpoints support up to 1000 instances per request.
        
        Args:
            feature_batch: List of feature dicts
            
        Returns:
            List of scoring results (same order as input)
        """
        if not self._initialized:
            self.initialize()

        if not self.circuit_breaker.should_allow_request():
            return [self._fallback_heuristic_score(f) for f in feature_batch]

        start_time = time.time()
        try:
            instances = [self._prepare_instance(f) for f in feature_batch]
            predictions = self._endpoint.predict(instances=instances)

            latency_ms = (time.time() - start_time) * 1000
            per_item_latency = latency_ms / len(feature_batch)

            self.circuit_breaker.record_success()

            results = []
            for pred in predictions.predictions:
                score = pred.get("fraud_probability", pred.get("scores", [0.0])[0])
                results.append({
                    "fraud_probability": float(score),
                    "latency_ms": round(per_item_latency, 2),
                    "model_version": "fraud_classifier_v1",
                    "scoring_method": "vertex_ai_batch",
                })
            return results

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Batch prediction failed: {str(e)}")
            return [self._fallback_heuristic_score(f) for f in feature_batch]

    def _predict_single(self, features: Dict[str, Any]) -> float:
        """Make a single prediction call to Vertex AI endpoint."""
        instance = self._prepare_instance(features)
        prediction = self._endpoint.predict(
            instances=[instance],
            timeout=self.timeout_seconds,
        )

        # Extract fraud probability from response
        # BQML exported models return predictions in various formats
        pred = prediction.predictions[0]
        if isinstance(pred, dict):
            return float(pred.get("fraud_probability", pred.get("scores", [0.0])[-1]))
        elif isinstance(pred, (list, tuple)):
            return float(pred[-1])  # Last class probability (fraud)
        return float(pred)

    def _prepare_instance(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a prediction instance from raw features.
        
        Ensures all expected features are present with correct types.
        Missing features are filled with safe defaults.
        """
        instance = {}
        for feature_name in MODEL_FEATURE_NAMES:
            value = features.get(feature_name, DEFAULT_FEATURE_VALUES.get(feature_name, 0))
            # Convert booleans to float (model expects numeric)
            if isinstance(value, bool):
                value = float(value)
            instance[feature_name] = value
        return instance

    def _fallback_heuristic_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback scoring when Vertex AI is unavailable.
        
        This ensures the system never blocks transactions due to ML infra failures.
        Uses simple weighted heuristics that approximate model behavior.
        
        Conservative bias: prefers false positives (flagging legitimate transactions)
        over false negatives (missing fraud), since manual review is cheaper than fraud.
        """
        start_time = time.time()
        score = 0.0

        # Amount-based risk (high amounts = higher risk)
        amount = features.get("amount", 0)
        if amount > 5000:
            score += 0.3
        elif amount > 1000:
            score += 0.15
        elif amount > 500:
            score += 0.05

        # Merchant category risk
        merchant_risk = features.get("f_merchant_risk_score", 0.05)
        score += merchant_risk * 1.5

        # Velocity signal (multiple transactions in short window)
        velocity = features.get("f_velocity_count", 1)
        if velocity > 10:
            score += 0.3
        elif velocity > 5:
            score += 0.15

        # Night transaction premium
        if features.get("f_is_night_transaction", False):
            score += 0.08

        # International without PIN
        if features.get("is_international", False) and not features.get("pin_entered", True):
            score += 0.2

        # Online transaction without PIN (card-not-present)
        if features.get("is_card_not_present", 0):
            score += 0.12

        # Round amount indicator
        if features.get("f_is_round_amount", False):
            score += 0.05

        # Clamp to [0, 1]
        score = min(1.0, max(0.0, score))

        latency_ms = (time.time() - start_time) * 1000

        return {
            "fraud_probability": round(score, 4),
            "latency_ms": round(latency_ms, 2),
            "model_version": "fallback_heuristic_v1",
            "scoring_method": "fallback_heuristic",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return scoring performance statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )
        return {
            "total_predictions": self._total_predictions,
            "fallback_predictions": self._fallback_predictions,
            "fallback_rate": (
                self._fallback_predictions / self._total_predictions
                if self._total_predictions > 0
                else 0.0
            ),
            "avg_latency_ms": round(avg_latency, 2),
            "circuit_breaker_state": self.circuit_breaker.state,
        }


# =============================================================================
# Async Scorer (for Cloud Run FastAPI integration)
# =============================================================================

class AsyncFraudScorer:
    """
    Async wrapper around FraudScorer for use in FastAPI async endpoints.
    
    Runs the synchronous Vertex AI SDK calls in a thread pool executor
    to avoid blocking the asyncio event loop.
    """

    def __init__(self, project_id: str, endpoint_id: str, region: str = "us-central1"):
        self._scorer = FraudScorer(
            project_id=project_id,
            endpoint_id=endpoint_id,
            region=region,
        )

    async def initialize(self):
        """Initialize the underlying scorer in a thread."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._scorer.initialize)

    async def score_transaction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Async score a single transaction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._scorer.score_transaction, features
        )

    async def score_batch(self, feature_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Async batch scoring."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._scorer.score_batch, feature_batch
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return scoring stats."""
        return self._scorer.get_stats()
