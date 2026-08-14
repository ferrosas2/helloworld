"""
Feature Engineering Module for Real-Time Fraud Detection.

Computes features that serve as inputs to the Vertex AI fraud classifier.
Two categories of features:

1. Per-Element Features (computed per transaction, no state):
   - Amount transforms (log, bucket, round-number flag)
   - Terminal/merchant risk scores
   - Time-based features (hour, night flag)

2. Windowed Features (require cross-element state via Beam/Redis):
   - Velocity: transaction count per card in N-minute window
   - Amount aggregations: sum, mean, max in window
   - Behavioral deviation: current amount vs. card's historical mean
   - Geographic velocity: implied travel speed between transactions

In production, windowed features are computed in the Dataflow pipeline
using Apache Beam's StatefulDoFn or SlidingWindows. For the Cloud Run
API path, they can be fetched from a Redis/Memorystore feature store
that's updated by the Dataflow pipeline in real-time.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Merchant Category Risk Profiles (from historical data)
# =============================================================================

# These would be loaded from BigQuery in production and cached daily.
# Based on historical fraud confirmation rates per category.
CATEGORY_RISK_PROFILES = {
    "grocery": {"fraud_rate": 0.02, "avg_amount": 65.0, "stddev_amount": 40.0},
    "electronics": {"fraud_rate": 0.08, "avg_amount": 350.0, "stddev_amount": 500.0},
    "gas_station": {"fraud_rate": 0.05, "avg_amount": 45.0, "stddev_amount": 20.0},
    "restaurant": {"fraud_rate": 0.03, "avg_amount": 55.0, "stddev_amount": 35.0},
    "online_retail": {"fraud_rate": 0.12, "avg_amount": 120.0, "stddev_amount": 200.0},
    "jewelry": {"fraud_rate": 0.15, "avg_amount": 800.0, "stddev_amount": 2000.0},
    "travel": {"fraud_rate": 0.10, "avg_amount": 450.0, "stddev_amount": 600.0},
    "pharmacy": {"fraud_rate": 0.02, "avg_amount": 35.0, "stddev_amount": 25.0},
    "convenience_store": {"fraud_rate": 0.04, "avg_amount": 15.0, "stddev_amount": 12.0},
    "department_store": {"fraud_rate": 0.06, "avg_amount": 150.0, "stddev_amount": 200.0},
}

TERMINAL_RISK_MAP = {
    "chip": 0.02,        # EMV chip = lowest risk
    "contactless": 0.03, # NFC = slightly higher (lost phone risk)
    "swipe": 0.08,       # Magnetic stripe = cloneable
    "online": 0.12,      # Card-not-present = highest risk
}


# =============================================================================
# Per-Element Feature Computation
# =============================================================================

class FeatureComputer:
    """
    Computes per-element features for a single transaction.
    
    These features do not require cross-transaction state and can be
    computed independently for each event in the stream.
    """

    def compute_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all per-element features for a transaction.
        
        Args:
            transaction: Raw transaction dict from Pub/Sub
            
        Returns:
            Dict of feature name → feature value
        """
        amount = transaction.get("amount", 0.0)
        category = transaction.get("merchant_category", "other")
        terminal_type = transaction.get("terminal_type", "chip")
        timestamp_str = transaction.get("timestamp", "")

        features = {
            # Amount-based features
            "f_amount_log": self._log_transform(amount),
            "f_amount_zscore": self._category_zscore(amount, category),
            "f_is_round_amount": self._is_round_amount(amount),
            "f_amount_bucket": self._amount_bucket(amount),

            # Risk profile features
            "f_merchant_risk_score": self._merchant_risk(category),
            "f_terminal_type_risk": TERMINAL_RISK_MAP.get(terminal_type, 0.05),
            "f_is_high_risk_category": category in ("jewelry", "electronics", "online_retail"),

            # Time-based features
            "f_hour_of_day": self._extract_hour(timestamp_str),
            "f_is_night_transaction": self._is_night(timestamp_str),
            "f_is_weekend": self._is_weekend(timestamp_str),
            "f_day_of_week": self._day_of_week(timestamp_str),

            # Behavioral indicators
            "f_is_international": transaction.get("is_international", False),
            "f_pin_entered": transaction.get("pin_entered", True),
            "f_is_recurring": transaction.get("recurring", False),
            "f_is_card_not_present": (
                terminal_type == "online" and not transaction.get("pin_entered", True)
            ),

            # Composite risk signals
            "f_amount_exceeds_category_3sigma": (
                amount > self._category_threshold(category, sigma=3)
            ),
            "f_high_risk_no_pin": (
                TERMINAL_RISK_MAP.get(terminal_type, 0.05) > 0.05
                and not transaction.get("pin_entered", True)
            ),
        }

        return features

    @staticmethod
    def _log_transform(amount: float) -> float:
        """Log1p transform for amount (handles zero gracefully)."""
        return round(math.log1p(max(0, amount)), 4)

    @staticmethod
    def _is_round_amount(amount: float) -> bool:
        """Detect round amounts (fraudsters often use round numbers)."""
        return amount >= 100 and amount == int(amount)

    @staticmethod
    def _amount_bucket(amount: float) -> str:
        """Discretize amount into interpretable buckets."""
        if amount < 25:
            return "micro"
        elif amount < 100:
            return "small"
        elif amount < 500:
            return "medium"
        elif amount < 2000:
            return "large"
        elif amount < 10000:
            return "very_large"
        return "extreme"

    @staticmethod
    def _merchant_risk(category: str) -> float:
        """Historical fraud rate for the merchant category."""
        profile = CATEGORY_RISK_PROFILES.get(category, {"fraud_rate": 0.05})
        return profile["fraud_rate"]

    @staticmethod
    def _category_zscore(amount: float, category: str) -> float:
        """Z-score of transaction amount relative to category distribution."""
        profile = CATEGORY_RISK_PROFILES.get(
            category, {"avg_amount": 100.0, "stddev_amount": 100.0}
        )
        stddev = profile["stddev_amount"]
        if stddev == 0:
            return 0.0
        return round((amount - profile["avg_amount"]) / stddev, 3)

    @staticmethod
    def _category_threshold(category: str, sigma: float = 3) -> float:
        """Amount threshold at N sigma above category mean."""
        profile = CATEGORY_RISK_PROFILES.get(
            category, {"avg_amount": 100.0, "stddev_amount": 100.0}
        )
        return profile["avg_amount"] + sigma * profile["stddev_amount"]

    @staticmethod
    def _extract_hour(timestamp_str: str) -> int:
        """Extract hour of day (0-23) from ISO timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.hour
        except (ValueError, AttributeError):
            return 12

    @staticmethod
    def _is_night(timestamp_str: str) -> bool:
        """Flag transactions between 11 PM and 5 AM."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.hour >= 23 or dt.hour < 5
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _is_weekend(timestamp_str: str) -> bool:
        """Flag weekend transactions (different fraud patterns)."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.weekday() >= 5
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _day_of_week(timestamp_str: str) -> int:
        """Day of week (0=Monday, 6=Sunday)."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.weekday()
        except (ValueError, AttributeError):
            return 0


# =============================================================================
# Windowed Velocity Features (Redis-backed for API path)
# =============================================================================

class VelocityFeatureStore:
    """
    Windowed velocity features backed by Redis/Memorystore.
    
    In the Dataflow pipeline, velocity is computed using Beam's
    SlidingWindows + GroupByKey. For the Cloud Run API (on-demand scoring),
    we query a Redis feature store that's continuously updated by Dataflow.
    
    This class abstracts the Redis interaction for feature retrieval.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """
        Initialize Redis connection for velocity feature lookups.
        
        In production, redis_host points to a Memorystore instance.
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self._client = None

    def _get_client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            import redis
            self._client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
            )
        return self._client

    def get_velocity_features(self, card_token: str) -> Dict[str, Any]:
        """
        Retrieve pre-computed velocity features for a card token.
        
        Returns features written by the Dataflow pipeline:
        - velocity_5min_count
        - velocity_5min_total_amount
        - velocity_30min_count
        - velocity_30min_total_amount
        - distinct_merchants_1hr
        - last_txn_region
        - seconds_since_last_txn
        """
        try:
            client = self._get_client()
            key = f"velocity:{card_token}"
            data = client.hgetall(key)

            if not data:
                return self._default_velocity_features()

            return {
                "f_velocity_5min_count": int(data.get("5min_count", 1)),
                "f_velocity_5min_total_amount": float(data.get("5min_amount", 0)),
                "f_velocity_30min_count": int(data.get("30min_count", 1)),
                "f_velocity_30min_total_amount": float(data.get("30min_amount", 0)),
                "f_distinct_merchants_1hr": int(data.get("merchants_1hr", 1)),
                "f_last_txn_region": data.get("last_region", "unknown"),
                "f_seconds_since_last_txn": float(data.get("seconds_since_last", 3600)),
            }

        except Exception as e:
            logger.warning(f"Redis velocity lookup failed for {card_token}: {str(e)}")
            return self._default_velocity_features()

    @staticmethod
    def _default_velocity_features() -> Dict[str, Any]:
        """Safe defaults when velocity data is unavailable."""
        return {
            "f_velocity_5min_count": 1,
            "f_velocity_5min_total_amount": 0.0,
            "f_velocity_30min_count": 1,
            "f_velocity_30min_total_amount": 0.0,
            "f_distinct_merchants_1hr": 1,
            "f_last_txn_region": "unknown",
            "f_seconds_since_last_txn": 3600.0,
        }
