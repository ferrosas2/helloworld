"""
POS Transaction Event Simulator & Pub/Sub Publisher.

Simulates high-velocity retail point-of-sale transaction events and publishes
them to a Google Cloud Pub/Sub topic for downstream consumption by the
Dataflow streaming pipeline.

Usage:
    python pipeline/pubsub_publisher.py --num-events=1000 --rate=100
    python pipeline/pubsub_publisher.py --continuous --rate=50

Features:
    - Realistic transaction generation with configurable fraud injection rate
    - Adjustable publishing rate (events/second)
    - Supports burst mode to simulate peak traffic (e.g., Black Friday)
    - Pub/Sub message attributes for efficient filtering downstream
"""

import argparse
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from google.cloud import pubsub_v1
from google.api_core import retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Transaction Generation Configuration
# =============================================================================

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "gas_station", "restaurant",
    "online_retail", "jewelry", "travel", "pharmacy",
    "convenience_store", "department_store"
]

REGIONS = [
    "us-east-1", "us-west-2", "us-central-1",
    "eu-west-1", "eu-central-1", "ap-southeast-1"
]

# Fraud patterns for realistic simulation
FRAUD_PATTERNS = {
    "velocity_burst": {
        "description": "Multiple transactions in rapid succession from same card",
        "amount_range": (50, 500),
        "time_gap_seconds": 5,
    },
    "geo_impossible": {
        "description": "Card used in geographically distant locations within minutes",
        "amount_range": (100, 2000),
        "region_hop": True,
    },
    "high_value_anomaly": {
        "description": "Transaction amount far exceeds cardholder's typical spending",
        "amount_range": (5000, 50000),
        "category": "jewelry",
    },
    "card_testing": {
        "description": "Small sequential amounts to test stolen card validity",
        "amount_range": (1, 10),
        "rapid_fire": True,
    },
}


# =============================================================================
# Transaction Generator
# =============================================================================

class POSTransactionGenerator:
    """
    Generates realistic POS transaction events with configurable fraud injection.
    
    Produces transactions that mirror real retail POS data including:
    - Card token (hashed, no raw PAN)
    - Merchant details (ID, category, location)
    - Transaction metadata (amount, currency, timestamp)
    - Terminal information (ID, type)
    """

    def __init__(self, fraud_rate: float = 0.05):
        """
        Args:
            fraud_rate: Proportion of transactions that are fraudulent (default 5%)
        """
        self.fraud_rate = fraud_rate
        self.card_pool = [f"card_token_{uuid.uuid4().hex[:16]}" for _ in range(200)]
        self.merchant_pool = [f"merchant_{uuid.uuid4().hex[:8]}" for _ in range(50)]
        self.terminal_pool = [f"terminal_{uuid.uuid4().hex[:6]}" for _ in range(100)]
        logger.info(f"Initialized generator: {len(self.card_pool)} cards, "
                    f"{len(self.merchant_pool)} merchants, fraud_rate={fraud_rate:.1%}")

    def generate_legitimate_transaction(self) -> Dict[str, Any]:
        """Generate a normal, legitimate POS transaction."""
        return {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "card_token": random.choice(self.card_pool),
            "merchant_id": random.choice(self.merchant_pool),
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "terminal_id": random.choice(self.terminal_pool),
            "terminal_type": random.choice(["chip", "contactless", "swipe", "online"]),
            "amount": round(random.uniform(5.00, 500.00), 2),
            "currency": "USD",
            "region": random.choice(REGIONS),
            "is_international": random.random() < 0.08,
            "pin_entered": random.random() > 0.3,
            "recurring": random.random() < 0.15,
            # Ground truth label (for model evaluation — not available in real-time)
            "_label_fraud": False,
            "_fraud_pattern": None,
        }

    def generate_fraudulent_transaction(self) -> Dict[str, Any]:
        """Generate a fraudulent transaction matching known fraud patterns."""
        pattern_name = random.choice(list(FRAUD_PATTERNS.keys()))
        pattern = FRAUD_PATTERNS[pattern_name]

        amount_low, amount_high = pattern["amount_range"]

        txn = {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "card_token": random.choice(self.card_pool),
            "merchant_id": random.choice(self.merchant_pool),
            "merchant_category": pattern.get("category", random.choice(MERCHANT_CATEGORIES)),
            "terminal_id": random.choice(self.terminal_pool),
            "terminal_type": random.choice(["chip", "contactless", "swipe", "online"]),
            "amount": round(random.uniform(amount_low, amount_high), 2),
            "currency": "USD",
            "region": random.choice(REGIONS),
            "is_international": pattern_name == "geo_impossible" or random.random() < 0.3,
            "pin_entered": random.random() < 0.2,  # Fraudsters less likely to know PIN
            "recurring": False,
            # Ground truth label
            "_label_fraud": True,
            "_fraud_pattern": pattern_name,
        }
        return txn

    def generate_event(self) -> Dict[str, Any]:
        """Generate a single transaction event (legitimate or fraudulent)."""
        if random.random() < self.fraud_rate:
            return self.generate_fraudulent_transaction()
        return self.generate_legitimate_transaction()


# =============================================================================
# Pub/Sub Publisher
# =============================================================================

class TransactionPublisher:
    """
    Publishes POS transaction events to Google Cloud Pub/Sub.
    
    Implements batching and flow control for high-throughput publishing.
    Uses futures for async publishing with error tracking.
    """

    def __init__(self, project_id: str, topic_id: str):
        """
        Args:
            project_id: GCP project ID
            topic_id: Pub/Sub topic name (e.g., 'pos-transactions')
        """
        self.topic_path = f"projects/{project_id}/topics/{topic_id}"

        # Configure batching for high-throughput publishing
        batch_settings = pubsub_v1.types.BatchSettings(
            max_messages=100,        # Batch up to 100 messages
            max_bytes=1024 * 1024,   # Or 1MB, whichever comes first
            max_latency=0.1,         # Flush every 100ms
        )

        # Flow control to prevent memory issues during bursts
        flow_control = pubsub_v1.types.PublishFlowControl(
            message_limit=1000,
            byte_limit=10 * 1024 * 1024,  # 10MB buffer
            limit_exceeded_behavior=pubsub_v1.types.LimitExceededBehavior.BLOCK,
        )

        self.publisher = pubsub_v1.PublisherClient(
            batch_settings=batch_settings,
            publisher_options=pubsub_v1.types.PublisherOptions(
                flow_control=flow_control,
            ),
        )

        self.published_count = 0
        self.error_count = 0
        logger.info(f"Publisher initialized for topic: {self.topic_path}")

    def _publish_callback(self, future):
        """Callback for async publish results."""
        try:
            message_id = future.result(timeout=30)
            self.published_count += 1
        except Exception as e:
            self.error_count += 1
            logger.error(f"Publish failed: {str(e)}")

    def publish_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Publish a single transaction event to Pub/Sub.
        
        Message attributes are set for efficient server-side filtering:
        - region: enables regional subscription filters
        - merchant_category: enables category-specific consumers
        - risk_tier: pre-classification hint (based on amount thresholds)
        """
        # Serialize transaction payload
        message_data = json.dumps(transaction).encode("utf-8")

        # Set message attributes for downstream filtering
        attributes = {
            "region": transaction.get("region", "unknown"),
            "merchant_category": transaction.get("merchant_category", "unknown"),
            "terminal_type": transaction.get("terminal_type", "unknown"),
            "risk_tier": self._classify_risk_tier(transaction),
            "event_time": transaction.get("timestamp", ""),
        }

        # Publish asynchronously with retry
        future = self.publisher.publish(
            self.topic_path,
            data=message_data,
            **attributes,
        )
        future.add_done_callback(self._publish_callback)

    def _classify_risk_tier(self, transaction: Dict[str, Any]) -> str:
        """Pre-classify risk tier based on simple heuristics for routing."""
        amount = transaction.get("amount", 0)
        if amount > 5000:
            return "high"
        elif amount > 1000:
            return "medium"
        return "low"

    def flush(self) -> None:
        """Flush any pending messages in the batch."""
        self.publisher.transport.close()
        logger.info(f"Publisher flushed. Published: {self.published_count}, Errors: {self.error_count}")

    def get_stats(self) -> Dict[str, int]:
        """Return publishing statistics."""
        return {
            "published": self.published_count,
            "errors": self.error_count,
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_publisher(
    project_id: str,
    topic_id: str,
    num_events: int = 1000,
    rate: int = 100,
    fraud_rate: float = 0.05,
    continuous: bool = False,
    burst_mode: bool = False,
):
    """
    Run the POS event publisher.
    
    Args:
        project_id: GCP project ID
        topic_id: Pub/Sub topic name
        num_events: Number of events to publish (ignored if continuous=True)
        rate: Events per second
        fraud_rate: Fraction of events that are fraudulent
        continuous: Run indefinitely
        burst_mode: Simulate traffic bursts (2-5x normal rate for 10s intervals)
    """
    generator = POSTransactionGenerator(fraud_rate=fraud_rate)
    publisher = TransactionPublisher(project_id=project_id, topic_id=topic_id)

    interval = 1.0 / rate
    events_sent = 0
    start_time = time.time()

    logger.info(f"Starting publisher: rate={rate}/s, fraud_rate={fraud_rate:.1%}, "
                f"{'continuous' if continuous else f'num_events={num_events}'}")

    try:
        while continuous or events_sent < num_events:
            # Burst mode: periodically increase rate to simulate peak traffic
            current_interval = interval
            if burst_mode and random.random() < 0.05:
                burst_multiplier = random.uniform(2, 5)
                current_interval = interval / burst_multiplier
                logger.info(f"BURST MODE: {burst_multiplier:.1f}x rate for next batch")

            transaction = generator.generate_event()
            publisher.publish_transaction(transaction)
            events_sent += 1

            if events_sent % 500 == 0:
                elapsed = time.time() - start_time
                actual_rate = events_sent / elapsed if elapsed > 0 else 0
                stats = publisher.get_stats()
                logger.info(
                    f"Progress: {events_sent} events sent | "
                    f"Actual rate: {actual_rate:.1f}/s | "
                    f"Published: {stats['published']} | "
                    f"Errors: {stats['errors']}"
                )

            time.sleep(current_interval)

    except KeyboardInterrupt:
        logger.info("Publisher interrupted by user.")
    finally:
        elapsed = time.time() - start_time
        publisher.flush()
        logger.info(
            f"Publisher complete. Total events: {events_sent}, "
            f"Duration: {elapsed:.1f}s, "
            f"Avg rate: {events_sent / elapsed:.1f}/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="POS Transaction Event Simulator & Pub/Sub Publisher"
    )
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--topic-id", default="pos-transactions", help="Pub/Sub topic name")
    parser.add_argument("--num-events", type=int, default=1000, help="Number of events to publish")
    parser.add_argument("--rate", type=int, default=100, help="Events per second")
    parser.add_argument("--fraud-rate", type=float, default=0.05, help="Fraction of fraudulent events")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--burst-mode", action="store_true", help="Simulate traffic bursts")

    args = parser.parse_args()

    run_publisher(
        project_id=args.project_id,
        topic_id=args.topic_id,
        num_events=args.num_events,
        rate=args.rate,
        fraud_rate=args.fraud_rate,
        continuous=args.continuous,
        burst_mode=args.burst_mode,
    )
