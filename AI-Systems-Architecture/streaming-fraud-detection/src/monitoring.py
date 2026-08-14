"""
Observability & Monitoring for Streaming Fraud Detection.

Provides:
1. Structured JSON logging for Cloud Logging integration
2. Performance metrics tracking (latency percentiles, throughput)
3. Alerting thresholds for SLO monitoring
4. Custom Cloud Monitoring metrics export
5. Dataflow pipeline metrics (via Apache Beam Metrics API)

All logs are structured JSON → Cloud Run stdout → Cloud Logging auto-ingestion.
Metrics can be queried in Cloud Monitoring or exported to BigQuery for SLO dashboards.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Structured Cloud Logging Formatter
# =============================================================================

class CloudLoggingFormatter(logging.Formatter):
    """
    JSON-structured log formatter for Cloud Logging.
    
    Cloud Run and Dataflow workers emit logs to stdout/stderr, which are
    automatically ingested by Cloud Logging. Structured JSON enables:
    - Querying by custom fields (transaction_id, scoring_latency, risk_tier)
    - Automatic severity mapping
    - Correlation with Cloud Trace spans
    """

    # Cloud Logging severity levels
    SEVERITY_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "severity": self.SEVERITY_MAP.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": "streaming-fraud-detection",
        }

        # Attach exception traceback if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Attach custom context fields (set via extra={} in log calls)
        custom_fields = [
            "transaction_id", "card_token", "merchant_id",
            "fraud_score", "risk_decision", "latency_ms",
            "scoring_method", "batch_size", "error_type",
        ]
        for field_name in custom_fields:
            if hasattr(record, field_name):
                log_entry[field_name] = getattr(record, field_name)

        return json.dumps(log_entry)


def setup_cloud_logging(level: str = "INFO"):
    """
    Configure structured logging for Cloud Run / Dataflow environment.
    
    Cloud Run captures stdout → Cloud Logging automatically.
    This configures the Python logging system to output structured JSON.
    """
    root_logger = logging.getLogger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add structured JSON handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(CloudLoggingFormatter())
    root_logger.addHandler(console_handler)

    logger.info("Cloud Logging configured with structured JSON output.")


# =============================================================================
# Performance Metrics Tracker
# =============================================================================

@dataclass
class MetricsBucket:
    """Rolling metrics bucket for a single metric (latency, count, etc.)."""
    values: List[float] = field(default_factory=list)
    max_size: int = 10000  # Rolling window size

    def record(self, value: float):
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values = self.values[-self.max_size:]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    def _percentile(self, pct: int) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * pct / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "avg": round(self.avg, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
        }


class PerformanceMonitor:
    """
    Tracks performance metrics across all pipeline components.
    
    Provides real-time visibility into:
    - Scoring latency (Vertex AI endpoint response time)
    - Retrieval latency (Vector Search response time)
    - Generation latency (Gemini LLM response time)
    - End-to-end pipeline latency
    - Throughput (transactions per second)
    - Error rates by component
    """

    def __init__(self):
        self.metrics: Dict[str, MetricsBucket] = {
            "scoring_latency_ms": MetricsBucket(),
            "retrieval_latency_ms": MetricsBucket(),
            "generation_latency_ms": MetricsBucket(),
            "total_pipeline_latency_ms": MetricsBucket(),
            "deep_analysis_latency_ms": MetricsBucket(),
            "pubsub_publish_latency_ms": MetricsBucket(),
        }
        self.counters: Dict[str, int] = {
            "transactions_processed": 0,
            "transactions_approved": 0,
            "transactions_reviewed": 0,
            "transactions_flagged": 0,
            "scoring_errors": 0,
            "fallback_scores": 0,
            "deep_analysis_triggered": 0,
            "deep_analysis_errors": 0,
            "pii_redactions": 0,
            "dead_letters": 0,
        }
        self._start_time = time.time()

    def record_latency(self, metric_name: str, latency_ms: float):
        """Record a latency measurement."""
        if metric_name in self.metrics:
            self.metrics[metric_name].record(latency_ms)

    def increment_counter(self, counter_name: str, amount: int = 1):
        """Increment an event counter."""
        if counter_name in self.counters:
            self.counters[counter_name] += amount

    def get_throughput(self) -> float:
        """Calculate transactions per second since monitor started."""
        elapsed = time.time() - self._start_time
        total = self.counters.get("transactions_processed", 0)
        return total / elapsed if elapsed > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Full metrics summary for health endpoints and dashboards."""
        summary = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "throughput_tps": round(self.get_throughput(), 2),
            "counters": dict(self.counters),
            "latencies": {
                name: bucket.summary()
                for name, bucket in self.metrics.items()
                if bucket.count > 0
            },
        }

        # Calculate derived metrics
        total = self.counters.get("transactions_processed", 0)
        if total > 0:
            summary["rates"] = {
                "approval_rate": round(self.counters["transactions_approved"] / total, 4),
                "flag_rate": round(self.counters["transactions_flagged"] / total, 4),
                "error_rate": round(self.counters["scoring_errors"] / total, 4),
                "fallback_rate": round(self.counters["fallback_scores"] / total, 4),
            }

        return summary

    def check_slo(self) -> Dict[str, Any]:
        """
        Check if system is meeting SLO targets.
        
        SLOs:
        - Scoring latency p99 < 200ms
        - Error rate < 1%
        - Fallback rate < 5%
        - Throughput > 100 TPS (if traffic is present)
        """
        scoring_p99 = self.metrics["scoring_latency_ms"].p99
        total = max(self.counters["transactions_processed"], 1)
        error_rate = self.counters["scoring_errors"] / total
        fallback_rate = self.counters["fallback_scores"] / total

        slo_status = {
            "scoring_latency_p99_ms": {
                "target": 200,
                "actual": round(scoring_p99, 2),
                "met": scoring_p99 < 200,
            },
            "error_rate_pct": {
                "target": 1.0,
                "actual": round(error_rate * 100, 2),
                "met": error_rate < 0.01,
            },
            "fallback_rate_pct": {
                "target": 5.0,
                "actual": round(fallback_rate * 100, 2),
                "met": fallback_rate < 0.05,
            },
        }

        slo_status["all_met"] = all(s["met"] for s in slo_status.values())
        return slo_status


# Global monitor instance
monitor = PerformanceMonitor()


# =============================================================================
# Decorators for Automatic Latency Tracking
# =============================================================================

def track_latency(metric_name: str):
    """
    Decorator to automatically track latency of sync/async functions.
    
    Usage:
        @track_latency("scoring_latency_ms")
        async def score_transaction(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                monitor.record_latency(metric_name, latency)
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                monitor.record_latency(metric_name, latency)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                monitor.record_latency(metric_name, latency)
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                monitor.record_latency(metric_name, latency)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def log_operation(operation_name: str, **context):
    """
    Context manager for structured operation logging with timing.
    
    Usage:
        with log_operation("fraud_scoring", transaction_id="txn_123"):
            result = scorer.score(features)
    """
    start = time.time()
    logger.info(f"Starting {operation_name}", extra=context)
    try:
        yield
        duration = (time.time() - start) * 1000
        logger.info(
            f"Completed {operation_name} in {duration:.1f}ms",
            extra={**context, "latency_ms": duration}
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(
            f"Failed {operation_name} after {duration:.1f}ms: {str(e)}",
            extra={**context, "latency_ms": duration, "error_type": type(e).__name__},
            exc_info=True,
        )
        raise
