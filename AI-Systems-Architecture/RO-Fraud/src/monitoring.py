"""
Monitoring and observability utilities for production GCP deployment.
Integrates with Cloud Logging, Cloud Monitoring, and provides metrics tracking.
"""
import logging
import time
from functools import wraps
from typing import Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Tracks performance metrics for RAG pipeline components.
    In production, these metrics should be exported to Cloud Monitoring.
    """
    
    def __init__(self):
        self.metrics = {
            "retrieval_latency": [],
            "generation_latency": [],
            "total_request_latency": [],
            "vector_search_results_count": []
        }
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value for monitoring."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            logger.info(f"Metric recorded: {metric_name}={value:.3f}")
    
    def get_average(self, metric_name: str) -> float:
        """Calculate average for a given metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return 0.0
    
    def get_summary(self) -> dict:
        """Get summary statistics for all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        return summary

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def track_latency(operation_name: str):
    """
    Decorator to track latency of operations.
    Usage: @track_latency("vector_search")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                performance_monitor.record_metric(f"{operation_name}_latency", latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                logger.error(f"{operation_name} failed after {latency:.3f}s: {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                performance_monitor.record_metric(f"{operation_name}_latency", latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                logger.error(f"{operation_name} failed after {latency:.3f}s: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

@contextmanager
def log_operation(operation_name: str, **context):
    """
    Context manager for structured logging of operations.
    
    Usage:
        with log_operation("fraud_analysis", claim_id="C-1001"):
            # perform operation
            pass
    """
    start_time = time.time()
    logger.info(f"Starting {operation_name}", extra=context)
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} in {duration:.3f}s", extra=context)
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation_name} after {duration:.3f}s: {str(e)}", 
            extra=context,
            exc_info=True
        )
        raise

class CloudLoggingFormatter(logging.Formatter):
    """
    Custom formatter for Cloud Logging structured logs.
    Formats logs in JSON-compatible structure for better querying in Cloud Logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        if hasattr(record, "claim_id"):
            log_obj["claim_id"] = record.claim_id
        if hasattr(record, "customer_id"):
            log_obj["customer_id"] = record.customer_id
        
        import json
        return json.dumps(log_obj)

def setup_cloud_logging():
    """
    Configure logging for Cloud Run environment with structured logging.
    Cloud Run automatically captures stdout/stderr and sends to Cloud Logging.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with Cloud Logging format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CloudLoggingFormatter())
    root_logger.addHandler(console_handler)
    
    logger.info("Cloud Logging configured successfully")
