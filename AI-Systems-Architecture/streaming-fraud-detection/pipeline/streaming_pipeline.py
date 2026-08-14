"""
Real-Time Streaming Fraud Detection Pipeline (Apache Beam / Cloud Dataflow).

This is the core streaming pipeline that:
1. Consumes POS transaction events from Cloud Pub/Sub
2. Sanitizes PII fields using regex-based redaction
3. Computes real-time features (velocity, behavioral, per-element)
4. Scores each transaction via Vertex AI Online Prediction
5. Routes transactions by risk tier (approve / review / flag)
6. Writes all enriched transactions to BigQuery (streaming insert)
7. Publishes high-risk flagged transactions to a Pub/Sub topic for
   downstream deep analysis (Cloud Run RAG service)

Usage:
    # Local runner (testing)
    python pipeline/streaming_pipeline.py \
        --project=YOUR_PROJECT \
        --region=us-central1 \
        --runner=DirectRunner

    # Dataflow runner (production)
    python pipeline/streaming_pipeline.py \
        --project=YOUR_PROJECT \
        --region=us-central1 \
        --runner=DataflowRunner \
        --temp_location=gs://YOUR_BUCKET/temp \
        --staging_location=gs://YOUR_BUCKET/staging \
        --streaming \
        --num_workers=3 \
        --max_num_workers=20 \
        --autoscaling_algorithm=THROUGHPUT_BASED
"""

import argparse
import json
import logging
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    StandardOptions,
    GoogleCloudOptions,
    WorkerOptions,
    SetupOptions,
)
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition
from apache_beam.io.gcp.pubsub import ReadFromPubSub, WriteToPubSub

from transforms import (
    ParseTransaction,
    SanitizePII,
    EnrichFeatures,
    ScoreTransaction,
    RouteByRisk,
    FormatForBigQuery,
    ComputeWindowedVelocity,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# BigQuery Table Schema
# =============================================================================

TRANSACTIONS_TABLE_SCHEMA = {
    "fields": [
        {"name": "transaction_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "event_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "card_token", "type": "STRING", "mode": "REQUIRED"},
        {"name": "merchant_id", "type": "STRING", "mode": "NULLABLE"},
        {"name": "merchant_category", "type": "STRING", "mode": "NULLABLE"},
        {"name": "terminal_id", "type": "STRING", "mode": "NULLABLE"},
        {"name": "terminal_type", "type": "STRING", "mode": "NULLABLE"},
        {"name": "amount", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "currency", "type": "STRING", "mode": "NULLABLE"},
        {"name": "region", "type": "STRING", "mode": "NULLABLE"},
        {"name": "is_international", "type": "BOOLEAN", "mode": "NULLABLE"},
        {"name": "pin_entered", "type": "BOOLEAN", "mode": "NULLABLE"},
        {"name": "recurring", "type": "BOOLEAN", "mode": "NULLABLE"},
        # Computed features
        {"name": "f_amount_log", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "f_is_round_amount", "type": "BOOLEAN", "mode": "NULLABLE"},
        {"name": "f_merchant_risk_score", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "f_terminal_type_risk", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "f_hour_of_day", "type": "INT64", "mode": "NULLABLE"},
        {"name": "f_is_night_transaction", "type": "BOOLEAN", "mode": "NULLABLE"},
        {"name": "f_amount_bucket", "type": "STRING", "mode": "NULLABLE"},
        # Scoring results
        {"name": "fraud_probability_score", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "risk_decision", "type": "STRING", "mode": "NULLABLE"},
        {"name": "risk_tier", "type": "STRING", "mode": "NULLABLE"},
        # Metadata
        {"name": "processing_timestamp", "type": "TIMESTAMP", "mode": "NULLABLE"},
        {"name": "scoring_latency_ms", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "pii_sanitized", "type": "BOOLEAN", "mode": "NULLABLE"},
    ]
}

DEAD_LETTER_TABLE_SCHEMA = {
    "fields": [
        {"name": "raw_data", "type": "BYTES", "mode": "NULLABLE"},
        {"name": "error", "type": "STRING", "mode": "NULLABLE"},
        {"name": "timestamp", "type": "TIMESTAMP", "mode": "NULLABLE"},
    ]
}


# =============================================================================
# Pipeline Options Configuration
# =============================================================================

class FraudDetectionOptions(PipelineOptions):
    """Custom pipeline options for fraud detection configuration."""

    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--input_subscription",
            default="projects/YOUR_PROJECT/subscriptions/pos-transactions-sub",
            help="Pub/Sub subscription for POS transaction events",
        )
        parser.add_argument(
            "--output_table",
            default="YOUR_PROJECT:fraud_detection.enriched_transactions",
            help="BigQuery output table (project:dataset.table)",
        )
        parser.add_argument(
            "--dead_letter_table",
            default="YOUR_PROJECT:fraud_detection.dead_letter_transactions",
            help="BigQuery dead letter table for malformed messages",
        )
        parser.add_argument(
            "--flagged_topic",
            default="projects/YOUR_PROJECT/topics/flagged-transactions",
            help="Pub/Sub topic for high-risk flagged transactions",
        )
        parser.add_argument(
            "--vertex_endpoint_id",
            required=True,
            help="Vertex AI Online Prediction endpoint ID",
        )
        parser.add_argument(
            "--high_risk_threshold",
            type=float,
            default=0.7,
            help="Score threshold for flagging high-risk transactions",
        )
        parser.add_argument(
            "--medium_risk_threshold",
            type=float,
            default=0.3,
            help="Score threshold for medium-risk review queue",
        )
        parser.add_argument(
            "--velocity_window_seconds",
            type=int,
            default=300,
            help="Sliding window size for velocity features (default 5 min)",
        )


# =============================================================================
# Pipeline Construction
# =============================================================================

def build_pipeline(pipeline_options: PipelineOptions, fraud_options: FraudDetectionOptions):
    """
    Constructs and returns the streaming fraud detection pipeline.
    
    Pipeline DAG:
        Pub/Sub → Parse → Sanitize → Features → Score → Route
                    |                                       |
                    └→ Dead Letter (BQ)                    ├→ Approve (BQ)
                                                           ├→ Review (BQ)
                                                           └→ Flag (BQ + Pub/Sub)
    """
    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    project_id = google_cloud_options.project

    with beam.Pipeline(options=pipeline_options) as pipeline:

        # ─── Read from Pub/Sub ────────────────────────────────────────────
        raw_messages = (
            pipeline
            | "ReadFromPubSub" >> ReadFromPubSub(
                subscription=fraud_options.input_subscription,
                with_attributes=True,
            )
        )

        # ─── Parse & Validate ────────────────────────────────────────────
        parsed = (
            raw_messages
            | "ParseTransactions" >> beam.ParDo(ParseTransaction()).with_outputs(
                ParseTransaction.VALID_OUTPUT,
                ParseTransaction.DEAD_LETTER,
            )
        )

        valid_transactions = parsed[ParseTransaction.VALID_OUTPUT]
        dead_letters = parsed[ParseTransaction.DEAD_LETTER]

        # ─── Dead Letter → BigQuery ──────────────────────────────────────
        _ = (
            dead_letters
            | "WriteDeadLetters" >> WriteToBigQuery(
                table=fraud_options.dead_letter_table,
                schema=DEAD_LETTER_TABLE_SCHEMA,
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                method=WriteToBigQuery.Method.STREAMING_INSERTS,
            )
        )

        # ─── PII Sanitization ────────────────────────────────────────────
        sanitized = (
            valid_transactions
            | "SanitizePII" >> beam.ParDo(SanitizePII())
        )

        # ─── Feature Engineering (Per-Element) ───────────────────────────
        enriched = (
            sanitized
            | "EnrichFeatures" >> beam.ParDo(EnrichFeatures())
        )

        # ─── Fraud Scoring (Vertex AI Online Prediction) ─────────────────
        scored = (
            enriched
            | "ScoreTransaction" >> beam.ParDo(
                ScoreTransaction(
                    project_id=project_id,
                    endpoint_id=fraud_options.vertex_endpoint_id,
                    region=google_cloud_options.region or "us-central1",
                )
            )
        )

        # ─── Route by Risk Level ─────────────────────────────────────────
        routed = (
            scored
            | "RouteByRisk" >> beam.ParDo(
                RouteByRisk(
                    high_risk_threshold=fraud_options.high_risk_threshold,
                    medium_risk_threshold=fraud_options.medium_risk_threshold,
                )
            ).with_outputs(
                RouteByRisk.APPROVE,
                RouteByRisk.REVIEW,
                RouteByRisk.FLAG,
            )
        )

        approved = routed[RouteByRisk.APPROVE]
        review = routed[RouteByRisk.REVIEW]
        flagged = routed[RouteByRisk.FLAG]

        # ─── All transactions → BigQuery (streaming insert) ──────────────
        all_scored = (
            (approved, review, flagged)
            | "FlattenAllPaths" >> beam.Flatten()
        )

        _ = (
            all_scored
            | "FormatForBQ" >> beam.ParDo(FormatForBigQuery())
            | "WriteToBigQuery" >> WriteToBigQuery(
                table=fraud_options.output_table,
                schema=TRANSACTIONS_TABLE_SCHEMA,
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                method=WriteToBigQuery.Method.STREAMING_INSERTS,
            )
        )

        # ─── High-risk flagged → Pub/Sub (trigger deep analysis) ─────────
        _ = (
            flagged
            | "SerializeFlagged" >> beam.Map(lambda txn: json.dumps(txn).encode("utf-8"))
            | "PublishFlagged" >> WriteToPubSub(
                topic=fraud_options.flagged_topic,
            )
        )

        logger.info("Pipeline constructed successfully.")


# =============================================================================
# Entry Point
# =============================================================================

def run():
    """Parse arguments and launch the streaming pipeline."""
    parser = argparse.ArgumentParser(
        description="Real-Time Streaming Fraud Detection Pipeline"
    )

    # Parse known args (Beam will parse its own)
    known_args, pipeline_args = parser.parse_known_args()

    # Configure pipeline options
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = True

    # Setup options for Dataflow workers
    setup_options = pipeline_options.view_as(SetupOptions)
    setup_options.save_main_session = True

    # Worker options for autoscaling
    worker_options = pipeline_options.view_as(WorkerOptions)
    if not worker_options.num_workers:
        worker_options.num_workers = 3
    if not worker_options.max_num_workers:
        worker_options.max_num_workers = 20

    # Custom fraud detection options
    fraud_options = pipeline_options.view_as(FraudDetectionOptions)

    logger.info("=" * 60)
    logger.info("Starting Real-Time Streaming Fraud Detection Pipeline")
    logger.info(f"  Project: {pipeline_options.view_as(GoogleCloudOptions).project}")
    logger.info(f"  Region: {pipeline_options.view_as(GoogleCloudOptions).region}")
    logger.info(f"  Input: {fraud_options.input_subscription}")
    logger.info(f"  Output: {fraud_options.output_table}")
    logger.info(f"  Flagged Topic: {fraud_options.flagged_topic}")
    logger.info(f"  Vertex Endpoint: {fraud_options.vertex_endpoint_id}")
    logger.info(f"  High Risk Threshold: {fraud_options.high_risk_threshold}")
    logger.info(f"  Medium Risk Threshold: {fraud_options.medium_risk_threshold}")
    logger.info("=" * 60)

    # Build and run the pipeline
    build_pipeline(pipeline_options, fraud_options)


if __name__ == "__main__":
    run()
