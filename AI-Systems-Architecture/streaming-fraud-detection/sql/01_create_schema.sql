-- =============================================================================
-- BigQuery Schema: Real-Time Fraud Detection Data Warehouse
-- =============================================================================
-- Creates the dataset and tables that receive streaming inserts from
-- the Dataflow pipeline. Partitioned by event_timestamp for cost-efficient
-- queries and clustered by card_token / risk_tier for fast lookups.
-- =============================================================================

-- Create dataset
CREATE SCHEMA IF NOT EXISTS `fraud_detection`
OPTIONS (
  description = 'Real-time POS fraud detection data warehouse',
  location = 'us-central1',
  default_table_expiration_days = 365
);

-- =============================================================================
-- Table 1: Enriched Transactions (streaming insert target from Dataflow)
-- =============================================================================
CREATE OR REPLACE TABLE `fraud_detection.enriched_transactions` (
  -- Core transaction fields
  transaction_id STRING NOT NULL,
  event_timestamp TIMESTAMP NOT NULL,
  card_token STRING NOT NULL,
  merchant_id STRING,
  merchant_category STRING,
  terminal_id STRING,
  terminal_type STRING,
  amount FLOAT64 NOT NULL,
  currency STRING DEFAULT 'USD',
  region STRING,
  is_international BOOLEAN DEFAULT FALSE,
  pin_entered BOOLEAN,
  recurring BOOLEAN DEFAULT FALSE,

  -- Computed features (from Dataflow pipeline)
  f_amount_log FLOAT64,
  f_is_round_amount BOOLEAN,
  f_merchant_risk_score FLOAT64,
  f_terminal_type_risk FLOAT64,
  f_hour_of_day INT64,
  f_is_night_transaction BOOLEAN,
  f_amount_bucket STRING,
  f_velocity_count INT64,
  f_velocity_total_amount FLOAT64,
  f_velocity_distinct_merchants INT64,

  -- Model scoring results
  fraud_probability_score FLOAT64,
  risk_decision STRING,  -- APPROVE | REVIEW | FLAG
  risk_tier STRING,      -- low | medium | high

  -- Pipeline metadata
  processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  scoring_latency_ms FLOAT64,
  pii_sanitized BOOLEAN DEFAULT TRUE,

  -- Ground truth (populated async by fraud investigation team)
  label_confirmed_fraud BOOLEAN,
  label_updated_at TIMESTAMP,
  investigation_notes STRING
)
PARTITION BY DATE(event_timestamp)
CLUSTER BY card_token, risk_tier, merchant_category
OPTIONS (
  description = 'Enriched POS transactions with real-time fraud scores from Dataflow pipeline',
  labels = [("team", "fraud-detection"), ("env", "production")]
);

-- =============================================================================
-- Table 2: Dead Letter Queue (malformed/unparseable messages)
-- =============================================================================
CREATE OR REPLACE TABLE `fraud_detection.dead_letter_transactions` (
  raw_data BYTES,
  error STRING,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
OPTIONS (
  description = 'Dead letter queue for malformed POS transaction messages from Pub/Sub',
  partition_expiration_days = 30
);

-- =============================================================================
-- Table 3: Model Performance Metrics (populated by evaluation jobs)
-- =============================================================================
CREATE OR REPLACE TABLE `fraud_detection.model_metrics` (
  evaluation_timestamp TIMESTAMP NOT NULL,
  model_version STRING NOT NULL,
  metric_name STRING NOT NULL,     -- precision, recall, f1, auc_roc, auc_pr
  metric_value FLOAT64 NOT NULL,
  threshold FLOAT64,
  dataset_size INT64,
  fraud_rate FLOAT64,
  notes STRING
)
PARTITION BY DATE(evaluation_timestamp)
OPTIONS (
  description = 'Historical model performance metrics for drift detection and reporting'
);

-- =============================================================================
-- Table 4: Anomaly Alerts (population-level anomalies from BQML)
-- =============================================================================
CREATE OR REPLACE TABLE `fraud_detection.anomaly_alerts` (
  alert_id STRING NOT NULL,
  detected_at TIMESTAMP NOT NULL,
  anomaly_type STRING,       -- volume_spike | score_drift | new_pattern | geo_cluster
  severity STRING,           -- info | warning | critical
  affected_dimension STRING, -- merchant_category | region | card_cohort
  affected_value STRING,
  expected_value FLOAT64,
  actual_value FLOAT64,
  deviation_pct FLOAT64,
  acknowledged BOOLEAN DEFAULT FALSE,
  resolved_at TIMESTAMP
)
PARTITION BY DATE(detected_at)
OPTIONS (
  description = 'Population-level anomaly alerts detected by scheduled BQML jobs'
);
