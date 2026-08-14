-- =============================================================================
-- BigQuery ML: Population-Level Anomaly Detection
-- =============================================================================
-- Detects unusual patterns in transaction volume, fraud rates, and scoring
-- distributions using ARIMA_PLUS time-series anomaly detection.
--
-- This runs as a scheduled query (hourly) to catch:
--   - Sudden spikes in fraud rate for a merchant category
--   - Unusual transaction volumes by region (indicating compromised terminals)
--   - Score distribution drift (model degradation signal)
--   - Geographic clustering of high-risk transactions
--
-- Anomalies are written to fraud_detection.anomaly_alerts for ops dashboards.
-- =============================================================================


-- =============================================================================
-- Step 1: Train an ARIMA_PLUS model on hourly transaction volume by category
-- =============================================================================

CREATE OR REPLACE MODEL `fraud_detection.volume_anomaly_model`
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'hour_bucket',
  time_series_data_col = 'txn_count',
  time_series_id_col = 'merchant_category',
  data_frequency = 'AUTO_FREQUENCY',
  decompose_time_series = TRUE,
  auto_arima = TRUE,
  clean_spikes_and_dips = TRUE
) AS

SELECT
  TIMESTAMP_TRUNC(event_timestamp, HOUR) AS hour_bucket,
  merchant_category,
  COUNT(*) AS txn_count
FROM
  `fraud_detection.enriched_transactions`
WHERE
  event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY
  hour_bucket, merchant_category
HAVING
  txn_count > 0
ORDER BY
  hour_bucket;


-- =============================================================================
-- Step 2: Detect anomalies in recent transaction volume
-- =============================================================================

-- Detect anomalous volume spikes/drops per merchant category
SELECT
  merchant_category,
  hour_bucket AS anomaly_timestamp,
  txn_count AS actual_value,
  anomaly_probability,
  lower_bound,
  upper_bound,
  CASE
    WHEN txn_count > upper_bound THEN 'volume_spike'
    WHEN txn_count < lower_bound THEN 'volume_drop'
  END AS anomaly_type,
  CASE
    WHEN anomaly_probability > 0.99 THEN 'critical'
    WHEN anomaly_probability > 0.95 THEN 'warning'
    ELSE 'info'
  END AS severity
FROM
  ML.DETECT_ANOMALIES(
    MODEL `fraud_detection.volume_anomaly_model`,
    STRUCT(0.95 AS anomaly_prob_threshold)
  )
WHERE
  is_anomaly = TRUE
ORDER BY
  anomaly_probability DESC;


-- =============================================================================
-- Step 3: Detect fraud rate anomalies by region (statistical approach)
-- =============================================================================

-- Identify regions where recent fraud rate significantly exceeds the baseline
WITH regional_baselines AS (
  SELECT
    region,
    COUNT(*) AS total_txn,
    COUNTIF(fraud_probability_score >= 0.7) AS flagged_count,
    SAFE_DIVIDE(COUNTIF(fraud_probability_score >= 0.7), COUNT(*)) AS flag_rate
  FROM
    `fraud_detection.enriched_transactions`
  WHERE
    event_timestamp BETWEEN
      TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  GROUP BY region
),

recent_rates AS (
  SELECT
    region,
    COUNT(*) AS total_txn,
    COUNTIF(fraud_probability_score >= 0.7) AS flagged_count,
    SAFE_DIVIDE(COUNTIF(fraud_probability_score >= 0.7), COUNT(*)) AS flag_rate
  FROM
    `fraud_detection.enriched_transactions`
  WHERE
    event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  GROUP BY region
)

SELECT
  r.region,
  r.flag_rate AS current_flag_rate,
  b.flag_rate AS baseline_flag_rate,
  SAFE_DIVIDE(r.flag_rate - b.flag_rate, b.flag_rate) * 100 AS deviation_pct,
  r.total_txn AS recent_volume,
  CASE
    WHEN SAFE_DIVIDE(r.flag_rate - b.flag_rate, b.flag_rate) > 2.0 THEN 'critical'
    WHEN SAFE_DIVIDE(r.flag_rate - b.flag_rate, b.flag_rate) > 1.0 THEN 'warning'
    ELSE 'info'
  END AS severity
FROM recent_rates r
JOIN regional_baselines b ON r.region = b.region
WHERE
  -- Only alert if recent rate is significantly higher than baseline
  r.flag_rate > b.flag_rate * 1.5
  AND r.total_txn >= 50  -- Minimum sample size to avoid noise
ORDER BY
  deviation_pct DESC;


-- =============================================================================
-- Step 4: Model score distribution drift detection
-- =============================================================================

-- Compare current score distribution to historical baseline
-- A shift in distribution indicates model degradation or new attack patterns
WITH baseline_stats AS (
  SELECT
    AVG(fraud_probability_score) AS baseline_avg_score,
    STDDEV(fraud_probability_score) AS baseline_stddev,
    APPROX_QUANTILES(fraud_probability_score, 100)[OFFSET(50)] AS baseline_median,
    APPROX_QUANTILES(fraud_probability_score, 100)[OFFSET(95)] AS baseline_p95
  FROM
    `fraud_detection.enriched_transactions`
  WHERE
    event_timestamp BETWEEN
      TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
      AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
),

recent_stats AS (
  SELECT
    AVG(fraud_probability_score) AS recent_avg_score,
    STDDEV(fraud_probability_score) AS recent_stddev,
    APPROX_QUANTILES(fraud_probability_score, 100)[OFFSET(50)] AS recent_median,
    APPROX_QUANTILES(fraud_probability_score, 100)[OFFSET(95)] AS recent_p95,
    COUNT(*) AS sample_size
  FROM
    `fraud_detection.enriched_transactions`
  WHERE
    event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
)

SELECT
  r.recent_avg_score,
  b.baseline_avg_score,
  ABS(r.recent_avg_score - b.baseline_avg_score) / NULLIF(b.baseline_stddev, 0) AS z_score_shift,
  r.recent_p95,
  b.baseline_p95,
  r.sample_size,
  CASE
    WHEN ABS(r.recent_avg_score - b.baseline_avg_score) / NULLIF(b.baseline_stddev, 0) > 3.0
      THEN 'CRITICAL: Score distribution has shifted significantly — possible model drift'
    WHEN ABS(r.recent_avg_score - b.baseline_avg_score) / NULLIF(b.baseline_stddev, 0) > 2.0
      THEN 'WARNING: Moderate score distribution shift detected'
    ELSE 'OK: Score distribution within normal range'
  END AS drift_assessment
FROM recent_stats r, baseline_stats b;
