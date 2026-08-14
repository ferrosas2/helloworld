-- =============================================================================
-- BigQuery ML: Model Evaluation & Performance Monitoring
-- =============================================================================
-- Evaluates the fraud classifier's performance and stores metrics
-- for historical tracking. Run weekly or after each retraining cycle.
--
-- Tracks: Precision, Recall, F1-Score, AUC-ROC, AUC-PR, Confusion Matrix
-- Writes results to fraud_detection.model_metrics for dashboarding.
-- =============================================================================


-- =============================================================================
-- Step 1: Overall model evaluation (latest model version)
-- =============================================================================

SELECT
  *
FROM
  ML.EVALUATE(MODEL `fraud_detection.fraud_classifier_v1`);


-- =============================================================================
-- Step 2: Evaluation at specific decision thresholds
-- =============================================================================

-- Evaluate at the production threshold (0.7 for FLAG, 0.3 for REVIEW)
SELECT
  *
FROM
  ML.EVALUATE(
    MODEL `fraud_detection.fraud_classifier_v1`,
    (
      SELECT
        amount,
        f_amount_log,
        f_is_round_amount,
        f_merchant_risk_score,
        f_terminal_type_risk,
        f_hour_of_day,
        f_is_night_transaction,
        is_international,
        pin_entered,
        recurring,
        f_velocity_count,
        f_velocity_total_amount,
        f_velocity_distinct_merchants,
        CASE WHEN merchant_category IN ('jewelry', 'electronics', 'online_retail') THEN 1 ELSE 0 END AS is_high_risk_category,
        CASE WHEN terminal_type = 'online' AND NOT pin_entered THEN 1 ELSE 0 END AS is_card_not_present,
        CASE WHEN f_velocity_count > 5 AND f_velocity_total_amount > 2000 THEN 1 ELSE 0 END AS velocity_alert_flag,
        label_confirmed_fraud
      FROM `fraud_detection.enriched_transactions`
      WHERE
        label_confirmed_fraud IS NOT NULL
        AND event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    ),
    STRUCT(0.7 AS threshold)
  );


-- =============================================================================
-- Step 3: Confusion matrix for understanding error distribution
-- =============================================================================

SELECT
  *
FROM
  ML.CONFUSION_MATRIX(MODEL `fraud_detection.fraud_classifier_v1`);


-- =============================================================================
-- Step 4: Feature importance (Global Explainability)
-- =============================================================================

-- Which features drive the model's decisions most?
SELECT
  *
FROM
  ML.GLOBAL_EXPLAIN(MODEL `fraud_detection.fraud_classifier_v1`)
ORDER BY
  attribution DESC;


-- =============================================================================
-- Step 5: ROC Curve data points for visualization
-- =============================================================================

SELECT
  *
FROM
  ML.ROC_CURVE(MODEL `fraud_detection.fraud_classifier_v1`);


-- =============================================================================
-- Step 6: Store evaluation results for historical tracking
-- =============================================================================

-- Insert latest evaluation metrics into the tracking table
INSERT INTO `fraud_detection.model_metrics`
  (evaluation_timestamp, model_version, metric_name, metric_value, threshold, dataset_size, fraud_rate, notes)

WITH eval_results AS (
  SELECT * FROM ML.EVALUATE(MODEL `fraud_detection.fraud_classifier_v1`)
),
dataset_stats AS (
  SELECT
    COUNT(*) AS total_rows,
    COUNTIF(label_confirmed_fraud = TRUE) / COUNT(*) AS fraud_rate
  FROM `fraud_detection.enriched_transactions`
  WHERE label_confirmed_fraud IS NOT NULL
    AND event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
)

SELECT
  CURRENT_TIMESTAMP() AS evaluation_timestamp,
  'fraud_classifier_v1' AS model_version,
  metric_name,
  metric_value,
  0.7 AS threshold,
  ds.total_rows AS dataset_size,
  ds.fraud_rate,
  'Scheduled weekly evaluation' AS notes
FROM (
  SELECT 'precision' AS metric_name, precision AS metric_value FROM eval_results
  UNION ALL
  SELECT 'recall', recall FROM eval_results
  UNION ALL
  SELECT 'f1_score', f1_score FROM eval_results
  UNION ALL
  SELECT 'log_loss', log_loss FROM eval_results
  UNION ALL
  SELECT 'roc_auc', roc_auc FROM eval_results
), dataset_stats ds;


-- =============================================================================
-- Step 7: Weekly performance comparison (detect degradation)
-- =============================================================================

-- Compare this week's metrics to last week to flag model degradation
WITH current_week AS (
  SELECT metric_name, metric_value
  FROM `fraud_detection.model_metrics`
  WHERE evaluation_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    AND model_version = 'fraud_classifier_v1'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY metric_name ORDER BY evaluation_timestamp DESC) = 1
),

previous_week AS (
  SELECT metric_name, metric_value
  FROM `fraud_detection.model_metrics`
  WHERE evaluation_timestamp BETWEEN
    TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
    AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    AND model_version = 'fraud_classifier_v1'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY metric_name ORDER BY evaluation_timestamp DESC) = 1
)

SELECT
  c.metric_name,
  c.metric_value AS current_value,
  p.metric_value AS previous_value,
  ROUND((c.metric_value - p.metric_value) / NULLIF(p.metric_value, 0) * 100, 2) AS change_pct,
  CASE
    WHEN c.metric_name IN ('precision', 'recall', 'f1_score', 'roc_auc')
      AND c.metric_value < p.metric_value * 0.95
      THEN 'DEGRADATION ALERT'
    WHEN c.metric_name = 'log_loss'
      AND c.metric_value > p.metric_value * 1.1
      THEN 'DEGRADATION ALERT'
    ELSE 'OK'
  END AS status
FROM current_week c
LEFT JOIN previous_week p ON c.metric_name = p.metric_name
ORDER BY c.metric_name;
