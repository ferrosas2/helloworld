-- =============================================================================
-- BigQuery ML: Train Fraud Classification Model (BOOSTED_TREE_CLASSIFIER)
-- =============================================================================
-- Trains a gradient-boosted tree classifier directly on the enriched
-- transactions table — zero data movement, zero external infrastructure.
--
-- The model learns from:
--   - Per-transaction features (amount, terminal, time-of-day)
--   - Velocity features (5-min card activity count)
--   - Merchant risk context (category historical fraud rate)
--   - Behavioral signals (PIN usage, international, recurring)
--
-- Scheduled to retrain nightly via Cloud Scheduler / Cloud Composer.
-- =============================================================================

CREATE OR REPLACE MODEL `fraud_detection.fraud_classifier_v1`
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['label_confirmed_fraud'],
  
  -- Hyperparameters (tuned via BQML HP tuning)
  num_parallel_tree = 5,
  max_iterations = 100,
  learn_rate = 0.1,
  max_tree_depth = 8,
  subsample = 0.8,
  min_split_loss = 0.01,
  l1_reg = 0.1,
  l2_reg = 1.0,
  
  -- Data split strategy
  data_split_method = 'AUTO_SPLIT',
  
  -- Enable feature importance for explainability
  enable_global_explain = TRUE,
  
  -- Cost-sensitive: fraud (positive class) is rare but costly
  -- Adjust class weights to penalize missed fraud more heavily
  auto_class_weights = TRUE
) AS

SELECT
  -- Features used for prediction
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
  
  -- Derived features (computed in-query)
  CASE
    WHEN merchant_category IN ('jewelry', 'electronics', 'online_retail') THEN 1
    ELSE 0
  END AS is_high_risk_category,
  
  CASE
    WHEN terminal_type = 'online' AND NOT pin_entered THEN 1
    ELSE 0
  END AS is_card_not_present,
  
  CASE
    WHEN f_velocity_count > 5 AND f_velocity_total_amount > 2000 THEN 1
    ELSE 0
  END AS velocity_alert_flag,
  
  -- Label (ground truth from fraud investigation team)
  label_confirmed_fraud

FROM
  `fraud_detection.enriched_transactions`
WHERE
  -- Only train on transactions with confirmed labels
  label_confirmed_fraud IS NOT NULL
  -- Use last 90 days of labeled data for training
  AND event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  -- Exclude dead/test transactions
  AND transaction_id IS NOT NULL
  AND amount > 0;


-- =============================================================================
-- After training, inspect model evaluation metrics
-- =============================================================================

-- View overall model evaluation
-- SELECT * FROM ML.EVALUATE(MODEL `fraud_detection.fraud_classifier_v1`);

-- View feature importance (global explainability)
-- SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `fraud_detection.fraud_classifier_v1`);

-- View confusion matrix
-- SELECT * FROM ML.CONFUSION_MATRIX(MODEL `fraud_detection.fraud_classifier_v1`);
