-- Detect anomalies using the trained ARIMA_PLUS model
SELECT
  *
FROM
  ML.DETECT_ANOMALIES(
    MODEL `my_dataset.demand_forecast_model`,
    STRUCT(0.95 AS anomaly_prob_threshold)
  )
WHERE
  is_anomaly = TRUE;
