-- Explainablity & XAI
-- Breaking down the forecast into seasonal, trend, and holiday effects
-- This breakdown is crucial for building business stakeholder trust and understanding the driving factors of demand.
SELECT
  *
FROM
  ML.EXPLAIN_FORECAST(
    MODEL `my_dataset.demand_forecast_model`,
    STRUCT(30 AS horizon, 0.95 AS confidence_level)
  );
