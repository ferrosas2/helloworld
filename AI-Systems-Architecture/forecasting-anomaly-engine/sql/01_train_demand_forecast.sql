-- Train Demand Forecast Model
CREATE OR REPLACE MODEL `my_dataset.demand_forecast_model`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='timestamp',
  time_series_data_col='demand_volume',
  time_series_id_col='item_id',
  data_frequency='AUTO',
  decompose_time_series=TRUE
) AS
SELECT 
  timestamp,
  item_id,
  demand_volume
FROM
  `my_project.my_dataset.historical_demand`;