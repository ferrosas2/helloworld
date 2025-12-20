/* 
   Generate Training Data for LTR from The Look E-Commerce Dataset
   Source: https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce
*/

WITH Positive_Interactions AS (
  -- Users who actually bought items (Label = 1)
  SELECT 
    CAST(o.user_id AS STRING) AS query_group_id, -- Cast to string to match grouping logic later
    oi.product_id,
    p.category,
    p.retail_price,
    p.cost,
    1 AS label -- RELEVANT
  FROM `bigquery-public-data.thelook_ecommerce.orders` o
  JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi ON o.order_id = oi.order_id
  JOIN `bigquery-public-data.thelook_ecommerce.products` p ON oi.product_id = p.id
  WHERE o.status = 'Complete'
  LIMIT 5000
),

Negative_Interactions AS (
  -- Random items the user likely didn't see or didn't buy (Label = 0)
  SELECT 
    CAST(u.id AS STRING) AS query_group_id,
    p.id AS product_id,
    p.category,
    p.retail_price,
    p.cost,
    0 AS label -- IRRELEVANT
  FROM `bigquery-public-data.thelook_ecommerce.users` u
  CROSS JOIN `bigquery-public-data.thelook_ecommerce.products` p
  WHERE rand() < 0.001 -- Downsample to keep it manageable
  LIMIT 5000
)

SELECT * FROM Positive_Interactions
UNION ALL
SELECT * FROM Negative_Interactions
ORDER BY query_group_id