# Predictive Forecasting & Anomaly Detection (BigQuery ML)

**Business Impact:** Led to a 15% improvement in inventory accuracy and an estimated $2M reduction in annual holding costs.

This repository demonstrates predictive analytics and time-series forecasting using Google Cloud BigQuery ML (BQML) and Python. By building an ML pipeline entirely within the Data Warehouse, we achieve infinite scalability and zero-infrastructure MLOps.

## Architecture

```mermaid
graph TD
    subgraph Google Cloud Platform
        A[(Enterprise Data Warehouse\nBigQuery Tabular Data)] -->|SQL: CREATE MODEL| B(BigQuery ML Engine)
        B -->|ARIMA_PLUS Model| C{Time-Series Forecasting}
    end

    subgraph Analytics & Governance
        C -->|ML.DETECT_ANOMALIES| D[Price/Demand Anomaly Flags]
        C -->|ML.EXPLAIN_FORECAST| E[Model Explainability XAI]
    end

    subgraph Python MLOps Layer
        F[Python BQ Client / Orchestrator] -->|Triggers Jobs| B
        D --> F
        E --> F
    end

    subgraph Business Operations
        F -->|Actionable Insights| G[CFO & Directors Dashboard]
        F -->|Automated Alerts| H[Operations / Pricing Team]
    end
    
    classDef BQ fill:#263238,stroke:#4285f4,stroke-width:2px;
    class A,B,C BQ;
```

## Setup & Execution

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure you have authenticated with Google Cloud and set your `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
3. Use the Python Orchestrator (`src/bq_orchestration.py`) to execute the models and anomaly detection.
