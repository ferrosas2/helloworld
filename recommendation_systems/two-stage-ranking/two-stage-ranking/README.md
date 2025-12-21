# Two-Stage Ranking System on AWS

**Production-grade XGBoost Learning-to-Rank (LambdaMART) implementation for high-scale auctions and e-commerce recommendations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-yellow.svg)](https://aws.amazon.com/sagemaker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Problem Statement

In high-scale auction platforms and e-commerce marketplaces, **ranking unique items by relevance** is fundamentally different from traditional recommendation systems:

- **No Click History**: Unique auction items lack historical click data
- **Sparse Interactions**: Items may only appear once
- **Multi-Objective Optimization**: Balance user preferences, business margins, and inventory velocity
- **Real-Time Constraints**: Rankings must be computed in < 100ms for live auctions

**Traditional CTR models fail** because they rely on historical engagement data that doesn't exist for unique items.

---

## 💡 Solution: Two-Stage Ranking Architecture

This system implements a **retrieval + re-ranking** pipeline optimized for AWS:

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│                 │       │                  │       │                 │
│  OpenSearch/    │  -->  │  XGBoost Ranker  │  -->  │  Ranked Results │
│  Elasticsearch  │       │  (LambdaMART)    │       │  (Top-K Items)  │
│                 │       │                  │       │                 │
│ Stage 1:        │       │ Stage 2:         │       │                 │
│ Fast Retrieval  │       │ Precision        │       │ Personalized    │
│ (100K -> 100)   │       │ Re-Ranking       │       │ for User        │
└─────────────────┘       └──────────────────┘       └─────────────────┘
```

### Stage 1: Retrieval (OpenSearch)
- **Vector Similarity Search** on item embeddings
- Filters by category, price range, location
- Returns **top 100-500 candidates** in ~20ms

### Stage 2: Re-Ranking (XGBoost LambdaMART)
- **Learning-to-Rank model** optimizes NDCG@10
- Features: `retail_price`, `cost`, `margin`, `category`, `freshness`
- **Pairwise ranking objective** learns relative item ordering
- Returns **top 10-20 items** in ~50ms

---

## 🏗️ Project Structure

```
two-stage-ranking/
├── src/
│   ├── train.py           # Modular training pipeline
│   └── inference.py       # SageMaker-compatible inference handler
├── infrastructure/
│   └── sagemaker.yaml     # CloudFormation/Terraform templates (future)
├── notebooks/
│   ├── ltr.ipynb          # Original exploratory analysis
│   └── ltr-training-data.sql  # BigQuery data extraction script
├── tests/
│   └── test_inference.py  # Unit tests (future)
├── Dockerfile             # Production container
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **AWS Account** with S3 access
- **Docker** (optional, for containerization)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"
# Enter your AWS Access Key ID, Secret Access Key
```

### 3. Train the Model Locally

```bash
python src/train.py --bucket ltr-models-frp --key data/ltr_training_data.csv --output-dir ./models --n-estimators 100 --learning-rate 0.1
```

**Expected Output:**
```
[INFO] Loading data from s3://ltr-models-frp/data/ltr_training_data.csv
[INFO] Successfully loaded 10000 rows
[INFO] Preprocessing data for ranker...
[INFO] Number of query groups: 3168
[INFO] Training model...
[INFO] Training completed successfully!
[INFO] Model saved at: ./models/model.json
```

### 4. Test Inference Locally

```bash
python src/inference.py --model-path ./models/model.json
```

**Sample Output:**
```
Rank   Item Name                      Category        Price      LTR Score    Retrieval Score
------------------------------------------------------------------------------------------
1      Luxury Briefcase               Bags            $99.99     0.0867       0.83
2      Premium Sunglasses             Accessories     $49.99     -0.0414      0.87
3      Vintage Rolex Watch            Watches         $89.99     -0.0662      0.92
4      Silk Tie Set                   Accessories     $79.99     -0.0662      0.85
5      Designer Leather Wallet        Accessories     $29.99     -0.1316      0.89
```
### 4. Run Full Demo (3 minutes)
```bash
python examples/demo_pipeline.py
```
---

## 🐳 Docker Deployment

### Build Container

```bash
docker build -t two-stage-ranking:latest .
```

### Run Training in Container

```bash
docker run \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -v $(pwd)/models:/opt/ml/model \
  two-stage-ranking:latest \
  --bucket ltr-models-frp \
  --key data/ltr_training_data.csv
```

### Deploy to Amazon ECR

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag two-stage-ranking:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/two-stage-ranking:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/two-stage-ranking:latest
```

---

## 📊 Data Source

### BigQuery Public Dataset: `thelook_ecommerce`

This project uses Google BigQuery's [thelook_ecommerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) public dataset, which simulates a **clothing retailer** with realistic e-commerce data. The dataset contains:

- **User purchase history**: "User bought Item X" interactions
- **Product catalog**: Clothing items with categories, prices, and costs
- **Order transactions**: Real-world purchase patterns and user behavior

**Why thelook_ecommerce?**
- Realistic multi-category product catalog (Intimates, Socks, Pants, etc.)
- Actual purchase signals for relevance labels
- Rich product metadata (retail price, cost, brand, department)
- Suitable for training learning-to-rank models

### Data Extraction

The training data is generated using a SQL query stored in:
```
notebooks/ltr-training-data.sql
```

This query:
1. Joins user purchase events with product details
2. Creates query groups (e.g., users searching for similar items)
3. Generates relevance labels based on purchase behavior
4. Extracts features: `retail_price`, `cost`, `category`, etc.

**To regenerate the dataset:**
```bash
# Run in BigQuery Console or using bq CLI
bq query --use_legacy_sql=false < notebooks/ltr-training-data.sql > data/ltr_training_data.csv
```

---

## 📊 Training Data Format

The model expects CSV data with the following schema:

| Column            | Type    | Description                                    |
|-------------------|---------|------------------------------------------------|
| `query_group_id`  | int     | Groups items belonging to the same query       |
| `label`           | int     | Relevance label (0=not purchased, 1=purchased) |
| `retail_price`    | float   | Item price shown to customer                   |
| `cost`            | float   | Internal cost (for margin calculation)         |
| `category`        | string  | Item category (Watches, Bags, etc.)            |

**Example (from thelook_ecommerce):**
```csv
query_group_id,label,retail_price,cost,category
1,1,89.99,45.0,Intimates
1,0,29.99,15.0,Socks
1,0,79.99,40.0,Pants
2,1,149.99,75.0,Outerwear & Coats
2,0,99.99,50.0,Sweaters
```

**Data Source:** Generated from `notebooks/ltr-training-data.sql` using BigQuery's `thelook_ecommerce` public dataset.

---

## 🧠 Model Architecture

### XGBoost LambdaMART

**Why LambdaMART?**
- **Pairwise ranking** learns relative ordering within groups
- **Gradient boosting** handles non-linear feature interactions
- **NDCG optimization** directly targets ranking quality
- **Production-ready** with < 50ms inference latency

**Hyperparameters:**
```python
{
    "objective": "rank:pairwise",    # LambdaMART pairwise loss
    "eval_metric": "ndcg",           # Optimize NDCG@K
    "learning_rate": 0.1,            # Conservative for generalization
    "n_estimators": 100,             # 100 boosting rounds
    "gamma": 1.0,                    # Regularization
    "min_child_weight": 0.1          # Prevent overfitting
}
```

---

## 🔬 Feature Engineering (Future Enhancements)

Current baseline uses **numeric features**:
- `retail_price`
- `cost`

**Production-ready features to add:**
- **One-Hot Encoding**: `category` → 50+ binary features
- **Profit Margin**: `(retail_price - cost) / retail_price`
- **Price Competitiveness**: Z-score within category
- **Item Freshness**: Days since listing
- **Seller Reputation**: Historical rating/reviews
- **User-Item Interactions**: Click-through rate, dwell time
- **Contextual Features**: Time of day, device type

**Additional evaluation metrics to implement:**
- **MAP** (Mean Average Precision): Precision at various recall levels
- **MRR** (Mean Reciprocal Rank): Position of first relevant item

---

## 📈 Evaluation Metrics

The model is currently evaluated using:

- **NDCG@10** (Normalized Discounted Cumulative Gain): Measures ranking quality with position discount

**Current Performance Target:**
- NDCG@10 > 0.75
- Inference latency < 50ms

---

## 🚀 Deployment Options

### Option 1: SageMaker Real-Time Endpoint

```python
import boto3

sagemaker = boto3.client('sagemaker')

# Create endpoint
sagemaker.create_endpoint(
    EndpointName='two-stage-ranking-prod',
    EndpointConfigName='two-stage-ranking-config'
)
```

### Option 2: Lambda Function (Batch Processing)

```python
import json
import boto3
from src.inference import RankingInferenceHandler

def lambda_handler(event, context):
    handler = RankingInferenceHandler()
    results = handler.handle_request(event['items'])
    return {'statusCode': 200, 'body': json.dumps(results)}
```

### Option 3: ECS/Fargate Service

Deploy as a REST API using FastAPI:

```python
from fastapi import FastAPI
from src.inference import RankingInferenceHandler

app = FastAPI()
handler = RankingInferenceHandler()

@app.post("/rank")
def rank_items(items: list):
    return handler.handle_request(items)
```

---

## 🛠️ Development Workflow

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Type Checking

```bash
mypy src/
```

### Code Formatting

```bash
black src/
isort src/
```

### AWS Documentation
- [SageMaker XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [SageMaker Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [OpenSearch Vector Search](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html)

### XGBoost Resources
- [XGBoost Ranking Tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/index.html)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Fernando Rosas**  


- GitHub: [@ferrosas2](https://github.com/ferrosas2)
- LinkedIn: [Connect](https://www.linkedin.com/in/ferrosas2/)


## 📌 Roadmap

- [x] Core training pipeline
- [x] SageMaker-compatible inference handler
- [x] Docker containerization
- [ ] Unit tests with pytest
- [ ] Integration with OpenSearch (Stage 1)
- [ ] CloudFormation/Terraform templates
- [ ] A/B testing framework
- [ ] Model monitoring dashboard
- [ ] Feature store integration

---

