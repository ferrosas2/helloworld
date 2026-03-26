# Fernando Rosas — Strategic Technology Leader | Google Certified ML Engineer

> **15+ years** building production-ready AI systems on GCP and AWS • GenAI • RAG • MLOps • Recommendation Systems

[![GitHub](https://img.shields.io/badge/GitHub-ferrosas2-181717?logo=github)](https://github.com/ferrosas2)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ferrosas2-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ferrosas2/)
[![Blog](https://img.shields.io/badge/Blog-Technical%20Articles-FF5722?logo=blogger)](https://rosasfernando2.blogspot.com/)
[![Portfolio](https://img.shields.io/badge/Portfolio-rosasfernando.com-4285F4)](https://rosasfernando.com)

---

## About

Strategic Technology Leader with over 15 years of experience designing and shipping production-ready AI systems across GCP and AWS. This repository is a portfolio of working code, notebooks, and reference implementations demonstrating end-to-end capabilities relevant to **Lead AI**, **AI Director**, and senior **Data Science / ML Engineering** roles.

**Core expertise:**

- **Generative AI** — Multi-model evaluation, prompt engineering, Bedrock integration (Claude, Titan, Nova)
- **RAG Systems** — Advanced retrieval-augmented generation with OpenSearch, vector embeddings, hybrid search
- **MLOps** — End-to-end ML pipelines with SageMaker & Vertex AI, Lambda orchestration, automated deployment
- **Production Systems** — Resilient architectures with circuit breakers, graceful degradation, multi-region failover
- **Recommendation Systems** — Two-stage ranking (retrieval + re-ranking), Learning-to-Rank with XGBoost

---

## Technical Skills

| Domain | Technologies |
|---|---|
| **AWS Services** | Bedrock, SageMaker, Lambda, OpenSearch, S3, DynamoDB, API Gateway, Step Functions, CloudWatch, Route 53 |
| **GCP & BigQuery** | Vertex AI, BigQuery, Cloud Functions, Vertex AI Search, Cloud Storage, Firestore, Cloud Endpoints, Cloud Workflows, Cloud Monitoring |
| **Foundation Models** | Claude 3 Sonnet, Claude Instant, Amazon Titan, Amazon Nova, Titan Embeddings, Cohere |
| **ML & Data Science** | XGBoost, Learning-to-Rank, Transformers, Vector Search, NLP, Time Series |
| **Python & Tools** | boto3, pandas, scikit-learn, HuggingFace, Docker, Git |

---

## Featured Projects

### 1. Resilient Financial Services AI Assistant `Production-Ready`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_2`](Cert-GenAI-Dev/Bonus_assignments/task_1_2)**

Enterprise-grade AI assistant with dynamic model routing, circuit breakers, and graceful degradation for regulated industries. Achieved 100% reliability with a multi-layer fallback strategy.

| Metric | Result |
|---|---|
| Latency Reduction | **66%** |
| Success Rate | **100%** |
| Multi-Region Failover | **< 1 min** |

**Stack:** Bedrock · Lambda · API Gateway · Step Functions · AppConfig · Route 53

---

### 2. Enterprise RAG System with OpenSearch `Production-Ready`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_4`](Cert-GenAI-Dev/Bonus_assignments/task_1_4)**

Production-grade Retrieval-Augmented Generation system using Amazon Bedrock Knowledge Bases and OpenSearch Serverless. Features semantic search, A/B testing framework, and automated cleanup.

| Metric | Result |
|---|---|
| Query Latency | **< 500 ms** |
| Subreddits Indexed | **50** |
| Vector Embeddings | **1536-dim** |

**Stack:** OpenSearch · Bedrock Knowledge Base · Titan Embeddings · Claude 3 · DynamoDB

---

### 3. Two-Stage Ranking System `Production-Ready`
**[`recommendation_systems/two-stage-ranking`](recommendation_systems/two-stage-ranking)**

Learning-to-Rank system with XGBoost for e-commerce recommendations. Combines fast candidate retrieval with precise re-ranking to optimize business metrics (margins, conversion rates).

| Metric | Result |
|---|---|
| Inference Time | **< 50 ms** |
| Items Handled | **100K+** |
| Ranking Quality | **NDCG@10** |

**Stack:** XGBoost · BigQuery · SageMaker · Docker · Lambda

Also includes a **Two-Tower model** ([`recommendation_systems/two-tower`](recommendation_systems/two-tower)) for deep-learning-based retrieval with an ETL pipeline, model training, and inference modules.

---

### 4. Insurance Claims Processing with GenAI `POC`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_1`](Cert-GenAI-Dev/Bonus_assignments/task_1_1)**

Automated extraction and summarization of unstructured insurance claims. Achieved a 95% cost reduction and 5.3× performance improvement by switching from Claude to Amazon Nova Micro.

| Metric | Result |
|---|---|
| Cost Reduction | **95%** |
| Speed Improvement | **5.3×** |
| Avg Latency | **2.8 s** |

**Stack:** Nova Micro · Claude 3 · S3 · Faker

---

### 5. Advanced RAG System for Historical Records `Research`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_5`](Cert-GenAI-Dev/Bonus_assignments/task_1_5)**

Multilingual RAG system for Los Altos de Jalisco parish records (Spanish / Latin / Nahuatl). Features hybrid search, query decomposition, and multi-step workflow orchestration.

| Metric | Result |
|---|---|
| Search Strategy | **Hybrid (keyword + semantic)** |
| Languages | **ES / LA / Nahuatl** |
| Monthly Savings | **$1,485** |

**Stack:** Cohere Embeddings · Query Decomposition · Reranking · Entity Extraction

---

### 6. Customer Feedback Analysis `[task_1_3]`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_3`](Cert-GenAI-Dev/Bonus_assignments/task_1_3)**

Sentiment analysis and topic extraction pipeline over customer feedback data using foundation models and model selection strategies.

---

### 7. Regression Metrics & Financial Analysis
**[`regrssion_metrics.ipynb`](regrssion_metrics.ipynb) · [`synthetic_transactions.csv`](synthetic_transactions.csv)**

Regression evaluation notebook with synthetic transaction data, demonstrating end-to-end modeling and evaluation workflows.

---

## Repository Structure

```
helloworld/
├── index.html                        # Portfolio website
├── Cert-GenAI-Dev/
│   └── Bonus_assignments/
│       ├── task_1_1/                 # Insurance Claims GenAI POC
│       ├── task_1_2/                 # Resilient Financial AI Assistant
│       ├── task_1_3/                 # Customer Feedback Analysis
│       ├── task_1_4/                 # Enterprise RAG with OpenSearch
│       └── task_1_5/                 # Advanced RAG for Historical Records
├── recommendation_systems/
│   ├── two-stage-ranking/            # LTR with XGBoost (retrieval + re-rank)
│   └── two-tower/                    # Deep learning retrieval model
└── GCP/                              # GCP reference implementations
```

---

## Contact

| Channel | Link |
|---|---|
| GitHub | [@ferrosas2](https://github.com/ferrosas2) |
| LinkedIn | [ferrosas2](https://www.linkedin.com/in/ferrosas2/) |
| Blog | [rosasfernando2.blogspot.com](https://rosasfernando2.blogspot.com/) |
| Portfolio | [rosasfernando.com](https://rosasfernando.com) |

---

*© 2026 Fernando Rosas. Built with passion for Data Science & ML Engineering.*
