# Fernando Rosas — Strategic Technology Leader
### Google Cloud Professional ML Engineer · AWS Certified AI Practitioner

**GCP · AWS · GenAI · RAG · MLOps · Recommendation Systems**

15+ years designing and shipping production AI systems for regulated industries (financial services, insurance, logistics). This repository is a working portfolio — real code, real infrastructure, real business impact — targeted at **Lead AI**, **AI Director**, and senior **ML / Data Science Engineering** roles.

[![GitHub](https://img.shields.io/badge/GitHub-ferrosas2-181717?logo=github)](https://github.com/ferrosas2)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ferrosas2-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ferrosas2/)
[![Blog](https://img.shields.io/badge/Blog-Technical%20Articles-FF5722?logo=blogger)](https://rosasfernando2.blogspot.com/)
[![Portfolio](https://img.shields.io/badge/Portfolio-rosasfernando.com-4285F4)](https://rosasfernando.com)
[![GCP ML Engineer](https://img.shields.io/badge/Google_Cloud-Professional_ML_Engineer-4285F4?logo=googlecloud&logoColor=white)](https://www.credly.com/badges/cd527485-7690-4a69-934b-97dcefb3d55a/public_url)
[![AWS AI Practitioner](https://img.shields.io/badge/AWS-Certified_AI_Practitioner-FF9900?logo=amazonaws&logoColor=white)](https://www.credly.com/badges/a66988c2-eb9a-4972-80e1-33ca9c8cea70/public_url)

---

## Certifications

| Credential | Issuer | Valid | Verify |
|---|---|---|---|
| **Professional Machine Learning Engineer** | Google Cloud | Sep 2024 – Sep 2026 | [Credly](https://www.credly.com/badges/cd527485-7690-4a69-934b-97dcefb3d55a/public_url) |
| **AWS Certified AI Practitioner** | Amazon Web Services | May 2026 – May 2029 | [Credly](https://www.credly.com/badges/a66988c2-eb9a-4972-80e1-33ca9c8cea70/public_url) |

---

## At a Glance

| I build systems that... | Example project |
|---|---|
| Detect fraud with grounded, auditable LLM output | [RO-Fraud RAG (GCP, live)](AI-Systems-Architecture/RO-Fraud) |
| Stay up when models fail — circuit breakers, multi-region failover | [Resilient Financial AI Assistant](Cert-GenAI-Dev/Bonus_assignments/task_1_2) |
| Cut GenAI cost by 90%+ without losing quality | [Insurance Claims: Nova vs Claude](Cert-GenAI-Dev/Bonus_assignments/task_1_1) |
| Rank 100K+ items in under 50ms | [Two-Stage Ranking (XGBoost LTR)](recommendation_systems/two-stage-ranking) |
| Forecast at warehouse scale without leaving the DW | [BigQuery ML Forecasting Engine](AI-Systems-Architecture/forecasting-anomaly-engine) |
| Do MLOps cleanly on Vertex AI | [Vertex Classic ML Pipeline](GCP/credit_risk/vertex-classic-ml) |

---

## Core Expertise

- **Generative AI** — multi-model evaluation, prompt engineering, grounding, cost/quality tradeoffs across Gemini, Claude, Titan, Nova, Cohere
- **RAG Systems** — Vertex AI Vector Search, OpenSearch Serverless, FAISS; hybrid search, query decomposition, PII sanitization, RAGAS quality gates
- **MLOps** — end-to-end pipelines on Vertex AI and SageMaker; Cloud Run / Lambda serving; Terraform IaC; Cloud Build / CI quality gates
- **Production Resilience** — circuit breakers, graceful degradation, multi-region failover, structured output validation
- **Recommendation Systems** — two-stage ranking (retrieval + re-rank), Learning-to-Rank with XGBoost, deep-learning two-tower retrieval

## Technical Skills

| Domain | Technologies |
|---|---|
| **GCP** | Vertex AI (Training, Experiments, Vizier HPT, Vector Search), BigQuery ML, Cloud Run, Cloud Build, Cloud Storage, Firestore, Cloud Workflows, Cloud Monitoring |
| **AWS** | Bedrock, SageMaker, Lambda, OpenSearch Serverless, S3, DynamoDB, API Gateway, Step Functions, CloudWatch, Route 53, AppConfig |
| **Foundation Models** | Gemini 2.5 Flash / 1.5 Pro, Claude 3 Sonnet / Instant, Amazon Nova Micro, Amazon Titan (LLM + Embeddings), Cohere |
| **ML & Data** | XGBoost, Learning-to-Rank, Two-Tower models, ARIMA_PLUS (BQML), Transformers, Vector Search, Time Series |
| **Engineering** | Python 3.10+, FastAPI, LangChain, Pydantic v2, Terraform, Docker, pytest, RAGAS |

---

## Featured Projects

### GCP · Production RAG & MLOps

#### 1. AI Fraud & Risk Analysis Engine — `Live on GCP`
**[`AI-Systems-Architecture/RO-Fraud`](AI-Systems-Architecture/RO-Fraud)**

Production-grade RAG for fraud detection at a Tier-1 financial services client. Deployed end-to-end on Cloud Run with Vertex AI Vector Search and Gemini 2.5 Flash. Includes RAGAS quality gates, PII sanitization, Terraform IaC, and Cloud Build CI/CD.

| Metric | Result |
|---|---|
| Fraud identification precision | **60% → 81%** (+35% relative) |
| Adjuster review time per claim | **1.5h → 45min** (-50%) |
| RAG faithfulness gate | **≥ 0.80** (RAGAS, CI-enforced) |

**Stack:** Vertex AI Vector Search · Gemini 2.5 Flash · LangChain · FastAPI · Cloud Run · Terraform · RAGAS

---

#### 2. Logistics & Fintech RAG Platform — `Prod-Ready`
**[`AI-Systems-Architecture/RAG`](AI-Systems-Architecture/RAG)**

Conversational analytics over unstructured freight and financial documents (Bills of Lading, Freight Claims). Emphasises deterministic output for financial compliance — zero-temperature generation, strict Pydantic schemas, explicit anti-hallucination guardrails.

**Stack:** Vertex AI (Gemini 1.5 Pro · text-embedding-gecko) · LangChain · FAISS · FastAPI · Docker

---

#### 3. Forecasting & Anomaly Engine — `Prod-Ready`
**[`AI-Systems-Architecture/forecasting-anomaly-engine`](AI-Systems-Architecture/forecasting-anomaly-engine)**

Time-series forecasting and anomaly detection directly in the data warehouse using BigQuery ML. Zero-infrastructure MLOps: `ARIMA_PLUS` for forecasting, `ML.DETECT_ANOMALIES` for outliers, `ML.EXPLAIN_FORECAST` for XAI.

| Metric | Result |
|---|---|
| Inventory accuracy | **+15%** |
| Annual holding cost reduction | **~$2M** |

**Stack:** BigQuery ML · ARIMA_PLUS · Python BQ Client · Cloud Scheduler

---

#### 4. Vertex Classic ML Pipeline — `Verified`
**[`GCP/credit_risk/vertex-classic-ml`](GCP/credit_risk/vertex-classic-ml)**

Reference MLOps pattern for classic ML on Vertex AI: clean separation of orchestration (`submit_jobs.py`) and execution (`train.py`), Bayesian HPT via Vertex Vizier, full experiment lineage in Vertex AI Experiments, artefacts persisted to GCS. Verified end-to-end (June 2026).

**Stack:** Vertex AI Training · Vertex AI Experiments · Vertex Vizier · `cloudml-hypertune` · GCS

---

### AWS · GenAI on Bedrock

#### 5. Resilient Financial Services AI Assistant — `Prod-Ready`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_2`](Cert-GenAI-Dev/Bonus_assignments/task_1_2)**

Enterprise AI assistant for regulated industries. Dynamic model routing, circuit breakers, and graceful degradation deliver 100% availability under upstream failures. Multi-region failover under one minute.

| Metric | Result |
|---|---|
| Latency reduction | **66%** |
| Success rate | **100%** |
| Multi-region failover | **< 1 min** |

**Stack:** Bedrock · Lambda · API Gateway · Step Functions · AppConfig · Route 53

---

#### 6. Enterprise RAG with OpenSearch — `Prod-Ready`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_4`](Cert-GenAI-Dev/Bonus_assignments/task_1_4)**

Production RAG on Bedrock Knowledge Bases and OpenSearch Serverless. 50 subreddits indexed with 1536-dim Titan embeddings, semantic search, A/B testing, and automated cleanup.

| Metric | Result |
|---|---|
| Query latency | **< 500 ms** |
| Corpora indexed | **50 subreddits** |
| Vector dimensions | **1536** (Titan) |

**Stack:** OpenSearch Serverless · Bedrock Knowledge Base · Titan Embeddings · Claude 3 · DynamoDB

---

#### 7. Insurance Claims GenAI — `POC`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_1`](Cert-GenAI-Dev/Bonus_assignments/task_1_1)**

Automated extraction and summarisation of unstructured insurance claims. A rigorous model comparison drove the switch from Claude to Amazon Nova Micro.

| Metric | Result |
|---|---|
| Cost reduction | **95%** |
| Throughput improvement | **5.3×** |
| Avg latency | **2.8 s** |

**Stack:** Nova Micro · Claude 3 · S3 · Faker

---

#### 8. Advanced RAG for Historical Records — `Research`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_5`](Cert-GenAI-Dev/Bonus_assignments/task_1_5)**

Multilingual RAG over Los Altos de Jalisco parish records (Spanish / Latin / Nahuatl). Hybrid keyword + semantic search, query decomposition, reranking, and entity extraction.

| Metric | Result |
|---|---|
| Search strategy | **Hybrid (keyword + semantic)** |
| Languages | **ES / LA / Nahuatl** |
| Monthly infra savings | **$1,485** |

**Stack:** Cohere Embeddings · Query Decomposition · Reranking · Entity Extraction

---

#### 9. Customer Feedback Analysis — `POC`
**[`Cert-GenAI-Dev/Bonus_assignments/task_1_3`](Cert-GenAI-Dev/Bonus_assignments/task_1_3)**

Sentiment and topic extraction pipeline over customer feedback, with a model-selection strategy comparing foundation models by cost and quality.

---

### Recommendation Systems

#### 10. Two-Stage Ranking (LTR + XGBoost) — `Prod-Ready`
**[`recommendation_systems/two-stage-ranking`](recommendation_systems/two-stage-ranking)**

E-commerce recommender combining fast candidate retrieval with a Learning-to-Rank re-ranker. Optimises business metrics directly (margin, conversion), not just relevance.

| Metric | Result |
|---|---|
| Inference latency | **< 50 ms** |
| Catalogue size | **100K+ items** |
| Ranking quality | **NDCG@10** |

**Stack:** XGBoost · BigQuery · SageMaker · Docker · Lambda

#### 11. Two-Tower Retrieval Model
**[`recommendation_systems/two-tower`](recommendation_systems/two-tower)**

Deep-learning two-tower retrieval with ETL, training, and inference modules — the retrieval side of a modern recommender stack.

---

### Analytics Notebook

**[`regrssion_metrics.ipynb`](regrssion_metrics.ipynb) · [`synthetic_transactions.csv`](synthetic_transactions.csv)** — regression evaluation walkthrough over synthetic transaction data.

---

## Repository Structure

```
helloworld/
├── AI-Systems-Architecture/
│   ├── RO-Fraud/                      # Live GCP RAG for fraud detection
│   ├── RAG/                           # Logistics & fintech RAG platform
│   └── forecasting-anomaly-engine/    # BigQuery ML forecasting + anomalies
├── GCP/
│   └── credit_risk/vertex-classic-ml/ # Vertex AI classic-ML MLOps pattern
├── Cert-GenAI-Dev/
│   └── Bonus_assignments/
│       ├── task_1_1/                  # Insurance Claims GenAI POC
│       ├── task_1_2/                  # Resilient Financial AI Assistant
│       ├── task_1_3/                  # Customer Feedback Analysis
│       ├── task_1_4/                  # Enterprise RAG with OpenSearch
│       └── task_1_5/                  # Multilingual RAG for Historical Records
├── recommendation_systems/
│   ├── two-stage-ranking/             # Learning-to-Rank with XGBoost
│   └── two-tower/                     # Deep-learning retrieval
├── index.html · styles.css · script.js  # Portfolio site
├── regrssion_metrics.ipynb              # Regression evaluation notebook
└── README.md
```

---

## Engineering Principles

Themes you will see repeated across these projects:

- **Grounding over cleverness.** LLM outputs are constrained by Pydantic schemas and cited context. If the model can't ground its answer, the request errors — it doesn't hallucinate.
- **Failure modes are designed, not discovered.** Circuit breakers, graceful degradation, and multi-region failover are first-class citizens, not afterthoughts.
- **Quality is measured, not asserted.** RAGAS gates in CI, NDCG@10 for rankers, cost-per-inference tracked alongside accuracy.
- **Infrastructure is code.** Terraform for GCP resources, Cloud Build / CI/CD pipelines, reproducible container builds.
- **Cost is a design constraint.** Choosing Gemini 2.5 Flash over 1.5 Pro, Nova Micro over Claude — with the numbers to justify each call.

---

## Contact

| Channel | Link |
|---|---|
| GitHub | [@ferrosas2](https://github.com/ferrosas2) |
| LinkedIn | [ferrosas2](https://www.linkedin.com/in/ferrosas2/) |
| Blog | [rosasfernando2.blogspot.com](https://rosasfernando2.blogspot.com/) |
| Portfolio | [rosasfernando.com](https://rosasfernando.com) |

---

*© 2026 Fernando Rosas · Built with a bias for shipping.*
