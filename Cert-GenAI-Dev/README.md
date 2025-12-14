# AWS Generative AI Projects Portfolio

[![AWS](https://img.shields.io/badge/AWS-Certified-orange?logo=amazon-aws)](https://aws.amazon.com/)
[![Bedrock](https://img.shields.io/badge/Amazon-Bedrock-blue)](https://aws.amazon.com/bedrock/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive collection of production-ready Generative AI solutions built using AWS services, demonstrating enterprise-grade implementation of Large Language Models (LLMs), RAG architectures, and MLOps best practices.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [Cert-GenAI-Dev: Insurance Claims Processing](#cert-genai-dev-insurance-claims-processing)
  - [Cert-GenAI-Dev-2: Enterprise AI Systems](#cert-genai-dev-2-enterprise-ai-systems)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Highlights](#project-highlights)
- [Architecture Patterns](#architecture-patterns)
- [Contact](#contact)

---

## 🎯 Overview

This repository contains advanced AWS Generative AI implementations completed as part of the **AWS Certified Generative AI Developer** certification. The projects demonstrate expertise in:

- **Foundation Model (FM) Integration**: Working with Claude 3 Sonnet, Claude Instant, Amazon Titan, and Amazon Nova models
- **Production ML Systems**: Implementing resilient, scalable, and cost-optimized AI architectures
- **RAG (Retrieval-Augmented Generation)**: Building vector stores with Amazon OpenSearch Serverless
- **MLOps**: Model benchmarking, A/B testing, lifecycle management, and automated deployment
- **AWS Services**: Bedrock, SageMaker, Lambda, API Gateway, Step Functions, DynamoDB, S3, CloudWatch, and more

---

## 🚀 Projects

### **Cert-GenAI-Dev: Insurance Claims Processing**

A complete proof-of-concept (POC) system for automated insurance claims processing using Amazon Bedrock foundation models.

#### 📂 Location
[`Cert-GenAI-Dev/Bonus_assignments/`](Cert-GenAI-Dev/Bonus_assignments/)

#### 🎯 Business Problem
Automate the extraction and summarization of unstructured insurance claim documents to reduce manual processing time and improve accuracy.

#### 🏗️ Architecture
- **Data Ingestion**: S3-based document storage with synthetic claim generation
- **Model Selection**: A/B testing between Claude 3 Sonnet (baseline) and Amazon Nova Micro (challenger)
- **Processing Pipeline**: Extract structured data → Generate summaries → Store results
- **Evaluation Framework**: Latency, quality, and cost metrics comparison

#### ✨ Key Features
- **Synthetic Data Generator**: Uses Faker library to create realistic test claims ([`synthetic-text-generator.ipynb`](Cert-GenAI-Dev/Bonus_assignments/synthetic-text-generator.ipynb))
- **Reusable Components**:
  - `PromptTemplateManager`: Centralized prompt management
  - `invoke_bedrock_model()`: Unified model invocation wrapper
  - `process_claim_with_model()`: End-to-end workflow orchestration
- **Multi-Model Evaluation**: Automated comparison of 2+ foundation models ([`poc-claims-v3.ipynb`](Cert-GenAI-Dev/Bonus_assignments/poc-claims-v3.ipynb))
- **S3 Integration**: Automatic upload of results to `s3://cert-genai-dev/bonus_1.1/outputs/`

#### 📊 Results
| Model | Avg Latency (s) | Extract (s) | Summary (s) | Cost per 1K Tokens |
|-------|----------------|-------------|-------------|-------------------|
| Claude 3 Sonnet | 14.82 | 1.93 | 10.08 | $0.003 |
| Amazon Nova Micro | 2.80 | 0.45 | 1.67 | <$0.001 |

**Outcome**: Nova Micro achieved **5.3x faster processing** at **>95% cost reduction** with comparable quality.

#### 🛠️ Technologies
- **AWS Services**: Bedrock, S3, IAM
- **Models**: Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`), Amazon Nova Micro (`amazon.nova-micro-v1:0`)
- **Python Libraries**: `boto3`, `faker`, `json`, `pathlib`

#### 📁 Key Files
- [`poc-claims-v3.ipynb`](Cert-GenAI-Dev/Bonus_assignments/poc-claims-v3.ipynb): Complete implementation with model comparison
- [`synthetic-text-generator.ipynb`](Cert-GenAI-Dev/Bonus_assignments/synthetic-text-generator.ipynb): Test data generation
- [`README.md`](Cert-GenAI-Dev/Bonus_assignments/README.md): Detailed architecture and setup guide

---

### **Cert-GenAI-Dev-2: Enterprise AI Systems**

Four advanced projects demonstrating production-grade AI system design with resilience, compliance, and MLOps best practices.

#### 📂 Location
[`Bonus_assignments/Cert-GenAI-Dev-2/`](Cert-GenAI-Dev/Bonus_assignments/Cert-GenAI-Dev-2/)

---

#### **Task 1.2: Resilient Financial Services AI Assistant**

Production-ready AI assistant with dynamic model routing, circuit breakers, and graceful degradation for regulated industries.

##### 🎯 Business Problem
Build a compliant, cost-effective AI system for financial services that handles failures gracefully while meeting regulatory requirements (FDIC disclaimers, Equal Housing Lender notices).

##### 🏗️ Architecture
```
API Gateway → Lambda (Model Abstraction Layer) → AppConfig → Bedrock Models
                ↓                                              ↓
         [Circuit Breaker]                            [Fallback Lambda]
                ↓                                              ↓
         [Step Functions]                         [Graceful Degradation Lambda]
```

##### ✨ Key Features
- **Foundation Model Benchmarking** ([Section 1-4](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/financials_ai_assistant.ipynb)):
  - Evaluates Claude Sonnet, Claude Instant, and Titan Express on 7 financial Q&A test cases
  - Measures latency, quality (word overlap similarity), and cost per 1K tokens
  - Generates [`model_evaluation_results.csv`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/) for data-driven selection

- **Dynamic Model Selection** ([Section 5-8](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/financials_ai_assistant.ipynb)):
  - AWS AppConfig integration for runtime model switching without redeployment
  - Use case-based routing (product questions, compliance, personalized outreach)
  - API Gateway + Lambda architecture ([`model_abstraction_lambda.py`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/model_abstraction_lambda.py))

- **Resilient System Design** ([Section 9-11](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/financials_ai_assistant.ipynb)):
  - **Circuit Breaker**: Step Functions state machine tracks failure rates in DynamoDB
  - **Fallback Model**: Titan Express as secondary option ([`fallback_model_lambda.py`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/fallback_model_lambda.py))
  - **Graceful Degradation**: Predefined regulation-compliant responses ([`graceful_degradation_lambda.py`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/graceful_degradation_lambda.py))
  - **Multi-Region HA**: CloudFormation template for Route 53 health checks

- **Model Customization & Lifecycle** ([Section 12-14](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/financials_ai_assistant.ipynb)):
  - SageMaker fine-tuning with HuggingFace Transformers ([`train.py`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/train.py))
  - A/B testing with traffic splitting (90/10)
  - Automated cleanup of old endpoints/models (>7 days)

##### 📊 Benchmark Results
| Model | Success Rate | Avg Latency (ms) | Avg Quality Score | Est. Cost ($) |
|-------|-------------|-----------------|------------------|---------------|
| Claude 3 Sonnet | 100% (7/7) | 2,547 | 0.412 | 0.003 |
| Claude Instant | 0% (0/7) | N/A | N/A | 0.000 |
| Titan Express | 100% (7/7) | 175 | 0.357 | 0.000 |

**Outcome**: Titan Express selected for production (66% faster, 100% success rate, comparable quality).

##### 🛠️ Technologies
- **AWS Services**: Bedrock, SageMaker, Lambda, API Gateway, AppConfig, Step Functions, DynamoDB, CloudWatch, Route 53, S3
- **Models**: Claude 3 Sonnet, Claude Instant, Titan Express, DistilGPT-2 (fine-tuned)
- **Python Libraries**: `boto3`, `transformers`, `datasets`, `pandas`

##### 📁 Key Files
- [`financials_ai_assistant.ipynb`](Cert-GenAI-Dev-2/Bonus_assignments/financials_ai_assistant.ipynb): Complete implementation (3,249 lines)
- [`train.py`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/train.py): SageMaker training script
- [`model_selection_strategy.json`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/model_selection_strategy.json): AppConfig routing rules
- [`cross_region_deployment.yaml`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/): CloudFormation template for HA
- [`README.md`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_2/README.md): Comprehensive architecture guide (389 lines)

---

#### **Task 1.3: Customer Feedback Analysis Pipeline**

End-to-end data validation and processing pipeline for sentiment analysis using AWS Glue, Lambda, and Amazon Comprehend.

##### 🎯 Business Problem
Automate the ingestion, validation, and sentiment analysis of customer feedback from multiple sources (CSV bulk data + individual text/JSON files).

##### 🏗️ Architecture
```
S3 (raw-data/) → EventBridge → Lambda (Validation) → Glue Crawler
                                         ↓
                                 Athena Queries ← Glue Data Catalog
                                         ↓
                         Lambda (Comprehend) → S3 (processed-data/)
                                         ↓
                                 CloudWatch Metrics & Reports
```

##### ✨ Key Features
- **Dual Data Sources**:
  - CSV bulk import: 96 pre-labeled customer reviews ([`clean-input-data.csv`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_3/))
  - Individual files: 3 test reviews (`.txt`, `.json`) for real-time processing

- **Automated Validation**:
  - Lambda function triggered by S3 events
  - Schema validation, data quality checks, deduplication
  - Rejects invalid files (logs to CloudWatch)

- **Sentiment Analysis Pipeline**:
  - AWS Glue Crawler catalogs validated data
  - Athena queries for data exploration
  - Lambda + Comprehend for batch sentiment detection
  - Outputs: Sentiment scores, key phrases, entity recognition

- **Unified Reporting**:
  - Combines test files + CSV data into single report
  - Sentiment distribution: 55.6% Positive, 22.2% Neutral, 22.2% Negative
  - Top phrases: "this restaurant" (frequency: 3), "good food" (frequency: 2)

##### 📊 Pipeline Statistics
- **Total Files Processed**: 99 (3 test + 96 CSV)
- **Success Rate**: 100%
- **Avg Processing Time**: 1.9s per file
- **CloudWatch Metrics**: Custom namespace `CustomerFeedback/Processing`

##### 🛠️ Technologies
- **AWS Services**: S3, Lambda, Glue, Athena, Comprehend, EventBridge, CloudWatch, IAM
- **Python Libraries**: `boto3`, `pandas`, `json`

##### 📁 Key Files
- [`customer_feedback.ipynb`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_3/customer_feedback.ipynb): Complete pipeline (5,132 lines)
- [`README.md`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_3/README.md): Architecture diagram and implementation guide (489 lines)
- [`reports/unified_comprehend_report_*.json`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_3/reports/): Sentiment analysis results

---

#### **Task 1.4: Enterprise RAG System with OpenSearch**

Production-grade Retrieval-Augmented Generation (RAG) system using Amazon Bedrock Knowledge Bases and OpenSearch Serverless.

##### 🎯 Business Problem
Build a scalable RAG system for querying 50 subreddits (technology, science, programming) with semantic search and A/B testing capabilities.

##### 🏗️ Architecture
```
Reddit Dataset (Kaggle) → Lambda (Document Processor) → S3 (docs-bucket)
                                                             ↓
                                                 OpenSearch Serverless
                                                             ↓
                                             Bedrock Knowledge Base
                                                             ↓
                                      Query Interface (Jupyter Notebook)
```

##### ✨ Key Features
- **Vector Store Setup**:
  - OpenSearch Serverless collection with 1536-dimension embeddings
  - Security policies: Encryption, network access, data access
  - Automatic index creation (`bedrock-knowledge-base-default-index`)

- **Knowledge Base Integration**:
  - Bedrock Knowledge Base with S3 data source
  - Lambda function for document preprocessing and metadata tagging
  - DynamoDB table (`reddit-kb-metadata`) for tracking ingestion status

- **Advanced Querying**:
  - Semantic search with Titan Embeddings G1 (`amazon.titan-embed-text-v1`)
  - Response generation with Claude 3 Sonnet/Haiku
  - A/B testing framework for model comparison

- **Complete Cleanup Script**:
  - [`AWS-Cleanup.ipynb`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_4/AWS-Cleanup.ipynb) for resource teardown
  - Deletes Knowledge Base, OpenSearch collection, Lambda, S3 buckets, DynamoDB, IAM roles

##### 📊 Performance Metrics
- **Documents Indexed**: 50 subreddits (2011-2024)
- **Query Latency**: <500ms (semantic search)
- **Embedding Model**: Titan Embeddings G1 (1536 dims)
- **Response Models**: Claude 3 Sonnet (quality), Claude 3 Haiku (speed)

##### 🛠️ Technologies
- **AWS Services**: Bedrock Knowledge Bases, OpenSearch Serverless, Lambda, S3, DynamoDB, IAM, CloudWatch
- **Models**: Titan Embeddings G1, Claude 3 Sonnet, Claude 3 Haiku
- **Python Libraries**: `boto3`, `opensearch-py`, `pandas`

##### 📁 Key Files
- [`Reddit=Vector-Store.ipynb`](Cert-GenAI-Dev-2/Bonus_assignments/task_1_4/Reddit=Vector-Store.ipynb): Complete implementation (4,978 lines)
- [`AWS-Cleanup.ipynb`](Bonus_assignments/Cert-GenAI-Dev-2/task_1_4/AWS-Cleanup.ipynb): Resource cleanup automation
- [`instructions.md`](Bonus_assignments/Cert-GenAI-Dev-2/task_1_4/instructions.md): Setup and deployment guide
- [`README.md`](Bonus_assignments/Cert-GenAI-Dev-2/task_1_4/README.md): Comprehensive architecture and usage (319 lines)

---

## 💡 Technical Skills Demonstrated

### **1. Foundation Model Expertise**
- ✅ Model selection and benchmarking (latency, quality, cost)
- ✅ Prompt engineering for enterprise use cases
- ✅ Multi-model comparison and A/B testing
- ✅ Integration with 5+ Bedrock models (Claude, Titan, Nova)

### **2. MLOps & Production Systems**
- ✅ SageMaker fine-tuning with HuggingFace Transformers
- ✅ Automated model deployment and versioning
- ✅ A/B testing with traffic splitting (90/10)
- ✅ Model lifecycle management (cleanup of old resources)
- ✅ Circuit breaker pattern for failure handling

### **3. AWS Architecture Patterns**
- ✅ Serverless workflows (Lambda + Step Functions)
- ✅ API Gateway integration with caching and retries
- ✅ Dynamic configuration with AppConfig
- ✅ Multi-region high availability (Route 53 health checks)
- ✅ Event-driven processing (S3 → EventBridge → Lambda)

### **4. Data Engineering**
- ✅ S3-based data lakes with structured ingestion
- ✅ AWS Glue for data cataloging and ETL
- ✅ Athena for ad-hoc SQL queries
- ✅ DynamoDB for metadata tracking and circuit breaker state

### **5. RAG & Vector Databases**
- ✅ OpenSearch Serverless setup with security policies
- ✅ Bedrock Knowledge Base integration
- ✅ Semantic search with embeddings (1536-dim Titan)
- ✅ Document preprocessing and chunking strategies

### **6. Monitoring & Observability**
- ✅ CloudWatch custom metrics and dashboards
- ✅ Automated alerting (SNS notifications)
- ✅ Cost tracking and optimization ($0.003 per request)
- ✅ Comprehensive logging for debugging

### **7. Compliance & Security**
- ✅ FDIC disclaimers and Equal Housing Lender notices
- ✅ Graceful degradation for regulated industries
- ✅ IAM least privilege policies
- ✅ Encryption at rest (S3, DynamoDB) and in transit (HTTPS)

---

## 🛠️ Technologies Used

### **AWS Services**
| Service | Use Case |
|---------|----------|
| **Amazon Bedrock** | Foundation model inference (Claude, Titan, Nova) |
| **SageMaker** | Model fine-tuning, deployment, and lifecycle management |
| **Lambda** | Serverless compute for validation, processing, and API handlers |
| **API Gateway** | RESTful API endpoints with throttling and caching |
| **Step Functions** | Orchestration for circuit breaker and retry logic |
| **OpenSearch Serverless** | Vector store for RAG with semantic search |
| **S3** | Data lake for documents, models, and results |
| **DynamoDB** | State management (circuit breaker, metadata tracking) |
| **Glue** | Data cataloging, ETL, and schema discovery |
| **Athena** | SQL queries on S3 data |
| **Comprehend** | Sentiment analysis, entity recognition, key phrases |
| **AppConfig** | Dynamic configuration without redeployment |
| **CloudWatch** | Metrics, logs, alarms, and dashboards |
| **EventBridge** | Event-driven triggers (S3 uploads) |
| **Route 53** | Health checks and DNS failover for HA |
| **IAM** | Fine-grained access control |

### **Foundation Models**
- **Claude 3 Sonnet** (`anthropic.claude-3-sonnet-20240229-v1:0`): High-quality responses
- **Claude 3 Haiku** (`anthropic.claude-3-haiku-20240307-v1:0`): Low-latency queries
- **Claude Instant** (`anthropic.claude-instant-v1`): Cost-optimized option
- **Amazon Titan Express** (`amazon.titan-text-express-v1`): Fast, reliable baseline
- **Amazon Nova Micro** (`amazon.nova-micro-v1:0`): Ultra-low-cost challenger
- **Titan Embeddings G1** (`amazon.titan-embed-text-v1`): 1536-dim vector embeddings

### **Python Libraries**
```python
boto3              # AWS SDK
transformers       # HuggingFace models
datasets           # SageMaker training data
pandas             # Data manipulation
faker              # Synthetic data generation
opensearch-py      # OpenSearch client
json, pathlib      # Core utilities
```

---

## 🚀 Getting Started

### **Prerequisites**
1. AWS Account with Bedrock model access enabled (us-east-1 region)
2. Python 3.9+ with Jupyter Notebook/JupyterLab
3. AWS CLI configured with credentials (`aws configure`)
4. IAM permissions for: Bedrock, SageMaker, Lambda, S3, DynamoDB, OpenSearch, Glue

### **Installation**
```bash
# Clone the repository
git clone https://github.com/aminajavaid30/RAG-Ingestion.git
cd RAG-Ingestion/AWS

# Install dependencies
pip install boto3 transformers datasets pandas faker opensearch-py

# Configure AWS credentials
aws configure
# Enter AWS Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

### **Quick Start: Insurance Claims Processing**
```bash
cd Cert-GenAI-Dev
jupyter notebook poc-claims-v3.ipynb
# Execute cells 1-7 to run the complete pipeline
```

### **Quick Start: Financial AI Assistant**
```bash
cd Cert-GenAI-Dev-2/task_1_2
jupyter notebook financials_ai_assistant.ipynb
# Execute cells sequentially (Parts 1-4)

# Test the deployed API
curl -X POST https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a 401k?", "use_case": "product_question"}'
```

---

## 🏆 Project Highlights

### **Cost Optimization**
- **95% cost reduction** by switching from Claude to Nova Micro in claims processing
- **Titan Express** selected over Claude Instant (100% reliability, $0 incremental cost)
- Circuit breaker prevents cascading failures (saves on retries)

### **Performance Improvements**
- **5.3x faster** document processing (Nova Micro: 2.8s vs. Claude: 14.82s)
- **66% latency reduction** in financial Q&A (Titan: 175ms vs. Claude: 2,547ms)
- **<500ms** semantic search queries in RAG system

### **Reliability Achievements**
- **100% success rate** with fallback + degradation layers
- **Multi-region HA** with Route 53 health checks (<1min failover)
- **Circuit breaker** stops traffic to failing models after 3 consecutive errors

### **Compliance Features**
- FDIC disclaimers, Equal Housing Lender notices
- Audit trail for all model invocations (CloudWatch Logs)
- Graceful degradation returns regulation-compliant static responses

---

## 🏗️ Architecture Patterns

### **1. Lambda + API Gateway + AppConfig**
```
User → API Gateway → Lambda → AppConfig (routing rules) → Bedrock
```
**Benefits**: No redeployment for model changes, caching, throttling

### **2. Circuit Breaker with Step Functions**
```
Lambda → Step Functions → DynamoDB (state) → Fallback Lambda
```
**Benefits**: Prevents cascading failures, automatic recovery

### **3. Event-Driven Processing**
```
S3 Upload → EventBridge → Lambda (validation) → Glue Crawler → Athena
```
**Benefits**: Real-time processing, scalable ingestion

### **4. RAG with OpenSearch**
```
Query → Bedrock KB → OpenSearch (vector search) → Titan Embeddings → Claude (generation)
```
**Benefits**: Semantic search, up-to-date information retrieval

---

## 📧 Contact

**Amina Javaid**  
AWS Certified Cloud Practitioner | Aspiring Solutions Architect | GenAI Enthusiast

- **GitHub**: [@aminajavaid30](https://github.com/aminajavaid30)
- **LinkedIn**: [linkedin.com/in/amina-javaid](https://linkedin.com/in/amina-javaid)
- **Email**: aminajavaid30@gmail.com

---

## 📝 License

This project is provided as-is for educational purposes as part of the AWS Certified Generative AI Developer certification.

---

## 🙏 Acknowledgments

- AWS for providing Bedrock and comprehensive GenAI services
- HuggingFace for Transformers library and model hub
- Anthropic for Claude models
- Amazon for Titan and Nova models

---

**⭐ If you found these projects helpful, please star the repository!**


