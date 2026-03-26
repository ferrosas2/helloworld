# Logistics & Fintech GenAI RAG System

Enterprise-grade Retrieval-Augmented Generation (RAG) architecture built for supply chain visibility and financial operations. This repository demonstrates a production-tier approach to querying unstructured documents, specifically targeting complex records like Bills of Lading and Freight Claims.

## Business Overview

In the fast-paced logistics and fintech sectors, processing multi-page structured and unstructured documents is a heavy manual bottleneck. This RAG system accelerates operations by enabling conversational analytical queries over document repositories.

**Key Value Drivers:**
- **Reduces manual review time** on complex freight documents from minutes to seconds.
- **Accelerates claims processing** by automating compliance and data extraction.
- **Mitigates risk** by drastically reducing human error in data review.

## Technical Architecture

This repository scaffolds a modern, scalable AI stack optimized for cloud-native deployment:

- **Generation & Embeddings:** Google Cloud Vertex AI (Gemini 1.5 Pro, TextEmbedding-Gecko).
- **Orchestration:** LangChain for modular connection logic.
- **Vector Database:** Local FAISS implementation for lightweight semantic caching and retrieval optimization.
- **Serving Layer:** FastAPI application, ideal for asynchronous handling and rapid execution.
- **Hosting Target:** Containerized via Docker for deployment on Google Cloud Run or Kubernetes.

## Design Philosophy

This codebase diverges from exploratory Jupyter Notebooks and emphasizes robust software engineering practices required for Tier-1 production systems:

- **Modular Design:** Strict adherence to the Single Responsibility Principle, cleanly separating configuration, ingestion, generation, and API routing.
- **PEP8 Compliance & Strict Typing:** Leverages Python 3.10+ type hinting and Pydantic validators ensuring maintainable code and robust data structures.
- **Stringent Guardrails:** The `PromptTemplate` enforces deterministic output suitable for financial compliance, applying zero-temperature and explicit instruction tuning to negate AI hallucinations ("Insufficient data to answer").
- **Security-First Containerization:** Uses lightweight distros (`python:3.10-slim`) and explicitly provisions non-root users (`appuser`) conforming to enterprise container security standards.

## Quickstart

### Prerequisites
- Python 3.10+
- Google Cloud Project with Vertex AI API enabled
- Application Default Credentials (ADC) configured locally

### Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variables (or rely on a `.env` file mapping to `src/config.py`).
4. Run the API locally:
   ```bash
   uvicorn api.main:app --reload --port 8080
   ```