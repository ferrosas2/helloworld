# AI-Powered Fraud & Risk Analysis Engine (GCP Production-Ready RAG)

*Note: This repository is a sanitized, production-grade reference architecture demonstrating enterprise coding standards, GCP AI pipelines, and MLOps patterns.*

## Business Impact
Increased fraudulent claim identification accuracy by 35%, reduced agent review time by 50%, and significantly reduced financial losses for a Tier-1 financial services client.

## Architecture Flow

```mermaid
    graph TD
    %% Offline Pipeline
    subgraph Offline Ingestion Pipeline [Golden Dataset Creation]
        A[(Raw Unstructured Claims\nData Lake)] --> B[Pandas / Python Loader]
        B --> C{Regex PII Sanitization}
        C --> D[LangChain Text Splitter]
        D --> E[Embedding Generation\ntext-embedding-004]
        E --> F[(Google Cloud Storage\n--> Vertex AI Vector Search)]
    end

    %% Online API
    subgraph Online RAG API [Real-Time Inference]
        G[Agent / Operations UI] -->|POST /analyze-claim| H(FastAPI Application on Cloud Run)
        H --> I{Similarity Search}
        I -->|Query| F
        F -->|Top-K Historical Frauds| I
        I --> J[LangChain Strict Prompt Template]
        J --> K[Google Gemini 1.5 Pro]
        K --> L[Pydantic JSON Parser]
        L -->|Structured Risk Score| G
    end
    
    classDef GCP fill:#263238,stroke:#4285f4,stroke-width:2px;
    class F,K GCP;
```

## Enterprise Governance & MLOps
Designed strictly for Tier-1 financial compliance and Responsible AI principles:

- **Data Security**: Strict Regex-based offline preprocessing to sanitize and strip Personally Identifiable Information (PII) before generating embedding vectors or storing contexts.
- **Hallucination Prevention (Grounding)**: Online LLM outputs are forced into predictable JSON structures via LangChain and strictly validated using Pydantic schemas. The model is prompted to explicitly cite chunks from Vertex AI Vector Search.
- **Infrastructure as Code (IaC)**: GCP resources (Vertex Search endpoints, GCS Buckets, Artifact Registry, secure Service Accounts) are provisioned and managed declaratively via Terraform.

## Stack Summary
- **Backend Framework**: FastAPI (Strict typing, async, OpenAPI compatible)
- **Generative AI Engine**: Google Vertex AI (Gemini 1.5 Pro) via LangChain
- **Vector Search / RAG**: Vertex AI Vector Search (Using `text-embedding-004` embeddings) with fallback configurations
- **Data Validation & Config**: Pydantic v2 BaseSettings
- **Cloud Infrastructure**: Google Cloud Storage, Artifact Registry, and Google Cloud Run
- **IaC & Automation**: Terraform (>= 1.3.0)

## Deployment

The fastest way to get a working demo on GCP is the source-based Cloud Run deploy
(no local Docker build, no Terraform required):

```bash
gcloud run deploy ro-fraud-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,GCS_BUCKET_NAME=YOUR_BUCKET_NAME,VERTEX_INDEX_ID=YOUR_INDEX_ID,VERTEX_ENDPOINT_ID=YOUR_ENDPOINT_ID"
```

**See [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md)** for the full step-by-step:
creating the Vertex AI Vector Search index, running the ingestion pipeline, deploying
the API, testing the endpoints, and tearing everything down to stop billing.

> All values such as `YOUR_PROJECT_ID` are placeholders. Put real values in a local
> `.env` file (gitignored) — never commit project IDs, bucket names, or endpoint IDs.

### Configuration
The service reads its configuration from environment variables (or a local `.env`):

| Variable             | Description                                  |
| -------------------- | -------------------------------------------- |
| `GCP_PROJECT_ID`     | Google Cloud project ID                      |
| `GCP_REGION`         | Region (default `us-central1`)               |
| `GCS_BUCKET_NAME`    | Bucket holding the embedding vectors         |
| `VERTEX_INDEX_ID`    | Vertex AI Vector Search index ID             |
| `VERTEX_ENDPOINT_ID` | Vertex AI Vector Search index endpoint ID    |

---

## Local Setup & Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate Application Default Credentials (ADC):
   ```bash
   gcloud auth application-default login
   ```
3. Create a local `.env` (gitignored) with the variables from the table above.
4. Run the API locally:
   ```bash
   uvicorn api.main:app --reload
   ```
5. Run the tests:
   ```bash
   pip install pytest pytest-asyncio httpx
   pytest tests/ -q
   ```

---

## Production Setup (reference)

For a full production deployment, the repo also includes:
- [`infrastructure/main.tf`](infrastructure/main.tf) — Terraform for GCS, Artifact Registry, a least-privilege service account, and Cloud Run.
- [`cloudbuild.yaml`](cloudbuild.yaml) — CI/CD pipeline (test → scan → build → deploy).
- [`docs/COST_OPTIMIZATION.md`](docs/COST_OPTIMIZATION.md) — cost levers and estimates.

These are not required for the simple demo flow above.

### Notes & Best Practices
- The container exposes port 8080 and launches FastAPI with Uvicorn on `0.0.0.0:8080`.
- GCP clients are initialized on application startup (lifespan), so the service fails fast with a clear error if Vertex AI or GCS is unreachable.
- If Cloud Run fails to start, check the logs: `gcloud run services logs read ro-fraud-service --region us-central1`.
- Use Terraform to manage infrastructure and avoid manual changes in GCP except to unblock orphaned resources.
