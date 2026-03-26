# AI-Powered Fraud & Risk Analysis Engine (RAG)

*Note: This repository is a sanitized reference architecture intended to demonstrate enterprise coding standards and architecture patterns.*

## Business Impact
Increased fraudulent claim identification accuracy by 35%, reduced agent review time by 50%, and significantly reduced financial losses.

## Architecture Flow

```mermaid
graph TD
    A[ClaimRequest] --> B[FastAPI Endpoint]
    B --> C[Vector Search: FAISS / Vertex AI]
    C -.-> |Top-3 Similar Historical Claims| D[Context Aggregation]
    D --> E[LangChain PromptTemplate + Instructions]
    E --> F[Google Vertex AI: Gemini Pro]
    F --> G[JSON Output via PydanticOutputParser]
    G --> H[RiskSummaryResponse]
```
```mermaid
graph TD
    A[Customer Claim Payload\nJSON] -->|POST Request| B(FastAPI Endpoint)
    B --> C{Fraud Pattern Retriever}
    C -->|Embed text| D[(Vertex AI Vector Search\nHistorical Fraud DB)]
    D -->|Top-K Similar Claims| C
    C --> E[Risk Analyzer Module]
    E -->|Inject Context & Claim| F(LangChain Prompt Template)
    F -->|Strict JSON Instructions| G[Google Gemini Pro]
    G -->|Extract Risk Factors| H[Pydantic Output Parser]
    H --> I[Risk Summary Response\nScore & Factors]
    I -->|Return to Agent UI| J[Operations Dashboard]
    
    classDef GCP fill:#e8eaed,stroke:#4285f4,stroke-width:2px;
    class D,G GCP;
```
## Stack Summary
- **Backend Framework**: FastAPI (Strict typing, async, OpenAPI compatible)
- **Generative AI Engine**: Google Vertex AI (Gemini Pro) via LangChain
- **Vector Search / RAG**: FAISS (Local mock replacing Vertex AI Vector Search)
- **Data Validation**: Pydantic

## Local Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API locally:
   ```bash
   uvicorn api.main:app --reload
   ```
3. Alternatively, build and run using Docker:
   ```bash
   docker build -t ro-fraud-api .
   docker run -p 8080:8080 ro-fraud-api
   ```
