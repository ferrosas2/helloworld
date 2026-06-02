import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from src.schema import ClaimRequest, RiskSummaryResponse
from src.retrieval import FraudPatternRetriever
from src.generation import RiskAnalyzerLLM
from src.config import settings
import uvicorn

# Configure professional standard logging for GCP operations and requests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-level handles for the GCP-backed components. They are populated on
# application startup (see lifespan) rather than at import time, so the module
# can be imported in tests/CI without live GCP connectivity.
retriever: Optional[FraudPatternRetriever] = None
analyzer: Optional[RiskAnalyzerLLM] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize GCP-backed clients once when the server starts.

    Fails fast if GCP resources are unreachable so Cloud Run surfaces a clear
    startup error instead of failing on the first request.
    """
    global retriever, analyzer

    logger.info("=========================================================")
    logger.info("Initializing Live GCP RO-Fraud RAG API System Bootstrap")
    logger.info(f"Targeting Google Cloud Project: {settings.GCP_PROJECT_ID}")
    logger.info(f"Deployment Region:             {settings.GCP_REGION}")
    logger.info(f"Vector Index ID:               {settings.VERTEX_INDEX_ID}")
    logger.info(f"Vector Endpoint ID:            {settings.VERTEX_ENDPOINT_ID}")
    logger.info(f"Google Cloud Storage Bucket:   {settings.GCS_BUCKET_NAME}")
    logger.info("=========================================================")

    try:
        retriever = FraudPatternRetriever()
        analyzer = RiskAnalyzerLLM()
        logger.info("GCP backend clients and models initialized successfully.")
    except Exception as init_err:
        logger.critical(f"FATAL: Application bootstrap failed while preparing GCP connectivity: {str(init_err)}")
        raise

    yield

    # Nothing to clean up explicitly; clients are released on process exit.
    logger.info("Shutting down RO-Fraud RAG API.")


app = FastAPI(
    title="AI-Powered Fraud & Risk Analysis Engine",
    description="Sanitized reference architecture for AI-assisted fraud detection using RAG on live GCP services.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for Cloud Run and load balancers.
    Returns 200 OK if the service process is running.
    """
    return {"status": "healthy", "service": "ro-fraud-api"}


@app.get("/readiness")
async def readiness_check():
    """
    Readiness probe to verify GCP dependencies are accessible.
    Validates connectivity to Vertex AI Vector Search.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready: clients not initialized.")
    try:
        # Lightweight query to confirm Vector Search connectivity.
        retriever.vector_store.similarity_search("test", k=1)
        return {
            "status": "ready",
            "vertex_vector_search": "connected",
            "vertex_ai_embeddings": "connected",
            "gemini_llm": "connected",
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.post("/api/v1/analyze-claim", response_model=RiskSummaryResponse)
async def analyze_claim_endpoint(request: ClaimRequest):
    """
    Main endpoint for fraud risk analysis using RAG architecture.

    **Process Flow:**
    1. Retrieve similar historical fraudulent claims from Vertex AI Vector Search
    2. Format current claim details with context
    3. Generate risk assessment using Gemini (configurable, default Gemini 2.5 Flash)
    4. Return structured risk score with explainability

    **Returns:**
    - fraud_probability_score: Float between 0.0-1.0
    - risk_factors: List of identified red flags
    - executive_summary: Human-readable explanation
    """
    if retriever is None or analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready: clients not initialized.")

    logger.info(f"Received claim analysis request for Claim ID: {request.claim_id}")
    try:
        # 1. Retrieve historical context
        historical_context = retriever.get_similar_historical_claims(request.claim_text)

        # 2. Format claim details
        claim_details = (
            f"Claim ID: {request.claim_id}, "
            f"Customer ID: {request.customer_id}, "
            f"Amount: ${request.claim_amount}, "
            f"Text: {request.claim_text}"
        )

        # 3. Generate risk summary via LLM
        risk_summary_dict = analyzer.analyze_claim(claim_details, historical_context)

        # 4. Return structured response
        return RiskSummaryResponse(**risk_summary_dict)

    except Exception as e:
        logger.error(f"Error processing claim {request.claim_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during risk analysis. LLM or formatting failure.",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
