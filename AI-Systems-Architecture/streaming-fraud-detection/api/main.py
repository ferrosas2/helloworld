"""
Cloud Run FastAPI Application: Real-Time Fraud Detection & Deep Analysis API.

This service provides two scoring paths accessible via REST:

1. FAST PATH: /api/v1/score-transaction
   - Computes features + calls Vertex AI Online Prediction
   - Returns fraud_probability_score + risk_decision in ~50ms
   - Used by Dataflow for inline scoring or by external clients

2. DEEP PATH: /api/v1/deep-analysis
   - RAG-based explainable analysis for flagged transactions
   - Retrieves similar historical fraud via Vector Search
   - Generates structured risk explanation via Gemini 2.5 Flash
   - Returns ~1-2 seconds with full explainability

3. BATCH PATH: /api/v1/score-batch
   - Scores up to 1000 transactions in a single request
   - Efficient for re-scoring historical data or bulk imports

Deployment:
    Cloud Run (serverless, scale-to-zero, autoscaling)
    Min instances: 1 (production), 0 (staging)
    Max instances: 50 (production)
    Concurrency: 80 requests per instance
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.schema import (
    TransactionRequest,
    BatchTransactionRequest,
    DeepAnalysisRequest,
    FraudScoreResponse,
    BatchScoreResponse,
    DeepAnalysisResponse,
    HealthResponse,
    ReadinessResponse,
)
from src.scoring import AsyncFraudScorer
from src.deep_analysis import DeepFraudAnalyzer
from src.feature_engineering import FeatureComputer
from src.monitoring import (
    setup_cloud_logging,
    monitor,
    track_latency,
    log_operation,
)

# Configure structured logging for Cloud Run → Cloud Logging
setup_cloud_logging(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan (Startup / Shutdown)
# =============================================================================

# Module-level handles for GCP-backed components
scorer: Optional[AsyncFraudScorer] = None
analyzer: Optional[DeepFraudAnalyzer] = None
feature_computer: Optional[FeatureComputer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize GCP-backed clients on startup, release on shutdown.
    
    Fails fast if GCP resources are unreachable so Cloud Run surfaces
    a clear startup error instead of failing on the first request.
    """
    global scorer, analyzer, feature_computer

    logger.info("=" * 65)
    logger.info("  Streaming Fraud Detection API — Starting Up")
    logger.info("=" * 65)
    logger.info(f"  Project:              {settings.GCP_PROJECT_ID}")
    logger.info(f"  Region:               {settings.GCP_REGION}")
    logger.info(f"  Prediction Endpoint:  {settings.VERTEX_PREDICTION_ENDPOINT_ID}")
    logger.info(f"  Vector Search Index:  {settings.VERTEX_INDEX_ID}")
    logger.info(f"  GCS Bucket:           {settings.GCS_BUCKET_NAME}")
    logger.info(f"  Gemini Model:         {settings.GEMINI_MODEL}")
    logger.info(f"  High Risk Threshold:  {settings.HIGH_RISK_THRESHOLD}")
    logger.info(f"  Medium Risk Threshold:{settings.MEDIUM_RISK_THRESHOLD}")
    logger.info("=" * 65)

    try:
        # Initialize feature computer (stateless, always succeeds)
        feature_computer = FeatureComputer()

        # Initialize Vertex AI scorer (connects to prediction endpoint)
        scorer = AsyncFraudScorer(
            project_id=settings.GCP_PROJECT_ID,
            endpoint_id=settings.VERTEX_PREDICTION_ENDPOINT_ID,
            region=settings.GCP_REGION,
        )
        await scorer.initialize()

        # Initialize deep analysis engine (Vector Search + Gemini)
        analyzer = DeepFraudAnalyzer()

        logger.info("All GCP backends initialized successfully.")

    except Exception as e:
        logger.critical(
            f"FATAL: Startup failed during GCP initialization: {str(e)}",
            exc_info=True,
        )
        raise

    yield

    logger.info("Streaming Fraud Detection API shutting down.")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Streaming Fraud Detection API",
    description=(
        "Real-time POS transaction fraud scoring with dual-path architecture: "
        "fast ML scoring (~50ms) + explainable RAG analysis (~1-2s) for flagged transactions."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# CORS middleware (configure for your frontend domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Readiness Probes
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health_check():
    """
    Liveness probe for Cloud Run / load balancers.
    Returns 200 if the process is running (does not check dependencies).
    """
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/readiness", response_model=ReadinessResponse, tags=["Infrastructure"])
async def readiness_check():
    """
    Readiness probe — verifies all GCP dependencies are accessible.
    Used by Cloud Run to determine when the instance can receive traffic.
    """
    if scorer is None or analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: GCP clients not initialized."
        )

    return ReadinessResponse(
        status="ready",
        vertex_ai_prediction="connected",
        vertex_ai_vector_search="connected",
        gemini_llm="connected",
        pubsub="connected",
    )


@app.get("/metrics", tags=["Infrastructure"])
async def metrics_endpoint():
    """
    Internal metrics endpoint for monitoring dashboards.
    Returns latency percentiles, throughput, and SLO status.
    """
    return {
        "metrics": monitor.get_summary(),
        "slo": monitor.check_slo(),
    }


# =============================================================================
# Fast Path: Real-Time Fraud Scoring (~50ms)
# =============================================================================

@app.post(
    "/api/v1/score-transaction",
    response_model=FraudScoreResponse,
    tags=["Scoring"],
)
async def score_transaction(request: TransactionRequest):
    """
    Score a single POS transaction for fraud probability (fast path).
    
    **Process Flow:**
    1. Compute per-element features (amount, merchant risk, time-of-day)
    2. Call Vertex AI Online Prediction endpoint
    3. Apply risk routing thresholds
    4. Return structured score + decision
    
    **Target Latency:** < 50ms p99
    
    **Returns:**
    - fraud_probability_score: 0.0 (legitimate) to 1.0 (certain fraud)
    - risk_decision: APPROVE | REVIEW | FLAG
    - risk_tier: low | medium | high
    """
    if scorer is None or feature_computer is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    logger.info(
        f"Scoring transaction: {request.transaction_id}",
        extra={"transaction_id": request.transaction_id},
    )

    try:
        # Step 1: Compute features
        transaction_dict = request.dict()
        features = feature_computer.compute_features(transaction_dict)

        # Step 2: Score via Vertex AI
        with log_operation("fraud_scoring", transaction_id=request.transaction_id):
            score_result = await scorer.score_transaction(features)

        fraud_score = score_result["fraud_probability"]
        monitor.record_latency("scoring_latency_ms", score_result["latency_ms"])
        monitor.increment_counter("transactions_processed")

        # Step 3: Route by risk level
        if fraud_score >= settings.HIGH_RISK_THRESHOLD:
            risk_decision = "FLAG"
            risk_tier = "high"
            monitor.increment_counter("transactions_flagged")
        elif fraud_score >= settings.MEDIUM_RISK_THRESHOLD:
            risk_decision = "REVIEW"
            risk_tier = "medium"
            monitor.increment_counter("transactions_reviewed")
        else:
            risk_decision = "APPROVE"
            risk_tier = "low"
            monitor.increment_counter("transactions_approved")

        return FraudScoreResponse(
            transaction_id=request.transaction_id,
            fraud_probability_score=fraud_score,
            risk_decision=risk_decision,
            risk_tier=risk_tier,
            scoring_latency_ms=score_result["latency_ms"],
            scoring_method=score_result["scoring_method"],
            model_version=score_result["model_version"],
        )

    except Exception as e:
        monitor.increment_counter("scoring_errors")
        logger.error(
            f"Scoring failed for {request.transaction_id}: {str(e)}",
            extra={"transaction_id": request.transaction_id, "error_type": type(e).__name__},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error during fraud scoring.",
        )


# =============================================================================
# Deep Path: RAG-Based Explainable Analysis (~1-2s)
# =============================================================================

@app.post(
    "/api/v1/deep-analysis",
    response_model=DeepAnalysisResponse,
    tags=["Deep Analysis"],
)
async def deep_analysis(request: DeepAnalysisRequest):
    """
    Perform RAG-based deep fraud analysis on a flagged transaction.
    
    **Process Flow:**
    1. Embed transaction context using Vertex AI Embeddings (text-embedding-004)
    2. Retrieve top-3 similar historical fraud cases from Vector Search
    3. Send transaction + context to Gemini 2.5 Flash for risk explanation
    4. Return structured risk assessment with cited evidence
    
    **Target Latency:** < 2000ms p99
    
    **When to use:**
    Called asynchronously after the fast path flags a transaction (score >= 0.7).
    The output enriches the case file for human fraud analysts.
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Deep analysis service not ready.")

    logger.info(
        f"Deep analysis requested: {request.transaction_id}",
        extra={"transaction_id": request.transaction_id, "fraud_score": request.fraud_probability_score},
    )

    try:
        monitor.increment_counter("deep_analysis_triggered")

        with log_operation("deep_analysis", transaction_id=request.transaction_id):
            result = analyzer.analyze_transaction(
                transaction_id=request.transaction_id,
                transaction_text=request.transaction_text,
                fraud_score=request.fraud_probability_score,
                top_k=3,
            )

        monitor.record_latency("deep_analysis_latency_ms", result.get("analysis_latency_ms", 0))

        return DeepAnalysisResponse(**result)

    except Exception as e:
        monitor.increment_counter("deep_analysis_errors")
        logger.error(
            f"Deep analysis failed for {request.transaction_id}: {str(e)}",
            extra={"transaction_id": request.transaction_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error during deep analysis.",
        )


# =============================================================================
# Batch Path: Bulk Scoring
# =============================================================================

@app.post(
    "/api/v1/score-batch",
    response_model=BatchScoreResponse,
    tags=["Scoring"],
)
async def score_batch(request: BatchTransactionRequest):
    """
    Score a batch of transactions (up to 1000) in a single request.
    
    More efficient than individual calls for:
    - Re-scoring historical transactions after model update
    - Bulk imports from external fraud management systems
    - Backfill scoring for transactions that missed the streaming path
    """
    if scorer is None or feature_computer is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    import time
    start_time = time.time()

    try:
        # Compute features for all transactions
        feature_batch = []
        for txn in request.transactions:
            features = feature_computer.compute_features(txn.dict())
            feature_batch.append(features)

        # Batch score via Vertex AI
        score_results = await scorer.score_batch(feature_batch)

        # Build responses
        results = []
        for txn, score_result in zip(request.transactions, score_results):
            fraud_score = score_result["fraud_probability"]

            if fraud_score >= settings.HIGH_RISK_THRESHOLD:
                risk_decision, risk_tier = "FLAG", "high"
            elif fraud_score >= settings.MEDIUM_RISK_THRESHOLD:
                risk_decision, risk_tier = "REVIEW", "medium"
            else:
                risk_decision, risk_tier = "APPROVE", "low"

            results.append(FraudScoreResponse(
                transaction_id=txn.transaction_id,
                fraud_probability_score=fraud_score,
                risk_decision=risk_decision,
                risk_tier=risk_tier,
                scoring_latency_ms=score_result["latency_ms"],
                scoring_method=score_result["scoring_method"],
                model_version=score_result["model_version"],
            ))

        total_latency = (time.time() - start_time) * 1000
        monitor.increment_counter("transactions_processed", len(results))

        return BatchScoreResponse(
            results=results,
            batch_size=len(results),
            total_latency_ms=round(total_latency, 2),
        )

    except Exception as e:
        logger.error(f"Batch scoring failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error during batch scoring.",
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info",
    )
