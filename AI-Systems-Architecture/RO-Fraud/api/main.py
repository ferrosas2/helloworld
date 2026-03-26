import logging
from fastapi import FastAPI, HTTPException
from urllib.error import URLError
from src.schema import ClaimRequest, RiskSummaryResponse
from src.retrieval import FraudPatternRetriever
from src.generation import RiskAnalyzerLLM
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Fraud & Risk Analysis Engine",
    description="Sanitized reference architecture for AI-assisted fraud detection using RAG.",
    version="1.0.0"
)

# Initialize dependencies on startup
retriever = FraudPatternRetriever()
analyzer = RiskAnalyzerLLM()

@app.post("/api/v1/analyze-claim", response_model=RiskSummaryResponse)
async def analyze_claim_endpoint(request: ClaimRequest):
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
        logger.error(f"Error processing claim {request.claim_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Internal Server Error during risk analysis. LLM or formatting failure."
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
