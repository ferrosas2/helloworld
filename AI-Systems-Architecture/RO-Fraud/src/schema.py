from pydantic import BaseModel, Field
from typing import List

class ClaimRequest(BaseModel):
    claim_id: str = Field(..., description="Unique identifier for the claim")
    customer_id: str = Field(..., description="Unique identifier for the customer")
    claim_text: str = Field(..., description="The textual description of the claim")
    claim_amount: float = Field(..., description="The amount claimed")

class RiskSummaryResponse(BaseModel):
    fraud_probability_score: float = Field(..., description="Score between 0.0 and 1.0 indicating fraud probability")
    risk_factors: List[str] = Field(..., description="List of identified risk factors")
    executive_summary: str = Field(..., description="A concise explanation of the risk assessment")
