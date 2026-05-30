import logging
from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain.output_parsers import PydanticOutputParser
from src.schema import RiskSummaryResponse
from src.config import settings

logger = logging.getLogger(__name__)

class RiskAnalyzerLLM:
    """
    LLM generation class handling communication with Google Vertex AI (Gemini).
    Implements a strict Pydantic output parser for reliable JSON schema enforcement.
    """
    def __init__(self):
        logger.info(f"Initializing RiskAnalyzerLLM with Google Vertex AI (gemini-1.5-pro) in project '{settings.GCP_PROJECT_ID}' and region '{settings.GCP_REGION}'")
        # Initialize Gemini via Vertex AI targeting user's live GCP resources with ADC.
        self.llm = VertexAI(
            model_name="gemini-1.5-pro",
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_REGION,
            temperature=0.0
        )
        self.parser = PydanticOutputParser(pydantic_object=RiskSummaryResponse)
        
        self.prompt = PromptTemplate(
            template=(
                "You are an expert fraud investigator. Compare the current claim against the historical context.\n"
                "Output ONLY a valid JSON matching the RiskSummaryResponse schema. Do not hallucinate.\n\n"
                "Historical Context (Past Fraudulent Claims):\n{historical_fraud_context}\n\n"
                "Current Claim Details:\n{current_claim}\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["current_claim", "historical_fraud_context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.parser

    def analyze_claim(self, claim: str, context: List[str]) -> Dict[str, Any]:
        """
        Analyzes a claim using context and prompts the LLM to return a structured JSON response.
        """
        try:
            logger.info("Analyzing claim using LLM chain.")
            context_str = "\n".join(context)
            response: RiskSummaryResponse = self.chain.invoke({
                "current_claim": claim,
                "historical_fraud_context": context_str
            })
            return response.dict()
        except Exception as e:
            logger.error(f"Failed to generate or parse LLM response: {str(e)}")
            raise
