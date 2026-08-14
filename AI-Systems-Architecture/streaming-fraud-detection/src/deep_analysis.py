"""
RAG-Based Deep Fraud Analysis (Explainable AI Path).

When the fast-path Vertex AI model flags a transaction as high-risk (score >= 0.7),
this module performs a deep analysis using Retrieval-Augmented Generation:

1. Embeds the flagged transaction context using Vertex AI Embeddings
2. Retrieves top-K similar historical fraud cases from Vertex AI Vector Search
3. Sends the transaction + historical context to Gemini 2.5 Flash
4. Returns a structured, explainable risk assessment with cited evidence

This is the "slow path" (~1-2 seconds) — it runs asynchronously after the
fast path has already made the approve/block decision. Its output enriches
the case file for human fraud analysts.

Architecture:
    Flagged Transaction → Embed → Vector Search (top-3) → Gemini → Structured Response
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_vertexai import (
    VertexAI,
    VertexAIEmbeddings,
    VectorSearchVectorStore,
)

from src.config import settings
from src.schema import DeepAnalysisResponse, RiskFactor

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Template for Fraud Analysis
# =============================================================================

FRAUD_ANALYSIS_PROMPT = PromptTemplate(
    template=(
        "You are an expert fraud investigator analyzing a flagged POS transaction.\n"
        "Your task is to assess the fraud risk based on the transaction details and\n"
        "similar historical fraud cases retrieved from the database.\n\n"
        "IMPORTANT RULES:\n"
        "- Base your analysis ONLY on the provided evidence\n"
        "- Do NOT hallucinate or invent fraud indicators not present in the data\n"
        "- If insufficient evidence exists, state that explicitly\n"
        "- Provide actionable recommendations for the fraud review team\n\n"
        "---\n\n"
        "FLAGGED TRANSACTION:\n{transaction_details}\n\n"
        "INITIAL FRAUD SCORE: {fraud_score}\n\n"
        "---\n\n"
        "SIMILAR HISTORICAL FRAUD CASES (from vector search):\n{historical_context}\n\n"
        "---\n\n"
        "Analyze this transaction and provide your assessment.\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["transaction_details", "fraud_score", "historical_context"],
    partial_variables={},
)


# =============================================================================
# Deep Analysis Engine
# =============================================================================

class DeepFraudAnalyzer:
    """
    RAG-based fraud analysis engine using Vertex AI Vector Search + Gemini.
    
    This mirrors the architecture in ../RO-Fraud/ but is adapted for the
    streaming context where transactions arrive via Pub/Sub after being
    flagged by the fast-path model.
    """

    def __init__(self):
        """Initialize connections to Vertex AI services."""
        logger.info("Initializing DeepFraudAnalyzer...")
        logger.info(f"  Project: {settings.GCP_PROJECT_ID}")
        logger.info(f"  Region: {settings.GCP_REGION}")
        logger.info(f"  LLM Model: {settings.GEMINI_MODEL}")

        # Embedding model for encoding transaction text
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_REGION,
        )

        # Vector Search store for retrieving similar historical fraud
        self.vector_store = VectorSearchVectorStore.from_components(
            project_id=settings.GCP_PROJECT_ID,
            region=settings.GCP_REGION,
            gcs_bucket_name=settings.GCS_BUCKET_NAME,
            index_id=settings.VERTEX_INDEX_ID,
            endpoint_id=settings.VERTEX_ENDPOINT_ID,
            embedding=self.embeddings,
        )

        # Gemini LLM for generating explainable risk analysis
        self.llm = VertexAI(
            model_name=settings.GEMINI_MODEL,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_REGION,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        )

        # Output parser for structured JSON response
        self.parser = PydanticOutputParser(pydantic_object=DeepAnalysisResponse)

        # Build the LangChain chain
        self.prompt = FRAUD_ANALYSIS_PROMPT.partial(
            format_instructions=self.parser.get_format_instructions()
        )
        self.chain = self.prompt | self.llm | self.parser

        logger.info("DeepFraudAnalyzer initialized successfully.")

    def analyze_transaction(
        self,
        transaction_id: str,
        transaction_text: str,
        fraud_score: float,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform deep RAG-based analysis on a flagged transaction.
        
        Args:
            transaction_id: Unique transaction identifier
            transaction_text: Formatted transaction details for context retrieval
            fraud_score: Initial fraud score from the fast-path model
            top_k: Number of similar historical cases to retrieve
            
        Returns:
            Dict containing the full DeepAnalysisResponse fields
        """
        start_time = time.time()
        logger.info(f"Starting deep analysis for transaction: {transaction_id}")

        try:
            # Step 1: Retrieve similar historical fraud cases
            retrieval_start = time.time()
            similar_cases = self._retrieve_similar_fraud(transaction_text, top_k=top_k)
            retrieval_latency = (time.time() - retrieval_start) * 1000
            logger.info(
                f"Retrieved {len(similar_cases)} similar cases "
                f"in {retrieval_latency:.1f}ms"
            )

            # Step 2: Format context for the LLM
            historical_context = self._format_historical_context(similar_cases)

            # Step 3: Generate explainable risk analysis via Gemini
            generation_start = time.time()
            response: DeepAnalysisResponse = self.chain.invoke({
                "transaction_details": transaction_text,
                "fraud_score": f"{fraud_score:.2f}",
                "historical_context": historical_context,
            })
            generation_latency = (time.time() - generation_start) * 1000

            total_latency = (time.time() - start_time) * 1000
            logger.info(
                f"Deep analysis complete: "
                f"retrieval={retrieval_latency:.1f}ms, "
                f"generation={generation_latency:.1f}ms, "
                f"total={total_latency:.1f}ms"
            )

            # Merge timing metadata
            result = response.dict()
            result["transaction_id"] = transaction_id
            result["analysis_latency_ms"] = round(total_latency, 2)

            return result

        except Exception as e:
            total_latency = (time.time() - start_time) * 1000
            logger.error(f"Deep analysis failed for {transaction_id}: {str(e)}")

            # Return a safe fallback response
            return {
                "transaction_id": transaction_id,
                "fraud_probability_score": fraud_score,
                "risk_factors": [
                    {
                        "factor": "Analysis unavailable",
                        "severity": "medium",
                        "evidence": f"Deep analysis failed: {str(e)}"
                    }
                ],
                "executive_summary": (
                    f"Automated deep analysis could not be completed. "
                    f"Initial model score: {fraud_score:.2f}. "
                    f"Manual review recommended."
                ),
                "similar_historical_cases": [],
                "recommended_action": "HOLD_FOR_REVIEW",
                "analysis_latency_ms": round(total_latency, 2),
                "confidence": 0.3,
            }

    def _retrieve_similar_fraud(self, query_text: str, top_k: int = 3) -> List[str]:
        """
        Query Vertex AI Vector Search for similar historical fraud patterns.
        
        Returns the page_content of the top-K most similar documents.
        """
        try:
            results = self.vector_store.similarity_search(query_text, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            return ["No historical fraud patterns available for comparison."]

    @staticmethod
    def _format_historical_context(cases: List[str]) -> str:
        """Format retrieved cases into a numbered list for the prompt."""
        if not cases:
            return "No similar historical fraud cases found in the database."

        formatted = []
        for i, case in enumerate(cases, 1):
            formatted.append(f"Case {i}:\n{case}")
        return "\n\n".join(formatted)
