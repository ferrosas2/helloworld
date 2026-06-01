import logging
from typing import List
from langchain_google_vertexai import VectorSearchVectorStore, VertexAIEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)

class FraudPatternRetriever:
    """
    Retriever class that connects directly to Google Vertex AI Vector Search (formerly Matching Engine).
    Retrieves the top similar past fraudulent claims using real vector embeddings.
    """
    def __init__(self):
        logger.info("Initializing FraudPatternRetriever with Production GCP Vertex AI Vector Search")
        
        # Initialize Vertex AI Embeddings (text-embedding-004)
        try:
            self.embeddings = VertexAIEmbeddings(
                model_name="text-embedding-004",
                project=settings.GCP_PROJECT_ID,
                location=settings.GCP_REGION
            )
            logger.info("Successfully established connection to Vertex AI Embeddings (text-embedding-004).")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI Embeddings: {str(e)}")
            raise
        
        # Initialize the production Vector Search client.
        # langchain-google-vertexai exposes the store via `from_components`, which
        # expects `project_id` and `region` (not `project`/`location`).
        try:
            self.vector_store = VectorSearchVectorStore.from_components(
                project_id=settings.GCP_PROJECT_ID,
                region=settings.GCP_REGION,
                gcs_bucket_name=settings.GCS_BUCKET_NAME,
                index_id=settings.VERTEX_INDEX_ID,
                endpoint_id=settings.VERTEX_ENDPOINT_ID,
                embedding=self.embeddings,
            )
            logger.info("Successfully established connection to Vertex AI VectorSearchVectorStore.")
        except Exception as init_err:
            logger.error(f"Failed to initialize VectorSearchVectorStore: {str(init_err)}")
            raise

    def get_similar_historical_claims(self, claim_text: str) -> List[str]:
        """
        Query real Vertex AI Vector Search and return the matching documents' contents.
        """
        logger.info(f"Querying real Vertex AI Vector Search for input text check: {claim_text[:60]}...")
        try:
            # Perform similarity search with top 3 neighbors
            results = self.vector_store.similarity_search(claim_text, k=3)
            
            # Extract page_content from retrieved langchain Document objects
            retrieved_claims = [doc.page_content for doc in results]
            
            logger.info(f"Successfully retrieved {len(retrieved_claims)} matching claims from Vector Search.")
            
            # Return safe default message if empty to avoid stopping the server/app
            if not retrieved_claims:
                logger.warning("Vertex Vector Search returned no results. Returning operational fallback notice.")
                return ["Info: No matching historical fraudulent claims found in the database for the given text."]
                
            return retrieved_claims
            
        except Exception as e:
            logger.error(f"Failed to query Vertex AI Vector Search: {str(e)}")
            # Raise exception to bubble up to FastAPI error handler (Tier-1 reliability)
            raise
