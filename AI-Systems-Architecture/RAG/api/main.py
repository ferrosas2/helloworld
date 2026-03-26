"""
FastAPI application serving the RAG pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
import logging

from src.config import settings
from src.generation import RAGGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Logistics & Fintech RAG API",
    description="API for querying unstructured Bills of Lading and Freight Claims utilizing RAG.",
    version="1.0.0"
)

# Global variables to hold singletons
vector_store = None
rag_generator = None

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question regarding the documents.")
    top_k: int = Field(default=3, description="Number of context chunks to retrieve.")

class QueryResponse(BaseModel):
    answer: str
    
@app.on_event("startup")
async def startup_event():
    """
    Initializes ML models and vector store on application startup.
    """
    global vector_store, rag_generator
    try:
        embeddings = VertexAIEmbeddings(
            project=settings.project_id,
            location=settings.region,
            model_name=settings.embedding_model
        )
        vector_store = FAISS.load_local(
            settings.vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True  # Required for trusted local models
        )
        rag_generator = RAGGenerator()
        logger.info("Successfully initialized ML components.")
    except Exception as e:
        logger.warning(f"ML components failed to initialize. Ensure index exists and GCP credentials are valid. Error: {e}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes/Cloud Run readiness probes.
    """
    return {"status": "healthy"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest):
    """
    Endpoint to process a user question, retrieve context, and generate an answer.
    """
    if vector_store is None or rag_generator is None:
        raise HTTPException(status_code=503, detail="Service unavailable: ML components not initialized. Check server logs.")
        
    try:
        # Retrieve top-k context chunks based on semantic similarity
        retrieved_docs = vector_store.similarity_search(payload.question, k=payload.top_k)
        
        # Generate the answer using the strict RAG pipeline
        answer = rag_generator.generate_answer(payload.question, retrieved_docs)
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while generating the response."
        )