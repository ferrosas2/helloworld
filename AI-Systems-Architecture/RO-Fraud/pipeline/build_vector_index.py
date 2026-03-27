import logging
import re
import os
import pandas as pd
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings

# Configure professional standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    """Mock loading raw unstructured historical claims data."""
    logger.info("Loading raw claims data into DataFrame.")
    data = [
        {"claim_id": "C-1001", "raw_text": "Ph: 555-123-4567. SSN: 123-45-6789. Claim for stolen laptop from car.", "fraud_confirmed": True, "resolution_notes": "Multiple inconsistencies in police report. Suspected serial fraudster."},
        {"claim_id": "C-1002", "raw_text": "Customer reported minor fender bender. No PII.", "fraud_confirmed": False, "resolution_notes": "Standard processing, approved. Valid repair estimates."},
        {"claim_id": "C-1003", "raw_text": "SSN 987-65-4321. Claimed $50k for water damage but weather was sunny.", "fraud_confirmed": True, "resolution_notes": "Weather data contradicts claim completely. Fabricated event."},
        {"claim_id": "C-1004", "raw_text": "Lost ring at beach. Ph: 555-999-0000.", "fraud_confirmed": False, "resolution_notes": "Approved after receipt and photo verification provided."},
        {"claim_id": "C-1005", "raw_text": "Stolen art piece, no police report filed. Contact: 555-111-2222.", "fraud_confirmed": True, "resolution_notes": "Repeat offender, identical claim filed 2 years ago across state lines."}
    ]
    return pd.DataFrame(data)

def clean_text_with_regex(text: str) -> str:
    """Removes PII (SSNs, phone numbers) and standardizes formatting for privacy."""
    # Redact SSN-like patterns (XXX-XX-XXXX)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)
    # Redact Phone-like patterns (XXX-XXX-XXXX)
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[REDACTED_PHONE]', text)
    # Strip unnecessary whitespaces and special characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_document_chunks(df: pd.DataFrame) -> List[Document]:
    """Filters fraudulent claims, sanitizes text, and chunks into LangChain Documents."""
    logger.info("Filtering and sanitizing data to build 'Golden Dataset' chunks...")
    fraud_df = df[df["fraud_confirmed"] == True]
    
    documents = []
    # Setting an optimal chunk size for context retention without exceeding token limits
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    for _, row in fraud_df.iterrows():
        cleaned_text = clean_text_with_regex(row["raw_text"])
        
        # Combine contextual text for maximum embedding quality
        combined_text = f"Claim Context: {cleaned_text}\nResolution Notes: {row['resolution_notes']}"
        
        chunks = splitter.split_text(combined_text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"claim_id": row["claim_id"]}
            )
            documents.append(doc)
            
    logger.info(f"Successfully created {len(documents)} document chunks from {len(fraud_df)} fraudulent claims.")
    return documents

def build_and_save_index(documents: List[Document], output_path: str):
    """Generates embeddings and builds a FAISS index, saving it to disk."""
    logger.info("Initializing VertexAIEmbeddings model...")
    try:
        # In production this will leverage actual GCP credentials via ADC
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    except Exception as e:
        logger.warning(f"Could not init VertexAIEmbeddings (mocking locally for demo without GCP ADC): {e}")
        from langchain_community.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=768)
        
    logger.info("Building vector space using FAISS...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vectorstore.save_local(output_path)
    logger.info(f"Vector Index established and serialized to {output_path}")

if __name__ == "__main__":
    logger.info("=== START: Offline Data Ingestion Pipeline ===")
    
    # Step 1: Load Data
    raw_df = load_raw_data()
    
    # Step 2: Clean, chunk and prepare documents
    docs = create_document_chunks(raw_df)
    
    # Step 3: Embed and save the Golden Dataset to disk
    # Save relatively in ../data/fraud_vector_db inside RO-Fraud/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "data", "fraud_vector_db")
    
    build_and_save_index(docs, db_path)
    
    logger.info("=== COMPLETE: Offline Data Ingestion Pipeline ===")
