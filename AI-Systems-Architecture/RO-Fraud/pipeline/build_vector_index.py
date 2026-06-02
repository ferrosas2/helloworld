import json
import logging
import re
import os
import pandas as pd
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import storage

# Ensure we import the settings correctly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings

# Configure professional standard logging for GCP Operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================================================================================
# OPERATIONAL NOTE:
# The actual creation of a Vertex AI Vector Search Index (formerly Matching Engine)
# and its eventual deployment to an Endpoint in the Google Cloud Console
# typically takes approximately 45 minutes. 
# This ingestion script executes the prerequisite steps:
# 1. Loads, sanitizes, and chunks raw claims data.
# 2. Generates semantic embeddings with VertexAIEmbeddings (using text-embedding-004).
# 3. Formats the data into the strict JSONL format required by Vertex AI Vector Search.
# 4. Uploads these embeddings to a secure GCS bucket for indexing.
# ====================================================================================

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
    logger.info("Applying strict regex-based preprocessing rules to sanitize PII.")
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
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "claim_id": row["claim_id"],
                    "chunk_index": i
                }
            )
            documents.append(doc)
            
    logger.info(f"Successfully created {len(documents)} document chunks from {len(fraud_df)} fraudulent claims.")
    return documents

def generate_gcp_embeddings(documents: List[Document]) -> List[List[float]]:
    """Generates embeddings using VertexAIEmbeddings model 'text-embedding-004'."""
    logger.info("Initializing connection to Vertex AI Embeddings (model: text-embedding-004)...")
    embeddings_model = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_REGION
    )
    
    texts = [doc.page_content for doc in documents]
    logger.info(f"Generating vectors for {len(texts)} document chunks via Vertex AI Embeddings API.")
    embeddings_list = embeddings_model.embed_documents(texts)
    logger.info(f"Generated {len(embeddings_list)} embeddings successfully.")
    return embeddings_list

def save_to_strict_jsonl(documents: List[Document], embeddings: List[List[float]], output_path: str):
    """
    Format output into strict JSONL format required by Vertex AI Vector Search:
    {"id": "...", "embedding": [...], "restricts": [...]}
    """
    logger.info(f"Formatting embeddings and metadata into strict JSONL for Vertex Vector Search...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            claim_id = doc.metadata.get("claim_id", f"unknown_{i}")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            unique_id = f"{claim_id}_chunk_{chunk_idx}"
            
            item = {
                "id": unique_id,
                "embedding": embedding,
                "restricts": [
                    {
                        "namespace": "claim_id",
                        "allow": [claim_id]
                    }
                ]
            }
            f.write(json.dumps(item) + "\n")
            
    logger.info(f"Successfully serialized structured embeddings JSONL to: {output_path}")

def store_documents_for_retrieval(documents: List[Document], bucket_name: str, prefix: str = "documents"):
    """
    Writes each chunk's text to gs://{bucket}/{prefix}/{id} as plain text.

    langchain-google-vertexai's VectorSearchVectorStore uses a GCSDocumentStorage
    to resolve matched vector IDs back into their original text. It looks for each
    document at the blob `{prefix}/{id}`. The IDs MUST match exactly the IDs written
    into the embeddings JSONL (here: `{claim_id}_chunk_{chunk_index}`), otherwise a
    similarity search returns IDs that cannot be resolved ("not found in document
    storage").
    """
    logger.info(f"Storing {len(documents)} document texts for retrieval under '{prefix}/' ...")
    storage_client = storage.Client(project=settings.GCP_PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)

    for i, doc in enumerate(documents):
        claim_id = doc.metadata.get("claim_id", f"unknown_{i}")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        unique_id = f"{claim_id}_chunk_{chunk_idx}"

        blob = bucket.blob(f"{prefix}/{unique_id}")
        blob.upload_from_string(doc.page_content)

    logger.info(f"Stored document texts to gs://{bucket_name}/{prefix}/")

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str):
    """Uploads the computed JSONL file into the specified GCS bucket."""
    logger.info(f"Uploading local file {local_file_path} to Google Cloud Storage (Bucket: {bucket_name}, Path: {destination_blob_name})")
    try:
        storage_client = storage.Client(project=settings.GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(local_file_path)
        logger.info(f"File uploaded successfully to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to GCS: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("=== START: Offline Data Ingestion Pipeline (Production GCP Upgrade) ===")
    
    # Step 1: Load Data
    raw_df = load_raw_data()
    
    # Step 2: Clean, chunk and prepare documents
    docs = create_document_chunks(raw_df)
    
    # Step 3: Embed document chunks using real Vertex AI Text Embedding 004
    embeddings_list = generate_gcp_embeddings(docs)
    
    # Step 4: Format and serialize into the JSON-Lines format Vertex expects.
    # NOTE: Vertex AI Vector Search requires a `.json` extension (it parses the
    # content as JSON-Lines regardless), so the file is named accordingly.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_jsonl_path = os.path.join(base_dir, "..", "data", "vertex_vectors.json")
    save_to_strict_jsonl(docs, embeddings_list, local_jsonl_path)
    
    # Step 5: Upload the Golden Dataset embeddings file to Google Cloud Storage.
    # This file lives under `vector_search/`, which is the folder passed to the
    # index as `contentsDeltaUri`. Only vector-data files may live in that folder.
    gcs_blob_path = "vector_search/vertex_vectors.json"
    upload_to_gcs(
        local_file_path=local_jsonl_path,
        bucket_name=settings.GCS_BUCKET_NAME,
        destination_blob_name=gcs_blob_path
    )

    # Step 5b: Store each chunk's text under `documents/{id}` so the online
    # VectorSearchVectorStore can resolve matched IDs back into text at query time.
    store_documents_for_retrieval(docs, settings.GCS_BUCKET_NAME, prefix="documents")
    
    # Optional metadata map for local chunk-id -> text lookups. This is NOT vector
    # data, so it is uploaded OUTSIDE the `vector_search/` folder to avoid the index
    # ingestion trying (and failing) to parse it as embeddings.
    metadata_map_path = os.path.join(base_dir, "..", "data", "metadata_map.json")
    metadata_map = {
        f"{doc.metadata['claim_id']}_chunk_{doc.metadata['chunk_index']}": doc.page_content
        for doc in docs
    }
    with open(metadata_map_path, "w", encoding="utf-8") as mf:
        json.dump(metadata_map, mf, indent=2)
    upload_to_gcs(
        local_file_path=metadata_map_path,
        bucket_name=settings.GCS_BUCKET_NAME,
        destination_blob_name="metadata/metadata_map.json"
    )
    
    logger.info("=== END: Offline Data Ingestion Pipeline (GCP Upgrade Finished) ===")
    
    logger.info("=== COMPLETE: Offline Data Ingestion Pipeline ===")
