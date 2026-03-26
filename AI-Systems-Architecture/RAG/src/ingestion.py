"""
Document ingestion module for parsing, splitting, and embedding logistics/fintech documents.
"""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document

from src.config import settings

class DocumentIngestor:
    """
    Handles the ingestion pipeline: splitting text documents into chunks,
    generating embeddings via Vertex AI, and storing them in a local FAISS index.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """
        Initializes the ingestor with specific chunking parameters.
        
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = VertexAIEmbeddings(
            project=settings.project_id,
            location=settings.region,
            model_name=settings.embedding_model
        )
        self.vector_store_path = settings.vector_store_path

    def process_documents(self, documents: List[str]) -> None:
        """
        Processes a list of raw text documents into a persistent FAISS vector store.
        
        Args:
            documents (List[str]): A list of raw text documents (e.g., extracted from PDFs).
        """
        # Convert raw text into LangChain Document objects
        langchain_docs = [Document(page_content=doc) for doc in documents]
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(langchain_docs)
        
        # Create FAISS vector store from chunks
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        
        # Save vector store locally
        vector_store.save_local(self.vector_store_path)
        print(f"Successfully ingested {len(documents)} documents into {len(chunks)} chunks.")