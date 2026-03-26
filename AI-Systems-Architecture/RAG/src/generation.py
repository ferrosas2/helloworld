"""
Generation module handling the Retrieval-Augmented Generation (RAG) logic.
"""

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_core.documents import Document

from src.config import settings

class RAGGenerator:
    """
    Generates responses based on retrieved context using a strict PromptTemplate
    to prevent hallucinations and ensure compliance.
    """
    
    def __init__(self):
        """
        Initializes the RAG Generator, setting up the LLM and the strict prompt template.
        """
        self.llm = VertexAI(
            project=settings.project_id,
            location=settings.region,
            model_name=settings.llm_model_name,
            temperature=0.0  # Zero temperature for deterministic, factual responses
        )
        
        # Strict, robust PromptTemplate for financial/logistics compliance
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert logistics and fintech AI assistant.\n"
                "Your task is to answer the user's question based strictly on the provided context.\n\n"
                "Constraints:\n"
                "1. Answer ONLY using the facts from the provided context.\n"
                "2. If the answer cannot be determined from the context, you must reply exactly with: 'Insufficient data to answer.'\n"
                "3. Do not formulate assumptions, guess, or hallucinate information outside the context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )
        
        self.chain = self.prompt_template | self.llm

    def generate_answer(self, question: str, retrieved_docs: List[Document]) -> str:
        """
        Generates an answer for the given question using the retrieved documents.
        
        Args:
            question (str): The user's query.
            retrieved_docs (List[Document]): The context documents retrieved from the vector store.
            
        Returns:
            str: The LLM-generated answer.
        """
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        response = self.chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return response.strip()