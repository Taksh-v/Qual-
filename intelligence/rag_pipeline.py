import os
from typing import List, Dict, Optional
from intelligence.embedding_engine import EmbeddingEngine
from intelligence.ollama_integration import OllamaLLM
import logging

logger = logging.getLogger(__name__)

class NewsRAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for news data
    Combines embeddings, retrieval, and LLM generation
    """
    
    def __init__(self, 
                 embedding_model: str | None = None,
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "phi3:mini"):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_model: Sentence transformer model
            ollama_host: Ollama API endpoint
            ollama_model: Ollama model name
        """
        if embedding_model is None:
            embedding_model = (
                os.getenv("EMBED_MODEL_FINANCE")
                or os.getenv("EMBED_MODEL")
                or "all-MiniLM-L6-v2"
            )
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.llm = OllamaLLM(ollama_host, ollama_model)
        self.news_store: List[Dict] = []
    
    def ingest_news(self, news_items: List[Dict]) -> None:
        """
        Ingest and embed news items
        
        Args:
            news_items: Raw news from ingestion pipeline
        """
        logger.info(f"Ingesting {len(news_items)} news items...")
        
        # Embed all news items
        embedded_news = self.embedding_engine.embed_news_batch(news_items)
        
        # Store embeddings
        self.news_store.extend(embedded_news)
        
        logger.info(f"Total news items in store: {len(self.news_store)}")
    
    def query(self, question: str, top_k: int = 5, use_streaming: bool = False) -> str:
        """
        Execute RAG query: retrieve relevant news → generate answer with LLM
        
        Args:
            question: User question
            top_k: Number of news items to retrieve
            use_streaming: Whether to stream response
            
        Returns:
            Generated answer
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant news
        relevant_news = self.embedding_engine.similarity_search(
            question, 
            self.news_store, 
            top_k=top_k
        )
        
        if not relevant_news:
            logger.warning("No relevant news found")
            return "No relevant information found in news database."
        
        # Step 2: Build context from retrieved news
        context = self._build_context(relevant_news)
        
        logger.info(f"Retrieved {len(relevant_news)} relevant news items")
        
        # Step 3: Generate response with LLM
        if use_streaming:
            return self._generate_streamed_response(question, context)
        else:
            return self.llm.generate_response(question, context)
    
    def _build_context(self, news_items: List[Dict]) -> str:
        """
        Build context string from retrieved news items
        
        Args:
            news_items: Retrieved news items
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, item in enumerate(news_items, 1):
            context_parts.append(f"""
[Source {i}] {item.get('source', 'Unknown')}
Title: {item.get('title', 'N/A')}
Content: {item.get('content', 'N/A')}
""")
        
        return "\n".join(context_parts)
    
    def _generate_streamed_response(self, question: str, context: str) -> str:
        """
        Generate response with streaming
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Full response
        """
        full_response = ""
        
        for chunk in self.llm.generate_with_streaming(question, context):
            full_response += chunk
            print(chunk, end="", flush=True)
        
        print()  # Newline after streaming
        return full_response