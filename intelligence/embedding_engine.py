import os
import json
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    """
    Handles embedding of news data for semantic search
    Similar to market_engine but focused on news embeddings
    """
    
    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding model
        
        Args:
            model_name: Sentence transformer model to use
        """
        if model_name is None:
            model_name = (
                os.getenv("EMBED_MODEL_FINANCE")
                or os.getenv("EMBED_MODEL")
                or "all-MiniLM-L6-v2"
            )
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
    
    def embed_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """
        Embed a batch of news items
        
        Args:
            news_items: List of dicts with 'title', 'content', 'source' keys
            
        Returns:
            List of dicts with added 'embedding' field
        """
        if not news_items:
            return []

        texts = [
            f"{item.get('title', '')} {item.get('content', '')}".strip()
            for item in news_items
        ]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        embedded_items = []
        for item, embedding in zip(news_items, embeddings):
            item["embedding"] = np.asarray(embedding, dtype="float32").tolist()
            embedded_items.append(item)

        return embedded_items
    
    def similarity_search(self, query: str, news_embeddings: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Find most relevant news items for a query using cosine similarity
        
        Args:
            query: User query or question
            news_embeddings: List of embedded news items
            top_k: Number of top results to return
            
        Returns:
            Top K most relevant news items
        """
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Calculate similarities
        similarities = []
        for item in news_embeddings:
            score = float(np.dot(query_embedding, np.array(item["embedding"], dtype="float32")))
            similarities.append((item, score))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in similarities[:top_k]]