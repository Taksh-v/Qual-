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
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        
        Args:
            model_name: Sentence transformer model to use
        """
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
        embedded_items = []
        
        for item in news_items:
            # Combine title and content for embedding
            text_to_embed = f"{item.get('title', '')} {item.get('content', '')}"
            
            # Generate embedding
            embedding = self.model.encode(text_to_embed)
            
            # Add embedding to item
            item['embedding'] = embedding.tolist()
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
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for item in news_embeddings:
            score = np.dot(query_embedding, np.array(item['embedding']))
            similarities.append((item, score))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in similarities[:top_k]]