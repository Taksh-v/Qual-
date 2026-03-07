import requests
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OllamaLLM:
    """
    Integration with Ollama Phi3:mini model
    Mirrors the pattern used in market_engine.py
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "phi3:mini"):
        """
        Initialize Ollama connection
        
        Args:
            ollama_host: URL of Ollama API server
            model: Model name (default: phi3:mini)
        """
        self.host = ollama_host
        self.model = model
        self.api_endpoint = f"{ollama_host}/api/generate"
    
    def generate_response(self, prompt: str, context: Optional[str] = None, 
                         temperature: float = 0.7, max_tokens: int = 512) -> str:
        """
        Generate response using Ollama Phi3:mini
        
        Args:
            prompt: User question/query
            context: Additional context (from retrieved news embeddings)
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum response length
            
        Returns:
            Generated response from the model
        """
        # Build the full prompt with context
        if context:
            full_prompt = f"""Context from news sources:
{context}

Question: {prompt}

Answer based on the provided context:"""
        else:
            full_prompt = prompt
        
        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": temperature,
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response generated")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_streaming(self, prompt: str, context: Optional[str] = None,
                               temperature: float = 0.7):
        """
        Generate response with streaming for real-time output
        
        Args:
            prompt: User question
            context: Retrieved context
            temperature: Model temperature
            
        Yields:
            Response chunks as they arrive
        """
        if context:
            full_prompt = f"""Context from news sources:
{context}

Question: {prompt}

Answer:"""
        else:
            full_prompt = prompt
        
        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "temperature": temperature,
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"