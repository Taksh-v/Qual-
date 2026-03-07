from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag_pipeline import NewsRAGPipeline
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="News Intelligence API")

# Initialize RAG pipeline
rag_pipeline = NewsRAGPipeline()

class NewsIngestRequest(BaseModel):
    news_items: list[dict]

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_streaming: Optional[bool] = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources_used: int

@app.post("/ingest")
async def ingest_news(request: NewsIngestRequest):
    """
    Ingest news items and create embeddings
    """
    try:
        rag_pipeline.ingest_news(request.news_items)
        return {
            "status": "success",
            "total_items": len(rag_pipeline.news_store)
        }
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_news(request: QueryRequest):
    """
    Query the news intelligence system
    """
    try:
        answer = rag_pipeline.query(
            request.question,
            top_k=request.top_k,
            use_streaming=request.use_streaming
        )
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources_used=min(request.top_k, len(rag_pipeline.news_store))
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get pipeline statistics
    """
    return {
        "total_news_items": len(rag_pipeline.news_store),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "phi3:mini"
    }