from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag_core import ask_rag

app = FastAPI(title="News Intelligence RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    return ask_rag(req.question)

# python -m uvicorn api.app:app --reload
