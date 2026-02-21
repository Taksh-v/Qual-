from rag.query import run_query

def ask_rag(question: str) -> dict:
    """
    Single entry point for RAG queries
    """
    answer, sources = run_query(question)

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
