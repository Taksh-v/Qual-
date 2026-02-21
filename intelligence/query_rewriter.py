import subprocess

MODEL = "phi3:mini"

def rewrite_query(question: str) -> str:
    """
    Rewrite the user query to improve retrieval quality.
    If rewriting fails, return original question.
    """
    prompt = f"""
        Rewrite the following financial question to be more specific,
        explicit, and suitable for retrieving relevant news and reports.
        
        Only return the rewritten query.
        Do not explain.
        
        Question:
        {question}
        """.strip()

    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            return question

        rewritten = result.stdout.strip()

        # Safety fallback
        if len(rewritten) < 5:
            return question

        return rewritten

    except Exception:
        return question
