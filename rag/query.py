import os
import json
import faiss
import numpy as np
import subprocess
from intelligence.query_rewriter import rewrite_query


# ================= PATH CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "index", "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "data", "vector_db", "metadata.json")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3:mini"
TOP_K = 5
# ==============================================


def embed_query(text: str) -> np.ndarray:
    """
    Generate embedding for the user query using Ollama
    """
    result = subprocess.run(
        ["ollama", "run", EMBED_MODEL],
        input=text,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError("Embedding generation failed")

    embedding = json.loads(result.stdout)
    return np.array(embedding, dtype="float32").reshape(1, -1)


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_chunks(query_embedding, index, metadata):
    distances, indices = index.search(query_embedding, TOP_K)

    chunks = []
    for idx in indices[0]:
        if idx < len(metadata):
            chunks.append(metadata[idx])

    return chunks


def build_prompt(chunks, question):
    context = "\n\n".join(
        f"- {chunk['text']}" for chunk in chunks
    )

    return f"""
You are a financial analysis assistant.
Answer ONLY using the context below.
If the answer is not present, say:
"Insufficient data from available news."

Context:
{context}

Question:
{question}

Answer:
""".strip()


def ask_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL],
        input=prompt,
        text=True
    )
    return result.stdout


def main():
    # -------- Sanity checks --------
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"faiss.index not found at {FAISS_INDEX_PATH}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"metadata.json not found at {METADATA_PATH}")

    # -------- Load index & metadata --------
    index = faiss.read_index(FAISS_INDEX_PATH)
    metadata = load_metadata()

    print("\n📊 Finance RAG System Ready")
    print("Type 'exit' to quit\n")

    while True:
        question = input("🔍 Ask a question: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        optimized_question = rewrite_query(question)
        query_embedding = embed_query(optimized_question)
        chunks = retrieve_chunks(query_embedding, index, metadata)
        prompt = build_prompt(chunks, question)
        answer = ask_llm(prompt)

        print("\n🧠 Answer:\n")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
