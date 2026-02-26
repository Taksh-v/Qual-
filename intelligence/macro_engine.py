import subprocess
import json
from typing import Iterator

from intelligence.macro_reasoner import macro_reason
from intelligence.sector_mapper import sector_impact
from intelligence.market_outlook import market_outlook
from intelligence.context_retriever import retrieve_relevant_context, format_context


LLM_MODEL = "phi3:mini"


def call_llm(prompt: str) -> Iterator[str]:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
    }
    process = subprocess.Popen(
        [
            "curl",
            "-s",
            "http://localhost:11434/api/generate",
            "-d",
            json.dumps(payload),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.stdout:
        for line in process.stdout:
            try:
                data = json.loads(line)
                yield data.get("response", "")
            except json.JSONDecodeError:
                continue

    process.wait()
    if process.returncode != 0:
        stderr_output = f"curl exited with code {process.returncode}."
        if process.stderr:
            stderr_read = process.stderr.read()
            if stderr_read:
                stderr_output += f" Stderr: {stderr_read}"
        yield (
            "\n\n--- LLM CALL FAILED ---\n"
            f"Error: {stderr_output}\n"
            "Is the Ollama server running? You can start it with 'ollama serve'\n"
            "-----------------------\n"
        )


def macro_intelligence(question: str) -> Iterator[str]:
    # 0. Retrieve latest custom context from your indexed data.
    try:
        context_chunks = retrieve_relevant_context(question)
        custom_context = format_context(context_chunks)
    except Exception as e:
        context_chunks = []
        custom_context = f"Context retrieval unavailable: {e}"

    # 1. Macro reasoning (economist brain) + custom/latest context grounding.
    base_macro_prompt = macro_reason(question)
    macro_prompt = f"""
Use the CUSTOM CONTEXT below as your primary factual grounding.
If context and prior knowledge conflict, prioritize context.
If data is missing in context, explicitly say so.

CUSTOM CONTEXT (latest + relevant):
{custom_context}

USER QUESTION:
{question}

TASK:
{base_macro_prompt}
""".strip()

    yield "📌 MACRO INTELLIGENCE REPORT\n\n"
    if context_chunks:
        yield f"🗂️ Retrieved {len(context_chunks)} custom context chunks.\n\n"
    else:
        yield "🗂️ No custom context chunks found. Falling back to model reasoning.\n\n"

    yield "🧠 Macro Analysis:\n"

    macro_analysis = ""
    for token in call_llm(macro_prompt):
        macro_analysis += token
        yield token

    # 2. Sector impact (investment logic)
    sector_prompt = sector_impact(macro_analysis)

    yield "\n\n🏭 Sector Impact:\n"

    sector_analysis = ""
    for token in call_llm(sector_prompt):
        sector_analysis += token
        yield token

    # 3. Market outlook (strategist view)
    outlook_prompt = market_outlook(macro_analysis, sector_analysis)

    yield "\n\n📈 Market Outlook:\n"

    for token in call_llm(outlook_prompt):
        yield token


if __name__ == "__main__":
    print("\n📊 Macro Intelligence Engine Ready\n")
    while True:
        q = input("🔍 Ask a macro question (or 'exit'): ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        print("\n")
        for chunk in macro_intelligence(q):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 70)
