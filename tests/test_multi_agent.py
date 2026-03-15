import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from rag.query import run_query

async def main():
    q = "What is Apple's forward guidance and how does it compare to their current P/E ratio?"
    print(f"\nQuestion: {q}")
    print("Executing Multi-Agent Synthesized Query...\n")
    
    answer, _ = await run_query(q)
    print("\n--- PORTFOLIO MANAGER SYNTHESIS ---\n")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
