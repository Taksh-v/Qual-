
import sys
import asyncio
from rag.rag_core import ask_rag

async def main():
    result = await ask_rag("Summarize the recent SEC 8-K filing for K.")
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"- {s['metadata'].get('title', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(main())
