import subprocess
import json
import time

def test_rag_query():
    print("Testing API integration with SEC 8-K data...")
    # First, let's see what companies we actually have in the DB
    try:
        with open("data/vector_db/metadata.json", "r") as f:
            metadata = json.load(f)
            sec_docs = [m for m in metadata if 'SEC EDGAR' in m.get('metadata', {}).get('source', '')]
            print(f"Total SEC chunks in DB: {len(sec_docs)}")
            
            if len(sec_docs) > 0:
                companies = set([m['metadata'].get('company') for m in sec_docs if m.get('metadata', {}).get('company')])
                filtered = [c for c in companies if c and c.lower() != 'unknown' and c.strip()]
                print(f"Companies found: {list(filtered)[:5]}...")
                
                # Pick the first company to query about
                target_company = "a recent company"
                if len(filtered) > 0:
                    target_company = list(filtered)[0]
                
                query = f"Summarize the recent SEC 8-K filing for {target_company}."
                print(f"\nQuerying RAG system: '{query}'\n")
                
                # We'll use the CLI query script if it exists, otherwise curl the API
                # The prompt mentions `rag/rag_core.py` has `ask_rag()`
                
                script = f"""
import sys
import asyncio
from rag.rag_core import ask_rag

async def main():
    result = await ask_rag("{query}")
    print("ANSWER:")
    print(result["answer"])
    print("\\nSOURCES:")
    for s in result["sources"]:
        print(f"- {{s['metadata'].get('title', 'Unknown')}}")

if __name__ == "__main__":
    asyncio.run(main())
"""
                with open("temp_query.py", "w") as tf:
                    tf.write(script)
                
                res = subprocess.run(["python", "temp_query.py"], capture_output=True, text=True)
                print(res.stdout)
                
    except Exception as e:
        print(f"Error querying DB: {e}")

if __name__ == "__main__":
    test_rag_query()
