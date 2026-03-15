import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.rag_core import ask_rag

async def main():
    try:
        with open("data/vector_db/metadata.json", "r") as f:
            metadata = json.load(f)
            sec_docs = [m for m in metadata if 'SEC EDGAR' in m.get('metadata', {}).get('source', '')]
            
            companies = set()
            for m in sec_docs:
                title = m.get('metadata', {}).get('title', '')
                if title:
                    # SEC titles look like: 8-K - Company Name Inc (000000)
                    parts = title.split(' - ')
                    if len(parts) > 1:
                        company_part = parts[1].split(' (')[0].split(' 00')[0].strip()
                        companies.add(company_part)
                        
            companies = list(companies)
            print(f"Parsed companies: {companies[:5]}")
            if companies:
                target = companies[0]
                query = f"What is the latest 8-K filing about for {target}?"
                print(f"Targeting company: {target}")
                print(f"Query: {query}")
                print("-------------")
                
                result = await ask_rag(query)
                print(result["answer"])
            else:
                print("No companies found in DB to test against.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
