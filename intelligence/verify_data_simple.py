"""
Quick verification that your custom data is in the system
Run this after ingesting your news data
"""

from rag_pipeline import NewsRAGPipeline
import json

def quick_verify():
    pipeline = NewsRAGPipeline()
    
    # Load your ingested data
    from run_news_ingestion import ingest_news
    news = ingest_news()
    
    print("\n📊 QUICK DATA VERIFICATION\n")
    
    # 1. Check data count
    print(f"1️⃣  Data Ingested: {len(news)} articles")
    
    # 2. Embed data
    print(f"2️⃣  Embedding data...")
    pipeline.ingest_news(news)
    print(f"    ✅ {len(pipeline.news_store)} articles embedded")
    
    # 3. Check specific article
    print(f"\n3️⃣  Sample articles in system:")
    for i, item in enumerate(pipeline.news_store[:3], 1):
        print(f"    {i}. {item['title'][:60]}...")
        print(f"       Source: {item['source']}")
    
    # 4. Test a search
    print(f"\n4️⃣  Testing search for 'trade'...")
    results = pipeline.embedding_engine.similarity_search("trade", pipeline.news_store, top_k=3)
    print(f"    Found {len(results)} relevant articles:")
    for r in results:
        print(f"    - {r['title'][:70]}...")
    
    # 5. Test LLM integration
    print(f"\n5️⃣  Testing LLM response generation...")
    response = pipeline.query("What's the main topic in these news articles?", top_k=5)
    print(f"    ✅ LLM generated response ({len(response)} chars)")
    
    print("\n✅ VERIFICATION COMPLETE - All systems operational!\n")

if __name__ == "__main__":
    quick_verify()