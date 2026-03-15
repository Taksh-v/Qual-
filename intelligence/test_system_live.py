import json
import time
from datetime import datetime
from intelligence.rag_pipeline import NewsRAGPipeline
from run_news_ingestion import ingest_news

class LiveSystemTester:
    """Interactive testing of the complete system"""
    
    def __init__(self):
        self.pipeline = NewsRAGPipeline()
        self.test_results = {
            "data_ingestion": False,
            "embedding_creation": False,
            "retrieval": False,
            "llm_response": False,
            "summary": {}
        }
    
    def test_step_1_ingest_custom_data(self):
        """Step 1: Ingest actual news data"""
        print("\n" + "="*60)
        print("TEST STEP 1: INGESTING CUSTOM NEWS DATA")
        print("="*60)
        
        try:
            print("Extracting news from URLs...")
            news_items = ingest_news()  # Your actual ingestion function
            
            print(f"✅ Successfully extracted {len(news_items)} news articles")
            print("\nSample of ingested data:")
            
            for i, item in enumerate(news_items[:3], 1):
                print(f"\n  Article {i}:")
                print(f"    Title: {item.get('title', 'N/A')[:60]}...")
                print(f"    Source: {item.get('source', 'N/A')}")
                print(f"    Published: {item.get('published_date', 'N/A')}")
            
            self.test_results["data_ingestion"] = True
            self.test_results["summary"]["articles_extracted"] = len(news_items)
            
            return news_items
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["data_ingestion"] = False
            return []
    
    def test_step_2_create_embeddings(self, news_items):
        """Step 2: Create embeddings for all data"""
        print("\n" + "="*60)
        print("TEST STEP 2: CREATING EMBEDDINGS")
        print("="*60)
        
        try:
            print(f"Embedding {len(news_items)} articles...")
            start_time = time.time()
            
            self.pipeline.ingest_news(news_items)
            
            elapsed = time.time() - start_time
            
            print(f"✅ Successfully created embeddings in {elapsed:.2f} seconds")
            print(f"   Total items in system: {len(self.pipeline.news_store)}")
            
            # Verify embeddings
            sample_item = self.pipeline.news_store[0]
            embedding_size = len(sample_item.get("embedding", []))
            
            print(f"   Embedding dimension: {embedding_size}")
            print(f"   Embedding sample: {sample_item['embedding'][:5]}...")
            
            self.test_results["embedding_creation"] = True
            self.test_results["summary"]["embeddings_created"] = len(self.pipeline.news_store)
            self.test_results["summary"]["embedding_dimension"] = embedding_size
            
            return True
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["embedding_creation"] = False
            return False
    
    def test_step_3_semantic_search(self):
        """Step 3: Test semantic search retrieval"""
        print("\n" + "="*60)
        print("TEST STEP 3: TESTING SEMANTIC SEARCH RETRIEVAL")
        print("="*60)
        
        test_queries = [
            "What are the latest India-US trade developments?",
            "Tell me about stock market performance",
            "Any news about government policies?",
            "What happened with business deals?"
        ]
        
        try:
            for query in test_queries:
                print(f"\n📝 Query: '{query}'")
                
                results = self.pipeline.embedding_engine.similarity_search(
                    query,
                    self.pipeline.news_store,
                    top_k=3
                )
                
                if results:
                    print(f"   ✅ Found {len(results)} relevant articles:")
                    for i, result in enumerate(results, 1):
                        print(f"      {i}. {result.get('title', 'N/A')[:70]}...")
                        print(f"         Source: {result.get('source', 'N/A')}")
                else:
                    print(f"   ⚠️  No results found")
            
            self.test_results["retrieval"] = True
            return True
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["retrieval"] = False
            return False
    
    def test_step_4_llm_response_generation(self):
        """Step 4: Test LLM response generation"""
        print("\n" + "="*60)
        print("TEST STEP 4: TESTING LLM RESPONSE GENERATION")
        print("="*60)
        
        test_question = "Based on the news I provided, what's the overall market sentiment?"
        
        try:
            print(f"❓ Question: {test_question}\n")
            print("Generating response from Ollama Phi3:mini...\n")
            print("-" * 60)
            
            start_time = time.time()
            
            # Generate response with streaming
            response = self.pipeline.query(test_question, top_k=5, use_streaming=True)
            
            elapsed = time.time() - start_time
            
            print("-" * 60)
            print(f"\n✅ Response generated in {elapsed:.2f} seconds")
            print(f"   Response length: {len(response)} characters")
            
            self.test_results["llm_response"] = True
            self.test_results["summary"]["response_time"] = elapsed
            
            return True
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            print("\nTroubleshooting:")
            print("  - Is Ollama running? (Check: http://localhost:11434)")
            print("  - Is phi3:mini model installed? (Run: ollama pull phi3:mini)")
            self.test_results["llm_response"] = False
            return False
    
    def test_step_5_data_consistency(self):
        """Step 5: Verify data consistency"""
        print("\n" + "="*60)
        print("TEST STEP 5: VERIFYING DATA CONSISTENCY")
        print("="*60)
        
        checks = {
            "No duplicate titles": True,
            "All items have embeddings": True,
            "All embeddings same dimension": True,
            "All metadata preserved": True
        }
        
        try:
            titles = set()
            embedding_dims = set()
            
            for item in self.pipeline.news_store:
                # Check for duplicates
                title = item.get("title")
                if title in titles:
                    checks["No duplicate titles"] = False
                titles.add(title)
                
                # Check embeddings
                if "embedding" not in item:
                    checks["All items have embeddings"] = False
                else:
                    embedding_dims.add(len(item["embedding"]))
                
                # Check metadata
                required_fields = ["title", "content", "source"]
                for field in required_fields:
                    if field not in item:
                        checks["All metadata preserved"] = False
            
            if len(embedding_dims) == 1:
                print(f"✅ All embeddings have consistent dimension: {embedding_dims.pop()}")
            else:
                checks["All embeddings same dimension"] = False
            
            print(f"✅ Total unique articles: {len(titles)}")
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}")
            
            self.test_results["summary"]["consistency_checks"] = checks
            
            return all(checks.values())
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n")
        print("╔" + "="*58 + "╗")
        print("║" + " "*15 + "SYSTEM VERIFICATION TEST SUITE" + " "*12 + "║")
        print("║" + " "*58 + "║")
        print("║ Testing custom data integration with Ollama Phi3:mini RAG" + " "*1 + "║")
        print("╚" + "="*58 + "╝")
        
        # Run tests
        news_items = self.test_step_1_ingest_custom_data()
        
        if news_items:
            if self.test_step_2_create_embeddings(news_items):
                self.test_step_3_semantic_search()
                self.test_step_4_llm_response_generation()
                self.test_step_5_data_consistency()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results) - 1  # exclude 'summary'
        passed_tests = sum(1 for k, v in self.test_results.items() 
                          if k != 'summary' and v is True)
        
        print(f"\nTests Passed: {passed_tests}/{total_tests}")
        
        status_symbols = {
            True: "✅",
            False: "❌"
        }
        
        for test_name, status in self.test_results.items():
            if test_name != 'summary':
                symbol = status_symbols.get(status, "⚠️")
                print(f"  {symbol} {test_name.replace('_', ' ').title()}")
        
        print("\nDetailed Metrics:")
        for key, value in self.test_results.get("summary", {}).items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*60)
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            print(f"⚠️  {total_tests - passed_tests} test(s) failed. Review output above.")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    tester = LiveSystemTester()
    tester.run_all_tests()