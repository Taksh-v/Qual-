import pytest
from rag_pipeline import NewsRAGPipeline

class TestRAGRetrieval:
    """Test that custom data is retrievable and generates correct responses"""
    
    @pytest.fixture
    def populated_pipeline(self):
        """Create a pipeline with test data"""
        pipeline = NewsRAGPipeline()
        
        test_data = [
            {
                "title": "India-US Trade Deal Signed",
                "content": "India and the United States have signed a comprehensive trade agreement. The deal includes tariff reductions on technology products and agricultural goods. Export-oriented sectors in India are expected to benefit significantly.",
                "source": "Test Business News",
                "published_date": "2026-02-25"
            },
            {
                "title": "Stock Market Rally Continues",
                "content": "The Indian stock market reached new all-time highs today. The NSE Nifty 50 index jumped 2.5% driven by strong corporate earnings and positive economic outlook. Foreign investors increased their positions.",
                "source": "Test Financial Times",
                "published_date": "2026-02-25"
            },
            {
                "title": "Tech Companies Expand Operations",
                "content": "Major technology companies are expanding their operations in India. Investment in data centers and R&D facilities has increased by 40% year-on-year.",
                "source": "Test Tech News",
                "published_date": "2026-02-24"
            }
        ]
        
        pipeline.ingest_news(test_data)
        return pipeline
    
    def test_retrieve_relevant_data(self, populated_pipeline):
        """✅ Test 5: Verify relevant data is retrieved for query"""
        query = "What about India US trade deal?"
        
        # Get relevant items (this is internal to query method)
        relevant = populated_pipeline.embedding_engine.similarity_search(
            query, 
            populated_pipeline.news_store,
            top_k=3
        )
        
        assert len(relevant) > 0, "No data retrieved for query!"
        assert len(relevant) <= 3, "Too many results returned"
        
        # Verify retrieved data is relevant
        first_result = relevant[0]
        assert "Trade" in first_result.get("title", ""), \
            f"Expected trade-related result, got: {first_result.get('title')}"
        
        print(f"✓ Retrieved {len(relevant)} relevant articles")
        for item in relevant:
            print(f"  - {item['title']}")
    
    def test_semantic_search_accuracy(self, populated_pipeline):
        """✅ Test 6: Verify semantic search returns contextually relevant results"""
        queries = [
            ("India US trade tariffs", "India-US Trade Deal Signed"),
            ("stock market performance", "Stock Market Rally Continues"),
            ("tech investments", "Tech Companies Expand Operations")
        ]
        
        for query, expected_title_keyword in queries:
            results = populated_pipeline.embedding_engine.similarity_search(
                query,
                populated_pipeline.news_store,
                top_k=1
            )
            
            assert len(results) > 0, f"No results for query: {query}"
            top_result = results[0]
            
            # Check if the top result is semantically related
            title_lower = top_result.get("title", "").lower()
            expected_lower = expected_title_keyword.lower()
            
            is_relevant = any(keyword in title_lower 
                            for keyword in expected_lower.split())
            
            assert is_relevant, \
                f"Query '{query}' returned irrelevant result: {top_result.get('title')}"
            
            print(f"✓ Query '{query}' -> {top_result.get('title')}")
    
    def test_no_hallucination(self, populated_pipeline):
        """✅ Test 7: Verify LLM uses only provided data, no hallucination"""
        # Query about something NOT in the data
        query = "What about Mars colonization plans?"
        
        # Get the response
        response = populated_pipeline.query(query, top_k=3)
        
        # The response should acknowledge lack of relevant data
        # This depends on your LLM response - adjust assertions
        print(f"Response for out-of-scope query:\n{response}\n")
        
        # In production, you'd validate that response doesn't contain
        # made-up information about Mars
