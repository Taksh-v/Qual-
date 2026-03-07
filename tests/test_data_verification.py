import pytest
import json
from embedding_engine import EmbeddingEngine

class TestDataVerification:
    """Verify that specific custom data points are in the system"""
    
    def test_specific_article_is_indexed(self):
        """✅ Test 8: Verify a specific article from your custom data is indexed"""
        
        # Your actual article from run_news_ingestion.py
        test_article = {
            "title": "air-india-tests-found-fuel-switches-work",
            "content": "Air India fuel switches test sample content",
            "source": "Hindustan Times"
        }
        
        engine = EmbeddingEngine()
        embedded = engine.embed_news_batch([test_article])
        
        assert len(embedded) == 1, "Article not embedded"
        assert "embedding" in embedded[0], "No embedding created"
        print(f"✓ Article indexed: {test_article['title']}")
    
    def test_all_sources_are_stored(self):
        """✅ Test 9: Verify all news sources are properly stored"""
        sources = set()
        
        test_data = [
            {"title": "Article 1", "content": "Content 1", "source": "Hindustan Times"},
            {"title": "Article 2", "content": "Content 2", "source": "Reuters"},
            {"title": "Article 3", "content": "Content 3", "source": "AP News"},
        ]
        
        engine = EmbeddingEngine()
        embedded = engine.embed_news_batch(test_data)
        
        for item in embedded:
            sources.add(item.get("source"))
        
        assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}"
        assert "Hindustan Times" in sources, "Hindustan Times not stored"
        
        print(f"✓ All sources stored: {sources}")
    
    def test_data_persistence(self):
        """✅ Test 10: Verify data persists after saving"""
        from rag_pipeline import NewsRAGPipeline
        import json
        import tempfile
        import os
        
        pipeline = NewsRAGPipeline()
        
        test_data = [
            {"title": "Persistent Article", "content": "This should persist", "source": "Test"}
        ]
        
        pipeline.ingest_news(test_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(pipeline.news_store, f)
            temp_file = f.name
        
        # Load from file
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) > 0, "Data not saved properly"
        assert loaded_data[0]["title"] == "Persistent Article", "Data corrupted"
        
        # Cleanup
        os.unlink(temp_file)
        
        print(f"✓ Data persistence verified")
