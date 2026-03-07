import pytest
import json
from datetime import datetime
from embedding_engine import EmbeddingEngine
from rag_pipeline import NewsRAGPipeline

class TestDataIngestion:
    """Test that custom data is properly ingested and embedded"""
    
    @pytest.fixture
    def sample_news_data(self):
        """Sample news data matching your ingestion format"""
        return [
            {
                "title": "Test Article 1: India-US Trade Deal",
                "content": "India and US finalize trade agreement with tariff reductions...",
                "source": "Test Source 1",
                "url": "https://example.com/article1",
                "published_date": "2026-02-25"
            },
            {
                "title": "Test Article 2: Stock Market Rally",
                "content": "Stock market reaches all-time high amid positive economic indicators...",
                "source": "Test Source 2",
                "url": "https://example.com/article2",
                "published_date": "2026-02-25"
            }
        ]
    
    @pytest.fixture
    def rag_pipeline(self):
        """Initialize RAG pipeline for testing"""
        return NewsRAGPipeline()
    
    def test_data_is_ingested(self, rag_pipeline, sample_news_data):
        """✅ Test 1: Verify data is actually ingested into the system"""
        initial_count = len(rag_pipeline.news_store)
        
        # Ingest data
        rag_pipeline.ingest_news(sample_news_data)
        
        final_count = len(rag_pipeline.news_store)
        
        # Assert data was added
        assert final_count > initial_count, "News data was not ingested!"
        assert final_count == initial_count + len(sample_news_data), \
            f"Expected {len(sample_news_data)} items, got {final_count - initial_count}"
    
    def test_embeddings_created(self, rag_pipeline, sample_news_data):
        """✅ Test 2: Verify embeddings are created for each article"""
        rag_pipeline.ingest_news(sample_news_data)
        
        # Check each item has embedding
        for item in rag_pipeline.news_store:
            assert "embedding" in item, f"No embedding for: {item.get('title')}"
            assert isinstance(item["embedding"], list), "Embedding is not a list"
            assert len(item["embedding"]) > 0, "Embedding is empty"
            print(f"✓ {item['title']} -> Embedding size: {len(item['embedding'])}")
    
    def test_embedding_dimensions(self, rag_pipeline, sample_news_data):
        """✅ Test 3: Verify embedding dimensions are consistent"""
        rag_pipeline.ingest_news(sample_news_data)
        
        dimensions = set()
        for item in rag_pipeline.news_store:
            dim = len(item["embedding"])
            dimensions.add(dim)
        
        assert len(dimensions) == 1, "Embedding dimensions are inconsistent!"
        print(f"✓ All embeddings have consistent dimension: {dimensions.pop()}")
    
    def test_metadata_preserved(self, rag_pipeline, sample_news_data):
        """✅ Test 4: Verify original metadata is preserved"""
        rag_pipeline.ingest_news(sample_news_data)
        
        for i, original in enumerate(sample_news_data):
            ingested = rag_pipeline.news_store[i]
            
            # Check metadata
            assert ingested.get("title") == original["title"], "Title not preserved"
            assert ingested.get("content") == original["content"], "Content not preserved"
            assert ingested.get("source") == original["source"], "Source not preserved"
            assert ingested.get("url") == original["url"], "URL not preserved"
            
            print(f"✓ Metadata preserved for: {original['title']}")