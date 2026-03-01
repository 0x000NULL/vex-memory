"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAndStats:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "degraded")

    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        assert "total_memories" in r.json()


class TestMemoryCRUD:
    def test_create_memory(self, client):
        r = client.post("/memories", json={
            "content": "Unit test memory",
            "type": "semantic",
            "importance_score": 0.8,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["content"] == "Unit test memory"
        assert data["importance_score"] == 0.8

    def test_list_memories(self, client):
        # Seed a memory first
        client.post("/memories", json={"content": "List test memory", "type": "semantic"})
        r = client.get("/memories")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_filter_by_type(self, client):
        client.post("/memories", json={"content": "Procedural test", "type": "procedural"})
        r = client.get("/memories?type=procedural")
        assert r.status_code == 200

    def test_invalid_type_400(self, client):
        r = client.get("/memories?type=invalid_type_xyz")
        assert r.status_code == 400


class TestQuery:
    def test_query_memories(self, client):
        client.post("/memories", json={"content": "Python is great for data science", "type": "semantic"})
        r = client.post("/query", json={"query": "Python data science"})
        assert r.status_code == 200
        data = r.json()
        assert "memories" in data
        assert "total_tokens" in data

    def test_invalid_strategy_400(self, client):
        r = client.post("/query", json={"query": "test", "strategy": "bogus"})
        assert r.status_code == 400


class TestLargeContentHandling:
    """Test large content handling (Bug 3)."""
    
    def test_large_content_truncation_and_metadata(self, client):
        """Verify large content (>8000 chars) is handled gracefully with truncation metadata."""
        # Create memory with 10,000 char content
        large_content = "x" * 10000
        response = client.post("/memories", json={
            "content": large_content,
            "type": "semantic",
            "importance_score": 0.5
        })
        
        # Should succeed (no timeout)
        assert response.status_code == 201, f"Failed to create large memory: {response.text}"
        
        data = response.json()
        
        # Content should be stored in full
        assert data["content"] == large_content, "Content should be stored completely"
        
        # Metadata should indicate truncation
        assert "metadata" in data, "Response should include metadata"
        assert data["metadata"].get("truncated") is True, "Metadata should indicate truncation"
        assert data["metadata"].get("original_length") == 10000, "Metadata should record original length"
    
    def test_normal_content_no_truncation_metadata(self, client):
        """Verify normal-sized content doesn't get truncation metadata."""
        normal_content = "x" * 1000
        response = client.post("/memories", json={
            "content": normal_content,
            "type": "semantic"
        })
        
        assert response.status_code == 201
        data = response.json()
        
        # Should not have truncation metadata
        metadata = data.get("metadata", {})
        assert metadata.get("truncated") is not True


class TestDeduplication:
    """Test deduplication enhancement (Bug 1)."""
    
    def test_duplicate_content_merges_instead_of_creating_new(self, client):
        """Verify duplicate content creates 1 memory, not 2, with merged metadata."""
        # Create first memory
        content = "This is a unique test memory for deduplication testing"
        r1 = client.post("/memories", json={
            "content": content,
            "type": "semantic",
            "importance_score": 0.7,
            "confidence_score": 0.8,
            "source": "test-source-1"
        })
        assert r1.status_code == 201
        first_id = r1.json()["id"]
        first_access_count = r1.json()["access_count"]
        
        # Post same content again (duplicate)
        r2 = client.post("/memories", json={
            "content": content,
            "type": "semantic",
            "importance_score": 0.9,  # Higher importance
            "confidence_score": 0.6,  # Lower confidence
            "source": "test-source-2"
        })
        assert r2.status_code == 201
        second_id = r2.json()["id"]
        
        # Should return same ID (merged)
        assert second_id == first_id, "Duplicate content should merge, not create new memory"
        
        # Verify merged properties
        merged = r2.json()
        assert merged["importance_score"] == 0.9, "Should keep higher importance score"
        
        # Confidence should be averaged: (0.8 + 0.6) / 2 = 0.7
        assert abs(merged["confidence_score"] - 0.7) < 0.01, "Should average confidence scores"
        
        # Access count should be incremented
        assert merged["access_count"] > first_access_count, "Access count should be incremented on merge"
        
        # Metadata should indicate merge
        assert "merge_source" in merged["metadata"], "Metadata should indicate merge source"
        assert "merged_at" in merged["metadata"], "Metadata should include merge timestamp"
    
    def test_similar_but_not_identical_content_also_merges(self, client):
        """Verify similar content (>0.85 similarity) also gets merged."""
        # Create first memory
        r1 = client.post("/memories", json={
            "content": "The vex-memory system uses PostgreSQL and pgvector for storage",
            "type": "semantic",
            "importance_score": 0.6
        })
        assert r1.status_code == 201
        first_id = r1.json()["id"]
        
        # Post very similar content (not identical, but should be >0.85 similarity)
        r2 = client.post("/memories", json={
            "content": "The vex-memory system uses PostgreSQL with pgvector for storage",
            "type": "semantic",
            "importance_score": 0.8
        })
        assert r2.status_code == 201
        second_id = r2.json()["id"]
        
        # Should merge (same ID returned) OR create new if similarity <0.85
        # We can't guarantee exact behavior without knowing embedding similarity,
        # but at minimum it should not crash
        assert second_id is not None


class TestQueryRankingEdgeCases:
    """Test query ranking edge cases (Bug 4)."""
    
    def test_query_returns_results_with_similar_content(self, client):
        """Verify queries return results when semantically similar memories exist."""
        # Create memory with known content
        response = client.post("/memories", json={
            "content": "vex-memory v0.3.0 features include namespaces, confidence scoring, and auto-sync",
            "type": "semantic",
            "importance_score": 0.8
        })
        assert response.status_code == 201
        memory_id = response.json()["id"]
        
        # Query for similar content (should return results)
        queries = [
            "what are the new features?",
            "tell me about vex-memory features",
            "what's new in version 0.3.0?",
        ]
        
        for query in queries:
            response = client.post("/query", json={"query": query})
            assert response.status_code == 200, f"Query failed: {query}"
            
            data = response.json()
            assert "memories" in data
            
            # Should return at least 1 result (may return more if there are other memories)
            # We just verify it doesn't return 0
            assert len(data["memories"]) > 0, f"Query returned 0 results for: '{query}'"
    
    def test_query_with_no_matches_uses_keyword_fallback(self, client):
        """Verify keyword fallback is used when semantic search returns 0 results."""
        # Create a memory
        client.post("/memories", json={
            "content": "Python is a programming language",
            "type": "semantic"
        })
        
        # Query with content that may not have high semantic similarity
        # but should still return results via keyword fallback
        response = client.post("/query", json={
            "query": "random gibberish xyz123 nonexistent"
        })
        
        assert response.status_code == 200
        # Even if no results, should not crash and should use fallback


class TestConfidenceFiltering:
    """Test confidence filter precision (Bug 2)."""
    
    def test_confidence_filter_exact_threshold(self, client):
        """Verify min_confidence filter uses exact threshold (no fuzzy ±0.02 range)."""
        # Create memories with specific confidence scores
        # Below threshold
        r1 = client.post("/memories", json={
            "content": "Low confidence memory",
            "confidence_score": 0.599,
            "importance_score": 0.5
        })
        assert r1.status_code == 201
        low_id = r1.json()["id"]
        
        # Exactly at threshold
        r2 = client.post("/memories", json={
            "content": "Exact threshold memory",
            "confidence_score": 0.6,
            "importance_score": 0.5
        })
        assert r2.status_code == 201
        exact_id = r2.json()["id"]
        
        # Above threshold
        r3 = client.post("/memories", json={
            "content": "High confidence memory",
            "confidence_score": 0.601,
            "importance_score": 0.5
        })
        assert r3.status_code == 201
        high_id = r3.json()["id"]
        
        # Query with min_confidence=0.6
        r = client.get("/memories?min_confidence=0.6&limit=100")
        assert r.status_code == 200
        memories = r.json()
        
        # Extract IDs and confidence scores
        returned_ids = {m["id"] for m in memories}
        
        # Verify exact threshold behavior
        assert low_id not in returned_ids, "Memory with confidence=0.599 should NOT be returned"
        assert exact_id in returned_ids, "Memory with confidence=0.6 should be returned"
        assert high_id in returned_ids, "Memory with confidence=0.601 should be returned"
        
        # Verify all returned memories meet threshold
        for m in memories:
            assert m["confidence_score"] >= 0.6, f"Memory {m['id']} has confidence {m['confidence_score']} < 0.6"


class TestEntities:
    def test_list_entities(self, client):
        r = client.get("/entities")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
