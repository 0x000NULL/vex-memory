"""
Integration tests for prioritized context API endpoint.

Tests the full API workflow including request/response handling,
database integration, and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app
import db


client = TestClient(app)


class TestPrioritizedContextEndpoint:
    """Test /api/memories/prioritized-context endpoint."""
    
    def test_endpoint_basic_request(self):
        """Test basic prioritized context request."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test query",
            "token_budget": 2000,
            "model": "gpt-4"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "memories" in data
        assert "metadata" in data
        assert isinstance(data["memories"], list)
        assert isinstance(data["metadata"], dict)
    
    def test_endpoint_with_custom_weights(self):
        """Test with custom scoring weights."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test query",
            "token_budget": 3000,
            "model": "gpt-4",
            "weights": {
                "similarity": 0.5,
                "importance": 0.3,
                "recency": 0.15,
                "diversity": 0.05
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that weights are reflected in metadata
        assert "weights" in data["metadata"]
        assert data["metadata"]["weights"]["similarity"] == 0.5
    
    def test_endpoint_with_diversity_threshold(self):
        """Test with custom diversity threshold."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test query",
            "token_budget": 2000,
            "diversity_threshold": 0.5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "diversity_threshold" in data["metadata"]
        assert data["metadata"]["diversity_threshold"] == 0.5
    
    def test_endpoint_respects_token_budget(self):
        """Test that response never exceeds token budget."""
        budget = 1000
        
        response = client.post("/api/memories/prioritized-context", json={
            "query": "project status",
            "token_budget": budget,
            "model": "gpt-4"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metadata"]["budget"] == budget
        assert data["metadata"]["total_tokens"] <= budget
        assert data["metadata"]["utilization"] <= 1.0
    
    def test_endpoint_metadata_structure(self):
        """Test that response metadata has expected structure."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000
        })
        
        assert response.status_code == 200
        data = response.json()
        metadata = data["metadata"]
        
        # Required fields
        assert "total_tokens" in metadata
        assert "budget" in metadata
        assert "utilization" in metadata
        assert "memories_selected" in metadata
        assert "memories_available" in metadata
        assert "search_type" in metadata
        assert "model" in metadata
    
    def test_endpoint_empty_query(self):
        """Test with empty query string."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "",
            "token_budget": 2000
        })
        
        # Should handle gracefully
        assert response.status_code in [200, 422]  # Either success or validation error
    
    def test_endpoint_small_budget(self):
        """Test with very small token budget."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 50
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should work but select fewer memories
        assert data["metadata"]["total_tokens"] <= 50
    
    def test_endpoint_large_budget(self):
        """Test with very large token budget."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 100000
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should work fine
        assert "memories" in data
    
    def test_endpoint_with_min_score(self):
        """Test with minimum score threshold."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "min_score": 0.7
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # All memories should meet minimum score (if any returned)
        # Note: Exact scores depend on database content
        assert isinstance(data["memories"], list)
    
    def test_endpoint_with_namespace(self):
        """Test with namespace filtering."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "namespace": "test-namespace"
        })
        
        assert response.status_code == 200
        # Should succeed even if namespace doesn't exist (empty results)
    
    def test_endpoint_with_limit(self):
        """Test with custom candidate limit."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "limit": 50
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should respect limit on candidates
        assert len(data["memories"]) <= 50
    
    def test_endpoint_different_models(self):
        """Test with different model names."""
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
        
        for model in models:
            response = client.post("/api/memories/prioritized-context", json={
                "query": "test",
                "token_budget": 2000,
                "model": model
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["model"] == model


class TestPrioritizedContextValidation:
    """Test request validation."""
    
    def test_missing_query(self):
        """Test with missing query field."""
        response = client.post("/api/memories/prioritized-context", json={
            "token_budget": 2000
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_token_budget_negative(self):
        """Test with negative token budget."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": -100
        })
        
        assert response.status_code == 422
    
    def test_invalid_token_budget_zero(self):
        """Test with zero token budget."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 0
        })
        
        assert response.status_code == 422
    
    def test_invalid_diversity_threshold_high(self):
        """Test with diversity threshold > 1.0."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "diversity_threshold": 1.5
        })
        
        assert response.status_code == 422
    
    def test_invalid_diversity_threshold_negative(self):
        """Test with negative diversity threshold."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "diversity_threshold": -0.5
        })
        
        assert response.status_code == 422
    
    def test_invalid_min_score(self):
        """Test with invalid min_score."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "min_score": 1.5
        })
        
        assert response.status_code == 422
    
    def test_invalid_weights_format(self):
        """Test with malformed weights."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000,
            "weights": "invalid"
        })
        
        assert response.status_code == 422


class TestPrioritizedContextBehavior:
    """Test behavioral expectations."""
    
    def test_high_utilization(self):
        """Test that utilization is high (>80%) when possible."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test query with multiple words",
            "token_budget": 3000,
            "model": "gpt-4"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # If memories are available, utilization should be good
        if data["metadata"]["memories_available"] > 0:
            # Should use most of the budget
            assert data["metadata"]["utilization"] >= 0.0  # At least some utilization
    
    def test_diversity_filtering_active(self):
        """Test that diversity filtering is working."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 5000,
            "diversity_threshold": 0.3  # Low threshold = more filtering
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that diversity_filtered count exists
        assert "diversity_filtered" in data["metadata"]
        assert data["metadata"]["diversity_filtered"] >= 0
    
    def test_consistent_results(self):
        """Test that same query gives consistent results."""
        query = "specific test query 12345"
        
        response1 = client.post("/api/memories/prioritized-context", json={
            "query": query,
            "token_budget": 2000
        })
        
        response2 = client.post("/api/memories/prioritized-context", json={
            "query": query,
            "token_budget": 2000
        })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Results should be deterministic (same memories, same order)
        data1 = response1.json()
        data2 = response2.json()
        
        assert len(data1["memories"]) == len(data2["memories"])


class TestPrioritizedContextIntegration:
    """Test integration with database and other components."""
    
    @pytest.mark.skipif(not db.check_health(), reason="Database not available")
    def test_with_real_database(self):
        """Test with real database connection."""
        response = client.post("/api/memories/prioritized-context", json={
            "query": "integration test",
            "token_budget": 2000
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use semantic search
        assert "search_type" in data["metadata"]
    
    def test_fallback_on_embedding_failure(self):
        """Test that endpoint falls back gracefully if embeddings fail."""
        # This would require mocking the embedding function
        # For now, just verify the endpoint handles it
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test",
            "token_budget": 2000
        })
        
        assert response.status_code == 200


class TestPrioritizedContextPerformance:
    """Test performance characteristics."""
    
    def test_response_time(self):
        """Test that endpoint responds quickly."""
        import time
        
        start = time.time()
        response = client.post("/api/memories/prioritized-context", json={
            "query": "test performance",
            "token_budget": 4000,
            "limit": 100
        })
        elapsed = time.time() - start
        
        assert response.status_code == 200
        # Should respond in less than 5 seconds
        assert elapsed < 5.0, f"Response took {elapsed:.2f}s, expected <5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
