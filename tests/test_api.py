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
