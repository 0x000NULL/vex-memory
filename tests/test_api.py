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


class TestEntities:
    def test_list_entities(self, client):
        r = client.get("/entities")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
