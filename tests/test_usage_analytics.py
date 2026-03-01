"""
Tests for usage analytics module (Phase 3: Adaptive Learning)
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import usage_analytics
import db


@pytest.fixture
def setup_db():
    """Setup test database state."""
    # Clear test data
    with db.get_cursor() as cur:
        cur.execute("DELETE FROM query_logs WHERE namespace LIKE 'test_%'")
    yield
    # Cleanup
    with db.get_cursor() as cur:
        cur.execute("DELETE FROM query_logs WHERE namespace LIKE 'test_%'")


def test_is_enabled():
    """Test logging enabled check."""
    assert isinstance(usage_analytics.is_enabled(), bool)


def test_sanitize_query_disabled():
    """Test query sanitization when disabled."""
    with patch.dict('os.environ', {'SANITIZE_QUERIES': 'false'}):
        query = "Show me all memories about Python"
        result = usage_analytics.sanitize_query(query)
        assert result == query


def test_sanitize_query_enabled():
    """Test query sanitization when enabled."""
    with patch.dict('os.environ', {'SANITIZE_QUERIES': 'true'}):
        query = "Show me all memories about Python"
        result = usage_analytics.sanitize_query(query)
        assert result.startswith("<sanitized:")
        assert len(result) > 10


def test_log_query_disabled(setup_db):
    """Test that logging is skipped when disabled."""
    with patch('usage_analytics.USAGE_LOGGING_ENABLED', False):
        result = usage_analytics.log_query(
            namespace="test_namespace",
            query="test query",
            weights_used={"similarity": 0.5},
            memories_selected=["mem1", "mem2"],
            total_tokens_used=100,
            total_tokens_budget=200,
            memories_retrieved=10,
            memories_dropped=8,
            computation_time_ms=50.0
        )
        assert result is None


def test_log_query_success(setup_db):
    """Test successful query logging."""
    log_id = usage_analytics.log_query(
        namespace="test_namespace",
        query="test query",
        weights_used={"similarity": 0.4, "importance": 0.4, "recency": 0.2},
        memories_selected=["mem1", "mem2", "mem3"],
        total_tokens_used=150,
        total_tokens_budget=200,
        memories_retrieved=10,
        memories_dropped=7,
        computation_time_ms=45.5,
        user_feedback="good",
        metadata={"search_type": "semantic"}
    )
    
    assert log_id is not None
    
    # Verify data was stored
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM query_logs WHERE id = %s", (log_id,))
        row = cur.fetchone()
        
        assert row is not None
        assert row["namespace"] == "test_namespace"
        assert row["query"] == "test query"
        assert row["total_tokens_used"] == 150
        assert row["total_tokens_budget"] == 200
        assert row["memories_retrieved"] == 10
        assert row["memories_dropped"] == 7
        assert row["computation_time_ms"] == 45.5
        assert row["user_feedback"] == "good"
        
        weights = json.loads(row["weights_used"])
        assert weights["similarity"] == 0.4
        
        memories = json.loads(row["memories_selected"])
        assert len(memories) == 3


def test_get_namespace_analytics(setup_db):
    """Test retrieving analytics for a namespace."""
    # Log some queries
    for i in range(5):
        usage_analytics.log_query(
            namespace="test_analytics",
            query=f"query {i}",
            weights_used={"similarity": 0.5},
            memories_selected=[f"mem{i}"],
            total_tokens_used=100 + i*10,
            total_tokens_budget=200,
            memories_retrieved=10,
            memories_dropped=5,
            computation_time_ms=50.0
        )
    
    # Retrieve analytics
    logs = usage_analytics.get_namespace_analytics("test_analytics", limit=10)
    
    assert len(logs) == 5
    assert all(log["namespace"] == "test_analytics" for log in logs)


def test_get_namespace_analytics_date_filter(setup_db):
    """Test analytics with date filters."""
    # Log queries at different times
    now = datetime.utcnow()
    old_date = now - timedelta(days=10)
    
    usage_analytics.log_query(
        namespace="test_date_filter",
        query="old query",
        weights_used={"similarity": 0.5},
        memories_selected=["mem1"],
        total_tokens_used=100,
        total_tokens_budget=200,
        memories_retrieved=5,
        memories_dropped=2,
        computation_time_ms=30.0
    )
    
    # Query with date filter
    logs = usage_analytics.get_namespace_analytics(
        "test_date_filter",
        start_date=now - timedelta(hours=1)
    )
    
    assert len(logs) >= 1


def test_get_analytics_summary(setup_db):
    """Test analytics summary calculation."""
    # Log multiple queries
    for i in range(10):
        usage_analytics.log_query(
            namespace="test_summary",
            query=f"query {i}",
            weights_used={"similarity": 0.5},
            memories_selected=[f"mem{i}"],
            total_tokens_used=100 + i*5,
            total_tokens_budget=200,
            memories_retrieved=10 + i,
            memories_dropped=5,
            computation_time_ms=40.0 + i
        )
    
    summary = usage_analytics.get_analytics_summary("test_summary")
    
    assert summary["enabled"] is True
    assert summary["total_queries"] == 10
    assert summary["avg_tokens_used"] > 0
    assert summary["avg_tokens_budget"] == 200
    assert summary["avg_token_efficiency"] > 0
    assert summary["avg_memories_retrieved"] > 0
    assert summary["first_query"] is not None
    assert summary["last_query"] is not None


def test_get_top_queries(setup_db):
    """Test getting most frequent queries."""
    # Log repeated queries
    queries = ["query A", "query B", "query A", "query C", "query A"]
    
    for query in queries:
        usage_analytics.log_query(
            namespace="test_top_queries",
            query=query,
            weights_used={"similarity": 0.5},
            memories_selected=["mem1"],
            total_tokens_used=100,
            total_tokens_budget=200,
            memories_retrieved=5,
            memories_dropped=2,
            computation_time_ms=30.0
        )
    
    top_queries = usage_analytics.get_top_queries("test_top_queries", limit=3)
    
    assert len(top_queries) > 0
    # "query A" should be most frequent
    assert top_queries[0][0] == "query A"
    assert top_queries[0][1] == 3


def test_get_weight_usage_stats(setup_db):
    """Test weight usage statistics."""
    # Log queries with different weights
    weights1 = {"similarity": 0.4, "importance": 0.4, "recency": 0.2}
    weights2 = {"similarity": 0.5, "importance": 0.3, "recency": 0.2}
    
    for _ in range(3):
        usage_analytics.log_query(
            namespace="test_weight_stats",
            query="query",
            weights_used=weights1,
            memories_selected=["mem1"],
            total_tokens_used=100,
            total_tokens_budget=200,
            memories_retrieved=5,
            memories_dropped=2,
            computation_time_ms=30.0
        )
    
    for _ in range(2):
        usage_analytics.log_query(
            namespace="test_weight_stats",
            query="query",
            weights_used=weights2,
            memories_selected=["mem1"],
            total_tokens_used=100,
            total_tokens_budget=200,
            memories_retrieved=5,
            memories_dropped=2,
            computation_time_ms=30.0
        )
    
    stats = usage_analytics.get_weight_usage_stats("test_weight_stats")
    
    assert stats["namespace"] == "test_weight_stats"
    assert stats["unique_weight_configs"] == 2
    assert len(stats["weight_configs"]) == 2


def test_cleanup_old_logs(setup_db):
    """Test cleaning up old logs."""
    # Log some queries
    for i in range(5):
        usage_analytics.log_query(
            namespace="test_cleanup",
            query=f"query {i}",
            weights_used={"similarity": 0.5},
            memories_selected=["mem1"],
            total_tokens_used=100,
            total_tokens_budget=200,
            memories_retrieved=5,
            memories_dropped=2,
            computation_time_ms=30.0
        )
    
    # Manually set some to be old (simulate)
    cutoff = datetime.utcnow() - timedelta(days=100)
    with db.get_cursor() as cur:
        cur.execute("""
            UPDATE query_logs
            SET timestamp = %s
            WHERE namespace = 'test_cleanup'
            LIMIT 2
        """, (cutoff,))
    
    # Cleanup with 90 day retention
    deleted = usage_analytics.cleanup_old_logs(retention_days=90)
    
    # Should have deleted 2 old logs
    assert deleted >= 2


def test_export_json(setup_db):
    """Test JSON export."""
    # Log some data
    usage_analytics.log_query(
        namespace="test_export",
        query="test query",
        weights_used={"similarity": 0.5},
        memories_selected=["mem1"],
        total_tokens_used=100,
        total_tokens_budget=200,
        memories_retrieved=5,
        memories_dropped=2,
        computation_time_ms=30.0
    )
    
    # Export
    json_data = usage_analytics.export_namespace_data("test_export", format="json")
    
    data = json.loads(json_data)
    assert data["namespace"] == "test_export"
    assert data["total_queries"] >= 1
    assert "logs" in data


def test_export_csv(setup_db):
    """Test CSV export."""
    # Log some data
    usage_analytics.log_query(
        namespace="test_export_csv",
        query="test query",
        weights_used={"similarity": 0.5},
        memories_selected=["mem1"],
        total_tokens_used=100,
        total_tokens_budget=200,
        memories_retrieved=5,
        memories_dropped=2,
        computation_time_ms=30.0
    )
    
    # Export
    csv_data = usage_analytics.export_namespace_data("test_export_csv", format="csv")
    
    assert isinstance(csv_data, str)
    assert len(csv_data) > 0
    # Should have header row
    assert "namespace" in csv_data or "id" in csv_data


def test_delete_namespace_data(setup_db):
    """Test deleting all data for a namespace."""
    # Log some queries
    for i in range(5):
        usage_analytics.log_query(
            namespace="test_delete",
            query=f"query {i}",
            weights_used={"similarity": 0.5},
            memories_selected=["mem1"],
            total_tokens_used=100,
            total_tokens_budget=200,
            memories_retrieved=5,
            memories_dropped=2,
            computation_time_ms=30.0
        )
    
    # Delete all data
    deleted = usage_analytics.delete_namespace_data("test_delete")
    
    assert deleted == 5
    
    # Verify no data remains
    logs = usage_analytics.get_namespace_analytics("test_delete")
    assert len(logs) == 0


def test_log_query_with_metadata(setup_db):
    """Test logging with custom metadata."""
    metadata = {
        "search_type": "semantic",
        "model": "gpt-4",
        "diversity_threshold": 0.7
    }
    
    log_id = usage_analytics.log_query(
        namespace="test_metadata",
        query="test",
        weights_used={"similarity": 0.5},
        memories_selected=["mem1"],
        total_tokens_used=100,
        total_tokens_budget=200,
        memories_retrieved=5,
        memories_dropped=2,
        computation_time_ms=30.0,
        metadata=metadata
    )
    
    # Verify metadata was stored
    with db.get_cursor() as cur:
        cur.execute("SELECT metadata FROM query_logs WHERE id = %s", (log_id,))
        row = cur.fetchone()
        stored_metadata = json.loads(row["metadata"])
        
        assert stored_metadata["search_type"] == "semantic"
        assert stored_metadata["model"] == "gpt-4"


def test_analytics_summary_with_no_data():
    """Test analytics summary for namespace with no data."""
    summary = usage_analytics.get_analytics_summary("nonexistent_namespace")
    
    assert summary["enabled"] is True
    assert summary["total_queries"] == 0
