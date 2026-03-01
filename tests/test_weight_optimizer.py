"""
Tests for weight optimizer module (Phase 3: Adaptive Learning)
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import weight_optimizer
import usage_analytics
import db


@pytest.fixture
def setup_db():
    """Setup test database state."""
    # Clear test data
    with db.get_cursor() as cur:
        cur.execute("DELETE FROM query_logs WHERE namespace LIKE 'test_%'")
        cur.execute("DELETE FROM learned_weights WHERE namespace LIKE 'test_%'")
        cur.execute("DELETE FROM optimization_history WHERE namespace LIKE 'test_%'")
        cur.execute("DELETE FROM memory_nodes WHERE namespace_id IN (SELECT namespace_id FROM memory_namespaces WHERE name LIKE 'test_%')")
    yield
    # Cleanup
    with db.get_cursor() as cur:
        cur.execute("DELETE FROM query_logs WHERE namespace LIKE 'test_%'")
        cur.execute("DELETE FROM learned_weights WHERE namespace LIKE 'test_%'")
        cur.execute("DELETE FROM optimization_history WHERE namespace LIKE 'test_%'")


def create_test_memories(namespace: str, count: int = 10):
    """Helper to create test memories."""
    memory_ids = []
    
    with db.get_cursor() as cur:
        # Ensure namespace exists
        cur.execute("""
            INSERT INTO memory_namespaces (name, owner_agent)
            VALUES (%s, 'test_agent')
            ON CONFLICT (name) DO NOTHING
            RETURNING namespace_id
        """, (namespace,))
        
        result = cur.fetchone()
        if result:
            namespace_id = result["namespace_id"]
        else:
            cur.execute("SELECT namespace_id FROM memory_namespaces WHERE name = %s", (namespace,))
            namespace_id = cur.fetchone()["namespace_id"]
        
        # Create memories
        for i in range(count):
            cur.execute("""
                INSERT INTO memory_nodes (type, content, importance_score, namespace_id)
                VALUES ('semantic', %s, %s, %s)
                RETURNING id
            """, (f"Test memory {i} about topic {i % 3}", 0.5 + i*0.01, namespace_id))
            
            memory_ids.append(str(cur.fetchone()["id"]))
    
    return memory_ids


def create_test_query_logs(namespace: str, count: int = 60, memory_ids: list = None):
    """Helper to create test query logs."""
    if not memory_ids:
        memory_ids = create_test_memories(namespace)
    
    for i in range(count):
        # Vary the parameters to create diverse logs
        weights = {
            "similarity": 0.3 + (i % 3) * 0.1,
            "importance": 0.3 + (i % 4) * 0.1,
            "recency": 0.2 + (i % 2) * 0.1
        }
        
        selected = memory_ids[i % len(memory_ids):(i % len(memory_ids)) + 3]
        
        usage_analytics.log_query(
            namespace=namespace,
            query=f"test query {i}",
            weights_used=weights,
            memories_selected=selected,
            total_tokens_used=100 + i*5,
            total_tokens_budget=200,
            memories_retrieved=10,
            memories_dropped=7,
            computation_time_ms=40.0 + i*0.5
        )


def test_calculate_diversity_score():
    """Test diversity score calculation."""
    memory_contents = {
        "mem1": "The quick brown fox jumps over the lazy dog",
        "mem2": "The lazy dog sleeps under the tree",
        "mem3": "A completely different sentence about cats and birds"
    }
    
    # Test with 2 similar memories
    score1 = weight_optimizer.calculate_diversity_score(
        ["mem1", "mem2"],
        memory_contents
    )
    assert 0 <= score1 <= 1
    
    # Test with dissimilar memories
    score2 = weight_optimizer.calculate_diversity_score(
        ["mem1", "mem3"],
        memory_contents
    )
    assert score2 > score1  # Should be more diverse
    
    # Test with single memory
    score3 = weight_optimizer.calculate_diversity_score(
        ["mem1"],
        memory_contents
    )
    assert score3 == 0.0  # No diversity with single memory


def test_evaluate_weights(setup_db):
    """Test weight evaluation on validation data."""
    namespace = "test_evaluate"
    memory_ids = create_test_memories(namespace, count=10)
    
    # Create validation logs
    validation_logs = []
    for i in range(5):
        log_id = usage_analytics.log_query(
            namespace=namespace,
            query=f"query {i}",
            weights_used={"similarity": 0.4, "importance": 0.4, "recency": 0.2},
            memories_selected=memory_ids[i:i+3],
            total_tokens_used=150,
            total_tokens_budget=200,
            memories_retrieved=10,
            memories_dropped=7,
            computation_time_ms=45.0
        )
    
    logs = usage_analytics.get_namespace_analytics(namespace)
    
    # Get memory contents
    memory_contents = {}
    with db.get_cursor() as cur:
        cur.execute("SELECT id::text, content FROM memory_nodes WHERE id = ANY(%s)", (memory_ids,))
        for row in cur.fetchall():
            memory_contents[row["id"]] = row["content"]
    
    # Evaluate weights
    weights = {"similarity": 0.4, "importance": 0.4, "recency": 0.2}
    score = weight_optimizer.evaluate_weights(weights, logs, memory_contents)
    
    assert isinstance(score, float)
    assert score >= 0


def test_grid_search_insufficient_data(setup_db):
    """Test grid search with insufficient data."""
    namespace = "test_insufficient"
    
    # Create only 10 query logs (need 50)
    create_test_query_logs(namespace, count=10)
    
    with pytest.raises(ValueError, match="Insufficient data"):
        weight_optimizer.grid_search_weights(namespace, min_queries=50)


def test_grid_search_success(setup_db):
    """Test successful grid search."""
    namespace = "test_grid_search"
    memory_ids = create_test_memories(namespace, count=15)
    
    # Create sufficient query logs
    create_test_query_logs(namespace, count=60, memory_ids=memory_ids)
    
    # Run grid search with small search space
    search_space = {
        "similarity": [0.3, 0.5],
        "importance": [0.3, 0.4],
        "recency": [0.2]
    }
    
    best_weights, best_score, metadata = weight_optimizer.grid_search_weights(
        namespace=namespace,
        search_space=search_space,
        min_queries=50
    )
    
    assert best_weights is not None
    assert "similarity" in best_weights
    assert "importance" in best_weights
    assert "recency" in best_weights
    assert isinstance(best_score, float)
    assert metadata["training_queries"] > 0
    assert metadata["validation_queries"] > 0
    assert metadata["combinations_tested"] > 0


def test_save_learned_weights(setup_db):
    """Test saving learned weights."""
    namespace = "test_save_weights"
    weights = {"similarity": 0.45, "importance": 0.35, "recency": 0.2}
    
    weight_id = weight_optimizer.save_learned_weights(
        namespace=namespace,
        weights=weights,
        objective_score=1.25,
        training_queries=100,
        optimization_method="grid_search",
        metadata={"avg_diversity_score": 0.65, "avg_token_efficiency": 0.6}
    )
    
    assert weight_id is not None
    
    # Verify it was saved
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM learned_weights WHERE id = %s", (weight_id,))
        row = cur.fetchone()
        
        assert row is not None
        assert row["namespace"] == namespace
        assert row["is_active"] is True
        assert row["objective_score"] == 1.25
        assert row["training_queries"] == 100


def test_save_learned_weights_deactivates_old(setup_db):
    """Test that saving new weights deactivates old ones."""
    namespace = "test_deactivate"
    
    # Save first weights
    weight_id1 = weight_optimizer.save_learned_weights(
        namespace=namespace,
        weights={"similarity": 0.4, "importance": 0.4, "recency": 0.2},
        objective_score=1.0,
        training_queries=50
    )
    
    # Save second weights
    weight_id2 = weight_optimizer.save_learned_weights(
        namespace=namespace,
        weights={"similarity": 0.5, "importance": 0.3, "recency": 0.2},
        objective_score=1.2,
        training_queries=60
    )
    
    # Check that first is deactivated
    with db.get_cursor() as cur:
        cur.execute("SELECT is_active FROM learned_weights WHERE id = %s", (weight_id1,))
        row1 = cur.fetchone()
        assert row1["is_active"] is False
        
        cur.execute("SELECT is_active FROM learned_weights WHERE id = %s", (weight_id2,))
        row2 = cur.fetchone()
        assert row2["is_active"] is True


def test_get_learned_weights(setup_db):
    """Test retrieving learned weights."""
    namespace = "test_get_weights"
    weights = {"similarity": 0.45, "importance": 0.35, "recency": 0.2}
    
    # Save weights
    weight_optimizer.save_learned_weights(
        namespace=namespace,
        weights=weights,
        objective_score=1.25,
        training_queries=100,
        metadata={"avg_diversity_score": 0.65}
    )
    
    # Retrieve weights
    learned = weight_optimizer.get_learned_weights(namespace)
    
    assert learned is not None
    assert learned["namespace"] == namespace
    assert learned["weights"]["similarity"] == 0.45
    assert learned["objective_score"] == 1.25


def test_get_learned_weights_none(setup_db):
    """Test retrieving weights for namespace with none."""
    learned = weight_optimizer.get_learned_weights("nonexistent_namespace")
    assert learned is None


def test_log_optimization_run(setup_db):
    """Test logging optimization run to history."""
    namespace = "test_history"
    best_weights = {"similarity": 0.5, "importance": 0.3, "recency": 0.2}
    metadata = {
        "training_queries": 80,
        "validation_queries": 20,
        "combinations_tested": 12,
        "computation_time_ms": 150.5,
        "search_space": {"similarity": [0.3, 0.5]}
    }
    
    history_id = weight_optimizer.log_optimization_run(
        namespace=namespace,
        best_weights=best_weights,
        best_score=1.3,
        metadata=metadata
    )
    
    assert history_id is not None
    
    # Verify it was logged
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM optimization_history WHERE id = %s", (history_id,))
        row = cur.fetchone()
        
        assert row is not None
        assert row["namespace"] == namespace
        assert row["best_score"] == 1.3
        assert row["training_queries"] == 80


def test_optimize_namespace_full_workflow(setup_db):
    """Test complete optimization workflow."""
    namespace = "test_optimize_full"
    memory_ids = create_test_memories(namespace, count=20)
    
    # Create sufficient query logs
    create_test_query_logs(namespace, count=60, memory_ids=memory_ids)
    
    # Run optimization with small search space
    search_space = {
        "similarity": [0.4, 0.5],
        "importance": [0.3, 0.4],
        "recency": [0.2]
    }
    
    result = weight_optimizer.optimize_namespace(
        namespace=namespace,
        search_space=search_space,
        min_queries=50
    )
    
    # Verify result structure
    assert "weight_id" in result
    assert "history_id" in result
    assert "namespace" in result
    assert "best_weights" in result
    assert "objective_score" in result
    assert "metadata" in result
    
    # Verify weights were saved
    learned = weight_optimizer.get_learned_weights(namespace)
    assert learned is not None
    assert learned["id"] == result["weight_id"]


def test_optimize_namespace_insufficient_data(setup_db):
    """Test optimization with insufficient data."""
    namespace = "test_insufficient_optimize"
    
    # Create only 20 query logs
    create_test_query_logs(namespace, count=20)
    
    with pytest.raises(ValueError):
        weight_optimizer.optimize_namespace(namespace, min_queries=50)


def test_weight_normalization():
    """Test that weights are properly normalized."""
    namespace = "test_normalize"
    
    # Weights that don't sum to 1.0
    weights = {"similarity": 0.5, "importance": 0.5, "recency": 0.3}
    
    weight_id = weight_optimizer.save_learned_weights(
        namespace=namespace,
        weights=weights,
        objective_score=1.0,
        training_queries=50
    )
    
    # Note: Grid search normalizes weights during evaluation
    # This test verifies the storage doesn't fail


def test_diversity_score_empty_content():
    """Test diversity calculation with missing content."""
    memory_contents = {
        "mem1": "Some text here",
        "mem2": ""  # Empty content
    }
    
    score = weight_optimizer.calculate_diversity_score(
        ["mem1", "mem2"],
        memory_contents
    )
    
    # Should handle empty content gracefully
    assert isinstance(score, float)
    assert score >= 0


def test_evaluate_weights_empty_logs():
    """Test evaluation with empty logs."""
    score = weight_optimizer.evaluate_weights(
        {"similarity": 0.5, "importance": 0.3, "recency": 0.2},
        [],
        {}
    )
    
    assert score == 0.0
