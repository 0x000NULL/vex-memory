"""
Unit tests for prioritizer module.

Tests memory scoring, prioritization, diversity filtering, and budget enforcement.
"""

import pytest
from datetime import datetime, timedelta, timezone
from prioritizer import (
    MemoryPrioritizer,
    ScoringWeights,
    MemoryScore,
    prioritize_memories
)
from token_estimator import TokenEstimator


class TestScoringWeights:
    """Test ScoringWeights dataclass."""
    
    def test_default_weights(self):
        """Test default weight values (v1.2.0 with entity_coverage)."""
        weights = ScoringWeights()
        
        assert weights.similarity == 0.4
        assert weights.importance == 0.3
        assert weights.recency == 0.2
        assert weights.diversity == 0.05
        assert weights.entity_coverage == 0.05
    
    def test_custom_weights(self):
        """Test custom weight values."""
        weights = ScoringWeights(
            similarity=0.5,
            importance=0.3,
            recency=0.15,
            diversity=0.03,
            entity_coverage=0.02
        )
        
        assert weights.similarity == 0.5
        assert weights.importance == 0.3
    
    def test_weights_normalization(self):
        """Test that weights are normalized if they don't sum to 1."""
        weights = ScoringWeights(
            similarity=0.5,
            importance=0.5,
            recency=0.5,
            diversity=0.5,
            entity_coverage=0.5
        )
        
        # Should be normalized to sum to 1.0
        total = weights.similarity + weights.importance + weights.recency + weights.diversity + weights.entity_coverage
        assert abs(total - 1.0) < 0.01


class TestMemoryPrioritizer:
    """Test MemoryPrioritizer class."""
    
    @pytest.fixture
    def estimator(self):
        """Create token estimator."""
        return TokenEstimator()
    
    @pytest.fixture
    def prioritizer(self, estimator):
        """Create prioritizer."""
        return MemoryPrioritizer(token_estimator=estimator)
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        now = datetime.now(timezone.utc)
        
        return [
            {
                "id": "mem-1",
                "content": "Important recent memory about project launch",
                "importance_score": 0.9,
                "event_time": now - timedelta(days=1),
                "similarity_score": 0.95
            },
            {
                "id": "mem-2",
                "content": "Less important old memory about past meeting",
                "importance_score": 0.3,
                "event_time": now - timedelta(days=100),
                "similarity_score": 0.5
            },
            {
                "id": "mem-3",
                "content": "Medium importance recent memory",
                "importance_score": 0.6,
                "event_time": now - timedelta(days=7),
                "similarity_score": 0.7
            }
        ]
    
    def test_initialization(self, estimator):
        """Test prioritizer initialization."""
        prioritizer = MemoryPrioritizer(token_estimator=estimator)
        
        assert prioritizer.token_estimator is not None
        assert prioritizer.weights is not None
        assert prioritizer.recency_half_life_days == 30.0
    
    def test_initialization_custom_weights(self, estimator):
        """Test initialization with custom weights."""
        weights = ScoringWeights(similarity=0.5, importance=0.45, recency=0.0, diversity=0.0, entity_coverage=0.05)
        prioritizer = MemoryPrioritizer(token_estimator=estimator, weights=weights)
        
        assert prioritizer.weights.similarity == 0.5
        assert prioritizer.weights.importance == 0.45
    
    def test_prioritize_basic(self, prioritizer, sample_memories):
        """Test basic prioritization."""
        selected, metadata = prioritizer.prioritize(
            memories=sample_memories,
            token_budget=1000
        )
        
        assert len(selected) > 0
        assert len(selected) <= len(sample_memories)
        assert metadata["total_tokens"] <= 1000
        assert metadata["budget"] == 1000
        assert metadata["memories_available"] == 3
    
    def test_prioritize_empty_memories(self, prioritizer):
        """Test prioritization with empty memory list."""
        selected, metadata = prioritizer.prioritize(
            memories=[],
            token_budget=1000
        )
        
        assert len(selected) == 0
        assert metadata["total_tokens"] == 0
        assert metadata["utilization"] == 0.0
    
    def test_prioritize_respects_budget(self, prioritizer, sample_memories):
        """Test that prioritization never exceeds token budget."""
        budget = 100  # Small budget
        
        selected, metadata = prioritizer.prioritize(
            memories=sample_memories,
            token_budget=budget
        )
        
        assert metadata["total_tokens"] <= budget
        assert metadata["utilization"] <= 1.0
    
    def test_prioritize_high_score_first(self, prioritizer, sample_memories):
        """Test that higher-scored memories are selected first."""
        selected, metadata = prioritizer.prioritize(
            memories=sample_memories,
            token_budget=10000  # Large budget
        )
        
        # First memory should be the most important recent one
        assert selected[0]["id"] == "mem-1"
    
    def test_prioritize_with_min_score(self, prioritizer, sample_memories):
        """Test minimum score filtering."""
        selected, metadata = prioritizer.prioritize(
            memories=sample_memories,
            token_budget=10000,
            min_score=0.7
        )
        
        # Should filter out low-scored memories
        assert all(m.get("_score", 0) >= 0.7 for m in selected)
    
    def test_diversity_filtering(self, prioritizer):
        """Test diversity filtering removes similar memories."""
        now = datetime.now(timezone.utc)
        
        similar_memories = [
            {
                "id": "1",
                "content": "The quick brown fox jumps over lazy dog",
                "importance_score": 0.8,
                "event_time": now,
                "similarity_score": 0.9
            },
            {
                "id": "2",
                "content": "The quick brown fox jumps over the lazy dog",  # Very similar
                "importance_score": 0.8,
                "event_time": now,
                "similarity_score": 0.85
            },
            {
                "id": "3",
                "content": "Completely different topic about databases",
                "importance_score": 0.7,
                "event_time": now,
                "similarity_score": 0.8
            }
        ]
        
        selected, metadata = prioritizer.prioritize(
            memories=similar_memories,
            token_budget=10000,
            diversity_threshold=0.5
        )
        
        # Should filter out one of the similar memories
        assert metadata["diversity_filtered"] > 0
    
    def test_score_memory(self, prioritizer):
        """Test memory scoring."""
        now = datetime.now(timezone.utc)
        
        memory = {
            "id": "test",
            "content": "Test memory",
            "importance_score": 0.8,
            "event_time": now,
            "similarity_score": 0.9
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert isinstance(score_obj, MemoryScore)
        assert score_obj.memory_id == "test"
        assert 0.0 <= score_obj.score <= 1.0
        assert score_obj.token_count > 0
        assert "similarity" in score_obj.factors
        assert "importance" in score_obj.factors
        assert "recency" in score_obj.factors
    
    def test_calculate_recency_recent(self, prioritizer):
        """Test recency calculation for recent memory."""
        now = datetime.now(timezone.utc)
        
        memory = {
            "id": "recent",
            "event_time": now - timedelta(days=1)
        }
        
        recency = prioritizer._calculate_recency(memory)
        
        assert recency > 0.95  # Very recent should have high score
        assert recency <= 1.0
    
    def test_calculate_recency_old(self, prioritizer):
        """Test recency calculation for old memory."""
        now = datetime.now(timezone.utc)
        
        memory = {
            "id": "old",
            "event_time": now - timedelta(days=365)
        }
        
        recency = prioritizer._calculate_recency(memory)
        
        assert recency < 0.1  # Very old should have low score
        assert recency >= 0.0
    
    def test_calculate_recency_no_time(self, prioritizer):
        """Test recency calculation when no timestamp."""
        memory = {"id": "no-time"}
        
        recency = prioritizer._calculate_recency(memory)
        
        assert recency == 0.5  # Neutral score
    
    def test_jaccard_similarity(self, prioritizer):
        """Test Jaccard similarity calculation."""
        text1 = "the quick brown fox"
        text2 = "the quick brown dog"
        
        similarity = prioritizer._jaccard_similarity(text1, text2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be somewhat similar
    
    def test_jaccard_similarity_identical(self, prioritizer):
        """Test Jaccard similarity with identical texts."""
        text = "the quick brown fox"
        
        similarity = prioritizer._jaccard_similarity(text, text)
        
        assert similarity == 1.0
    
    def test_jaccard_similarity_different(self, prioritizer):
        """Test Jaccard similarity with completely different texts."""
        text1 = "the quick brown fox"
        text2 = "completely different words"
        
        similarity = prioritizer._jaccard_similarity(text1, text2)
        
        assert similarity < 0.3  # Should be very different
    
    def test_is_too_similar(self, prioritizer):
        """Test similarity checking."""
        memory = {"id": "test", "content": "the quick brown fox"}
        selected = ["the quick brown fox jumps"]
        
        is_similar = prioritizer._is_too_similar(memory, selected, threshold=0.5)
        
        assert is_similar  # Should be too similar
    
    def test_is_not_too_similar(self, prioritizer):
        """Test when memories are different enough."""
        memory = {"id": "test", "content": "the quick brown fox"}
        selected = ["completely different content"]
        
        is_similar = prioritizer._is_too_similar(memory, selected, threshold=0.5)
        
        assert not is_similar
    
    def test_update_weights(self, prioritizer):
        """Test weight updates."""
        new_weights = ScoringWeights(similarity=0.6, importance=0.35, recency=0.0, diversity=0.0, entity_coverage=0.05)
        
        prioritizer.update_weights(new_weights)
        
        assert prioritizer.weights.similarity == 0.6
        assert prioritizer.weights.importance == 0.35
    
    def test_get_weights(self, prioritizer):
        """Test getting current weights."""
        weights = prioritizer.get_weights()
        
        assert isinstance(weights, ScoringWeights)
        assert weights.similarity == 0.4  # Default


class TestConvenienceFunction:
    """Test module-level convenience function."""
    
    def test_prioritize_memories_function(self):
        """Test prioritize_memories convenience function."""
        now = datetime.now(timezone.utc)
        
        memories = [
            {
                "id": "1",
                "content": "Test memory",
                "importance_score": 0.8,
                "event_time": now,
                "similarity_score": 0.9
            }
        ]
        
        selected, metadata = prioritize_memories(
            memories=memories,
            token_budget=1000
        )
        
        assert len(selected) > 0
        assert metadata["total_tokens"] <= 1000


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create prioritizer."""
        return MemoryPrioritizer(token_estimator=TokenEstimator())
    
    def test_single_memory_exceeds_budget(self, prioritizer):
        """Test when single memory exceeds entire budget."""
        long_memory = {
            "id": "long",
            "content": "word " * 10000,  # Very long
            "importance_score": 0.9,
            "similarity_score": 0.95
        }
        
        selected, metadata = prioritizer.prioritize(
            memories=[long_memory],
            token_budget=100
        )
        
        # Should handle gracefully (either truncate or skip)
        assert metadata["total_tokens"] <= 100
    
    def test_all_memories_filtered_by_diversity(self, prioritizer):
        """Test when all memories are too similar."""
        identical_memories = [
            {
                "id": str(i),
                "content": "identical content",
                "importance_score": 0.8,
                "similarity_score": 0.9
            }
            for i in range(5)
        ]
        
        selected, metadata = prioritizer.prioritize(
            memories=identical_memories,
            token_budget=10000,
            diversity_threshold=0.5
        )
        
        # Should keep at least one
        assert len(selected) >= 1
    
    def test_missing_optional_fields(self, prioritizer):
        """Test memories with missing optional fields."""
        minimal_memory = {
            "id": "minimal",
            "content": "Just content"
        }
        
        selected, metadata = prioritizer.prioritize(
            memories=[minimal_memory],
            token_budget=1000
        )
        
        assert len(selected) > 0
    
    def test_string_event_time(self, prioritizer):
        """Test with event_time as string instead of datetime."""
        memory = {
            "id": "string-time",
            "content": "Test",
            "importance_score": 0.8,
            "event_time": "2024-01-01T00:00:00Z",
            "similarity_score": 0.9
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.score > 0
        assert score_obj.factors["recency"] >= 0.0


class TestPerformance:
    """Test performance requirements."""
    
    def test_prioritize_1000_memories_performance(self):
        """Test that prioritization completes quickly with 1000 memories."""
        import time
        
        prioritizer = MemoryPrioritizer(token_estimator=TokenEstimator())
        now = datetime.now(timezone.utc)
        
        # Generate 1000 memories
        memories = [
            {
                "id": f"mem-{i}",
                "content": f"Memory {i} with some content",
                "importance_score": 0.5 + (i % 5) * 0.1,
                "event_time": now - timedelta(days=i % 100),
                "similarity_score": 0.5 + (i % 50) * 0.01
            }
            for i in range(1000)
        ]
        
        start = time.time()
        selected, metadata = prioritizer.prioritize(
            memories=memories,
            token_budget=4000
        )
        elapsed = time.time() - start
        
        # Should complete in less than 100ms
        assert elapsed < 0.1, f"Prioritization took {elapsed:.3f}s, expected <0.1s"
        assert len(selected) > 0
        assert metadata["total_tokens"] <= 4000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
