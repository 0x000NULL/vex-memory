"""
Unit tests for type and namespace priority weighting.

Tests the priority system that gives different weights to different
memory types and namespaces.
"""

import pytest
from datetime import datetime, timezone
from prioritizer import (
    MemoryPrioritizer,
    PriorityMappings,
    ScoringWeights,
    TokenEstimator
)


class TestPriorityMappings:
    """Test PriorityMappings dataclass."""
    
    def test_default_type_priorities(self):
        """Test default type priority values."""
        mappings = PriorityMappings()
        
        assert mappings.type_priorities["episodic"] == 1.0
        assert mappings.type_priorities["semantic"] == 0.8
        assert mappings.type_priorities["procedural"] == 0.6
        assert mappings.type_priorities["meta"] == 0.4
    
    def test_default_namespace_priorities(self):
        """Test default namespace priority values."""
        mappings = PriorityMappings()
        
        assert mappings.namespace_priorities["main"] == 1.0
        assert mappings.namespace_priorities["shared"] == 0.7
        assert mappings.namespace_priorities["isolated"] == 0.3
    
    def test_custom_type_priorities(self):
        """Test custom type priority values."""
        mappings = PriorityMappings(
            type_priorities={"custom": 0.9, "other": 0.5}
        )
        
        assert mappings.type_priorities["custom"] == 0.9
        assert mappings.type_priorities["other"] == 0.5
    
    def test_custom_namespace_priorities(self):
        """Test custom namespace priority values."""
        mappings = PriorityMappings(
            namespace_priorities={"special": 0.95, "normal": 0.6}
        )
        
        assert mappings.namespace_priorities["special"] == 0.95
        assert mappings.namespace_priorities["normal"] == 0.6
    
    def test_get_type_priority_known(self):
        """Test getting priority for known type."""
        mappings = PriorityMappings()
        
        assert mappings.get_type_priority("episodic") == 1.0
        assert mappings.get_type_priority("semantic") == 0.8
    
    def test_get_type_priority_unknown(self):
        """Test getting priority for unknown type (defaults to 0.5)."""
        mappings = PriorityMappings()
        
        assert mappings.get_type_priority("unknown") == 0.5
        assert mappings.get_type_priority("") == 0.5
    
    def test_get_namespace_priority_known(self):
        """Test getting priority for known namespace."""
        mappings = PriorityMappings()
        
        assert mappings.get_namespace_priority("main") == 1.0
        assert mappings.get_namespace_priority("shared") == 0.7
    
    def test_get_namespace_priority_unknown(self):
        """Test getting priority for unknown namespace (defaults to 0.5)."""
        mappings = PriorityMappings()
        
        assert mappings.get_namespace_priority("unknown") == 0.5
        assert mappings.get_namespace_priority("") == 0.5


class TestTypePriority:
    """Test type priority scoring."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create prioritizer with token estimator."""
        return MemoryPrioritizer(token_estimator=TokenEstimator())
    
    def test_episodic_type_priority(self, prioritizer):
        """Test episodic memories get highest type priority."""
        memory = {
            "id": "test-1",
            "content": "Test memory",
            "type": "episodic",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 1.0
    
    def test_semantic_type_priority(self, prioritizer):
        """Test semantic memories get medium-high type priority."""
        memory = {
            "id": "test-2",
            "content": "Test memory",
            "type": "semantic",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 0.8
    
    def test_procedural_type_priority(self, prioritizer):
        """Test procedural memories get medium type priority."""
        memory = {
            "id": "test-3",
            "content": "Test memory",
            "type": "procedural",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 0.6
    
    def test_meta_type_priority(self, prioritizer):
        """Test meta memories get lowest type priority."""
        memory = {
            "id": "test-4",
            "content": "Test memory",
            "type": "meta",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 0.4
    
    def test_unknown_type_priority(self, prioritizer):
        """Test unknown type defaults to neutral priority (0.5)."""
        memory = {
            "id": "test-5",
            "content": "Test memory",
            "type": "unknown_type",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 0.5
    
    def test_missing_type_priority(self, prioritizer):
        """Test missing type field defaults to neutral priority (0.5)."""
        memory = {
            "id": "test-6",
            "content": "Test memory",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 0.5
    
    def test_case_insensitive_type(self, prioritizer):
        """Test type matching is case-insensitive."""
        memory = {
            "id": "test-7",
            "content": "Test memory",
            "type": "EPISODIC",  # Uppercase
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 1.0


class TestNamespacePriority:
    """Test namespace priority scoring."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create prioritizer with token estimator."""
        return MemoryPrioritizer(token_estimator=TokenEstimator())
    
    def test_main_namespace_priority(self, prioritizer):
        """Test main namespace gets highest priority."""
        memory = {
            "id": "test-1",
            "content": "Test memory",
            "namespace": "main",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 1.0
    
    def test_shared_namespace_priority(self, prioritizer):
        """Test shared namespace gets medium priority."""
        memory = {
            "id": "test-2",
            "content": "Test memory",
            "namespace": "shared",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 0.7
    
    def test_isolated_namespace_priority(self, prioritizer):
        """Test isolated namespace gets lowest priority."""
        memory = {
            "id": "test-3",
            "content": "Test memory",
            "namespace": "isolated",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 0.3
    
    def test_namespace_in_metadata(self, prioritizer):
        """Test namespace can be in metadata field."""
        memory = {
            "id": "test-4",
            "content": "Test memory",
            "metadata": {"namespace": "main"},
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 1.0
    
    def test_unknown_namespace_priority(self, prioritizer):
        """Test unknown namespace defaults to neutral priority (0.5)."""
        memory = {
            "id": "test-5",
            "content": "Test memory",
            "namespace": "unknown_ns",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 0.5
    
    def test_missing_namespace_priority(self, prioritizer):
        """Test missing namespace defaults to neutral priority (0.5)."""
        memory = {
            "id": "test-6",
            "content": "Test memory",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 0.5
    
    def test_case_insensitive_namespace(self, prioritizer):
        """Test namespace matching is case-insensitive."""
        memory = {
            "id": "test-7",
            "content": "Test memory",
            "namespace": "MAIN",  # Uppercase
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["namespace_priority"] == 1.0


class TestCombinedPriorities:
    """Test combined type and namespace priorities."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create prioritizer with token estimator."""
        return MemoryPrioritizer(token_estimator=TokenEstimator())
    
    def test_high_priority_memory(self, prioritizer):
        """Test memory with both high type and namespace priority."""
        memory = {
            "id": "high-pri",
            "content": "Important episodic memory in main namespace",
            "type": "episodic",
            "namespace": "main",
            "importance_score": 0.8,
            "similarity_score": 0.9,
            "event_time": datetime.now(timezone.utc)
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        # Type priority = 1.0, namespace priority = 1.0
        # Multiplier = (1.0 + 1.0) / 2 = 1.0
        assert score_obj.factors["type_priority"] == 1.0
        assert score_obj.factors["namespace_priority"] == 1.0
        assert score_obj.score >= 0.75  # Should have high score
    
    def test_low_priority_memory(self, prioritizer):
        """Test memory with both low type and namespace priority."""
        memory = {
            "id": "low-pri",
            "content": "Meta memory in isolated namespace",
            "type": "meta",
            "namespace": "isolated",
            "importance_score": 0.8,
            "similarity_score": 0.9,
            "event_time": datetime.now(timezone.utc)
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        # Type priority = 0.4, namespace priority = 0.3
        # Multiplier = (0.4 + 0.3) / 2 = 0.35
        assert score_obj.factors["type_priority"] == 0.4
        assert score_obj.factors["namespace_priority"] == 0.3
        # Score should be reduced by ~65% due to low priorities
        assert score_obj.score < 0.4
    
    def test_mixed_priority_memory(self, prioritizer):
        """Test memory with mixed priorities."""
        memory = {
            "id": "mixed-pri",
            "content": "Semantic memory in isolated namespace",
            "type": "semantic",
            "namespace": "isolated",
            "importance_score": 0.8,
            "similarity_score": 0.9,
            "event_time": datetime.now(timezone.utc)
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        # Type priority = 0.8, namespace priority = 0.3
        # Multiplier = (0.8 + 0.3) / 2 = 0.55
        assert score_obj.factors["type_priority"] == 0.8
        assert score_obj.factors["namespace_priority"] == 0.3
        # Score should be moderate
        assert 0.4 < score_obj.score < 0.7


class TestPrioritySelection:
    """Test that priorities affect memory selection."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create prioritizer with token estimator."""
        return MemoryPrioritizer(token_estimator=TokenEstimator())
    
    def test_high_priority_selected_first(self, prioritizer):
        """Test that high priority memories are selected first."""
        now = datetime.now(timezone.utc)
        
        memories = [
            {
                "id": "low",
                "content": "Low priority meta/isolated memory",
                "type": "meta",
                "namespace": "isolated",
                "importance_score": 0.9,
                "similarity_score": 0.9,
                "event_time": now
            },
            {
                "id": "high",
                "content": "High priority episodic/main memory",
                "type": "episodic",
                "namespace": "main",
                "importance_score": 0.5,
                "similarity_score": 0.5,
                "event_time": now
            }
        ]
        
        selected, metadata = prioritizer.prioritize(
            memories=memories,
            token_budget=500
        )
        
        # High priority should be selected first despite lower base scores
        assert selected[0]["id"] == "high"
    
    def test_priority_affects_ranking(self, prioritizer):
        """Test that priorities affect overall ranking."""
        now = datetime.now(timezone.utc)
        
        memories = [
            {
                "id": "episodic-main",
                "content": f"Memory {i}",
                "type": "episodic",
                "namespace": "main",
                "importance_score": 0.5,
                "similarity_score": 0.5,
                "event_time": now
            }
            for i in range(3)
        ] + [
            {
                "id": "meta-isolated",
                "content": f"Memory {i}",
                "type": "meta",
                "namespace": "isolated",
                "importance_score": 0.9,
                "similarity_score": 0.9,
                "event_time": now
            }
            for i in range(3)
        ]
        
        selected, metadata = prioritizer.prioritize(
            memories=memories,
            token_budget=500
        )
        
        # Episodic/main should dominate selection despite lower base scores
        episodic_count = sum(1 for m in selected if m["type"] == "episodic")
        assert episodic_count >= 2


class TestPriorityMappingUpdates:
    """Test updating priority mappings."""
    
    @pytest.fixture
    def prioritizer(self):
        """Create prioritizer with token estimator."""
        return MemoryPrioritizer(token_estimator=TokenEstimator())
    
    def test_update_priority_mappings(self, prioritizer):
        """Test updating priority mappings."""
        new_mappings = PriorityMappings(
            type_priorities={"custom": 0.95},
            namespace_priorities={"special": 0.85}
        )
        
        prioritizer.update_priority_mappings(new_mappings)
        
        assert prioritizer.priority_mappings.type_priorities["custom"] == 0.95
        assert prioritizer.priority_mappings.namespace_priorities["special"] == 0.85
    
    def test_get_priority_mappings(self, prioritizer):
        """Test getting current priority mappings."""
        mappings = prioritizer.get_priority_mappings()
        
        assert isinstance(mappings, PriorityMappings)
        assert mappings.type_priorities["episodic"] == 1.0
    
    def test_custom_mappings_affect_scoring(self, prioritizer):
        """Test that custom mappings affect scoring."""
        # Set custom mappings
        custom_mappings = PriorityMappings(
            type_priorities={"vip": 2.0},  # Very high priority
            namespace_priorities={"vip_ns": 2.0}
        )
        prioritizer.update_priority_mappings(custom_mappings)
        
        memory = {
            "id": "vip-memory",
            "content": "VIP memory",
            "type": "vip",
            "namespace": "vip_ns",
            "importance_score": 0.5,
            "similarity_score": 0.5
        }
        
        score_obj = prioritizer._score_memory(memory)
        
        assert score_obj.factors["type_priority"] == 2.0
        assert score_obj.factors["namespace_priority"] == 2.0
        # Score should be boosted significantly (close to 1.0 with 2x multiplier)
        assert score_obj.score >= 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
