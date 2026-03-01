"""
Unit tests for weight tuner module.

Tests weight optimization, preset configurations, and benchmarking.
"""

import pytest
from datetime import datetime, timezone
from weight_tuner import (
    WeightTuner,
    WeightConfig,
    BenchmarkResult,
    get_recommended_weights
)
from prioritizer import ScoringWeights, TokenEstimator


class TestWeightConfig:
    """Test WeightConfig dataclass."""
    
    def test_weight_config_creation(self):
        """Test creating a weight config."""
        weights = ScoringWeights(
            similarity=0.4,
            importance=0.3,
            recency=0.2,
            diversity=0.05,
            entity_coverage=0.05
        )
        
        config = WeightConfig(
            weights=weights,
            name="Test Config",
            description="Test description"
        )
        
        assert config.weights == weights
        assert config.name == "Test Config"
        assert config.description == "Test description"
    
    def test_to_dict(self):
        """Test converting weight config to dictionary."""
        weights = ScoringWeights(
            similarity=0.4,
            importance=0.3,
            recency=0.2,
            diversity=0.05,
            entity_coverage=0.05
        )
        
        config = WeightConfig(
            weights=weights,
            name="Test",
            description="Desc"
        )
        
        d = config.to_dict()
        
        assert d["name"] == "Test"
        assert d["description"] == "Desc"
        assert d["weights"]["similarity"] == 0.4
        assert d["weights"]["importance"] == 0.3


class TestWeightTunerPresets:
    """Test WeightTuner preset configurations."""
    
    @pytest.fixture
    def tuner(self):
        """Create weight tuner."""
        return WeightTuner()
    
    def test_presets_exist(self, tuner):
        """Test that all expected presets exist."""
        expected_presets = [
            "balanced",
            "relevance_focused",
            "recency_focused",
            "diversity_focused",
            "entity_focused",
            "importance_focused"
        ]
        
        for preset in expected_presets:
            config = tuner.get_preset(preset)
            assert config is not None
            assert isinstance(config, WeightConfig)
    
    def test_balanced_preset(self, tuner):
        """Test balanced preset values."""
        config = tuner.get_preset("balanced")
        
        assert config.name == "Balanced"
        assert 0.3 <= config.weights.similarity <= 0.4
        assert 0.25 <= config.weights.importance <= 0.35
    
    def test_relevance_focused_preset(self, tuner):
        """Test relevance_focused preset."""
        config = tuner.get_preset("relevance_focused")
        
        assert config.name == "Relevance Focused"
        # Should prioritize similarity
        assert config.weights.similarity > 0.5
    
    def test_recency_focused_preset(self, tuner):
        """Test recency_focused preset."""
        config = tuner.get_preset("recency_focused")
        
        assert config.name == "Recency Focused"
        # Should prioritize recency
        assert config.weights.recency > 0.35
    
    def test_diversity_focused_preset(self, tuner):
        """Test diversity_focused preset."""
        config = tuner.get_preset("diversity_focused")
        
        assert config.name == "Diversity Focused"
        # Should have higher diversity and entity coverage
        assert config.weights.diversity > 0.15
    
    def test_entity_focused_preset(self, tuner):
        """Test entity_focused preset."""
        config = tuner.get_preset("entity_focused")
        
        assert config.name == "Entity Focused"
        # Should prioritize entity coverage
        assert config.weights.entity_coverage > 0.2
    
    def test_importance_focused_preset(self, tuner):
        """Test importance_focused preset."""
        config = tuner.get_preset("importance_focused")
        
        assert config.name == "Importance Focused"
        # Should prioritize importance
        assert config.weights.importance >= 0.5
    
    def test_unknown_preset(self, tuner):
        """Test getting unknown preset returns None."""
        config = tuner.get_preset("nonexistent")
        assert config is None
    
    def test_list_presets(self, tuner):
        """Test listing all presets."""
        presets = tuner.list_presets()
        
        assert len(presets) >= 6
        
        # Check structure
        for preset in presets:
            assert "name" in preset
            assert "key" in preset
            assert "description" in preset


class TestBenchmarking:
    """Test weight configuration benchmarking."""
    
    @pytest.fixture
    def tuner(self):
        """Create weight tuner."""
        return WeightTuner(token_estimator=TokenEstimator())
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        now = datetime.now(timezone.utc)
        
        return [
            {
                "id": f"mem-{i}",
                "content": f"Memory {i} with content",
                "importance_score": 0.5 + (i % 5) * 0.1,
                "event_time": now,
                "similarity_score": 0.5 + (i % 10) * 0.05
            }
            for i in range(20)
        ]
    
    def test_benchmark_config(self, tuner, sample_memories):
        """Test benchmarking a weight configuration."""
        config = tuner.get_preset("balanced")
        
        result = tuner.benchmark_config(
            config=config,
            memories=sample_memories,
            token_budget=500
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.config == config
        assert result.score >= 0.0
        assert len(result.metrics) > 0
        assert "count" in result.metrics
        assert "utilization" in result.metrics
    
    def test_benchmark_with_ground_truth(self, tuner, sample_memories):
        """Test benchmarking with ground truth comparison."""
        config = tuner.get_preset("balanced")
        ground_truth = ["mem-0", "mem-1", "mem-2", "mem-5"]
        
        result = tuner.benchmark_config(
            config=config,
            memories=sample_memories,
            token_budget=500,
            ground_truth=ground_truth
        )
        
        assert "ground_truth_overlap" in result.metrics
        assert "ground_truth_recall" in result.metrics
        assert result.metrics["ground_truth_recall"] >= 0.0
        assert result.metrics["ground_truth_recall"] <= 1.0
    
    def test_benchmark_with_custom_evaluate(self, tuner, sample_memories):
        """Test benchmarking with custom evaluation function."""
        config = tuner.get_preset("balanced")
        
        # Custom eval: score based on number of memories selected
        def custom_eval(selected, metadata):
            return len(selected) / 10.0  # Score = count / 10
        
        result = tuner.benchmark_config(
            config=config,
            memories=sample_memories,
            token_budget=500,
            evaluate_fn=custom_eval
        )
        
        # Score should be based on count
        expected_score = result.metrics["count"] / 10.0
        assert abs(result.score - expected_score) < 0.01
    
    def test_compare_presets(self, tuner, sample_memories):
        """Test comparing all preset configurations."""
        results = tuner.compare_presets(
            memories=sample_memories,
            token_budget=500
        )
        
        # Should have results for all presets
        assert len(results) >= 6
        
        # Should be sorted by score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
        
        # All should have valid metrics
        for result in results:
            assert result.score >= 0.0
            assert "count" in result.metrics


class TestGridSearch:
    """Test grid search functionality."""
    
    @pytest.fixture
    def tuner(self):
        """Create weight tuner."""
        return WeightTuner(token_estimator=TokenEstimator())
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        now = datetime.now(timezone.utc)
        
        return [
            {
                "id": f"mem-{i}",
                "content": f"Memory {i} content",
                "importance_score": 0.5,
                "event_time": now,
                "similarity_score": 0.5 + (i % 10) * 0.05
            }
            for i in range(15)
        ]
    
    def test_grid_search_basic(self, tuner, sample_memories):
        """Test basic grid search."""
        results = tuner.grid_search(
            memories=sample_memories,
            token_budget=300,
            similarity_range=(0.3, 0.5, 0.1),  # 3 values
            importance_range=(0.2, 0.4, 0.1),  # 3 values
            recency_range=(0.2, 0.3, 0.1),     # 2 values
            top_k=5
        )
        
        # Should return top 5 results
        assert len(results) <= 5
        
        # Should be sorted by score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_grid_search_with_ground_truth(self, tuner, sample_memories):
        """Test grid search with ground truth."""
        ground_truth = ["mem-0", "mem-1", "mem-2"]
        
        results = tuner.grid_search(
            memories=sample_memories,
            token_budget=300,
            similarity_range=(0.3, 0.5, 0.2),
            importance_range=(0.2, 0.4, 0.2),
            recency_range=(0.1, 0.3, 0.1),
            ground_truth=ground_truth,
            top_k=3
        )
        
        assert len(results) > 0
        
        # All should have ground truth metrics
        for result in results:
            assert "ground_truth_recall" in result.metrics
    
    def test_frange_helper(self):
        """Test the _frange helper method."""
        values = WeightTuner._frange(0.1, 0.3, 0.1)
        
        assert len(values) == 3
        assert values[0] == 0.1
        assert values[1] == 0.2
        assert values[2] == 0.3


class TestRecommendations:
    """Test weight recommendations."""
    
    @pytest.fixture
    def tuner(self):
        """Create weight tuner."""
        return WeightTuner()
    
    def test_recommend_balanced(self, tuner):
        """Test balanced recommendation."""
        config = tuner.recommend_weights("balanced")
        
        assert config.name == "Balanced"
        assert isinstance(config.weights, ScoringWeights)
    
    def test_recommend_relevance(self, tuner):
        """Test relevance_focused recommendation."""
        config = tuner.recommend_weights("relevance_focused")
        
        assert config.name == "Relevance Focused"
    
    def test_recommend_unknown_defaults_to_balanced(self, tuner):
        """Test unknown use case defaults to balanced."""
        config = tuner.recommend_weights("unknown_use_case")
        
        assert config.name == "Balanced"
    
    def test_convenience_function(self):
        """Test convenience function for getting weights."""
        weights = get_recommended_weights("balanced")
        
        assert isinstance(weights, dict)
        assert "similarity" in weights
        assert "importance" in weights
        assert "recency" in weights
        assert "diversity" in weights
        assert "entity_coverage" in weights


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""
    
    def test_to_dict(self):
        """Test converting benchmark result to dictionary."""
        config = WeightConfig(
            weights=ScoringWeights(),
            name="Test",
            description="Desc"
        )
        
        result = BenchmarkResult(
            config=config,
            score=0.85,
            metrics={"count": 5, "utilization": 0.9},
            selected_memory_ids=["a", "b", "c"]
        )
        
        d = result.to_dict()
        
        assert d["score"] == 0.85
        assert d["metrics"]["count"] == 5
        assert d["selected_count"] == 3
        assert "config" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
