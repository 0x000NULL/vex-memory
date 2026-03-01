"""
Tests for confidence scoring system
"""

import pytest
from confidence import assign_confidence, bulk_assign_confidence


class TestConfidenceScoring:
    """Test auto-tagging accuracy"""
    
    def test_high_confidence_markers(self):
        """Definite statements should have high confidence"""
        content = "Ethan's birthday is December 20, 1995"
        score = assign_confidence(content, "semantic", {})
        assert score >= 0.85, f"Expected high confidence, got {score}"
    
    def test_medium_confidence_markers(self):
        """Likely/probably statements should have medium confidence"""
        content = "Ethan probably prefers dark mode based on past choices"
        score = assign_confidence(content, "semantic", {})
        assert 0.6 <= score <= 0.85, f"Expected medium confidence, got {score}"
    
    def test_low_confidence_markers(self):
        """Uncertain statements should have low confidence"""
        content = "Maybe the server is on port 8080, but I'm not sure"
        score = assign_confidence(content, "semantic", {})
        assert score <= 0.6, f"Expected low confidence, got {score}"
    
    def test_episodic_memory_high_confidence(self):
        """Episodic memories (witnessed events) should have high base confidence"""
        content = "User deployed the API at 3:00 PM today"
        score = assign_confidence(content, "episodic", {})
        assert score >= 0.8, f"Expected high confidence for episodic, got {score}"
    
    def test_emotional_memory_medium_confidence(self):
        """Emotional memories are subjective, medium confidence"""
        content = "User seemed frustrated with the Docker setup"
        score = assign_confidence(content, "emotional", {})
        assert 0.6 <= score <= 0.85, f"Expected medium confidence for emotional, got {score}"
    
    def test_procedural_memory_confidence(self):
        """Procedural how-tos should have decent confidence"""
        content = "To deploy: run docker-compose up -d and wait for containers"
        score = assign_confidence(content, "procedural", {})
        assert score >= 0.7, f"Expected good confidence for procedural, got {score}"
    
    def test_verified_source_boost(self):
        """Verified sources should boost confidence"""
        content = "The API runs on port 3000"
        metadata = {"source": "verified", "verified": True}
        score = assign_confidence(content, "semantic", metadata)
        assert score >= 0.85, f"Expected high confidence for verified source, got {score}"
    
    def test_question_mark_reduces_confidence(self):
        """Questions should have lower confidence"""
        content = "Is the database running on localhost?"
        score = assign_confidence(content, "semantic", {})
        assert score <= 0.7, f"Expected lower confidence for question, got {score}"
    
    def test_specific_details_boost(self):
        """Specific numbers/dates should increase confidence"""
        content = "Server version 2.3.1 deployed on 2026-02-15 at 14:30"
        score = assign_confidence(content, "semantic", {})
        assert score >= 0.8, f"Expected high confidence for specific details, got {score}"
    
    def test_bulk_assignment(self):
        """Bulk assignment should work correctly"""
        memories = [
            {"id": "1", "content": "Fact: Earth is round", "type": "semantic", "metadata": {}},
            {"id": "2", "content": "Maybe it will rain tomorrow", "type": "semantic", "metadata": {}},
        ]
        results = bulk_assign_confidence(memories)
        assert len(results) == 2
        assert results["1"] > results["2"], "Definite fact should have higher confidence than speculation"


class TestQueryFiltering:
    """Test query filtering by confidence"""
    
    @pytest.mark.skipif(True, reason="Requires database connection")
    def test_min_confidence_filter(self):
        """Filter memories by minimum confidence threshold"""
        # This would require actual DB connection
        # Placeholder for integration test
        pass


class TestRankingWithConfidence:
    """Test ranking boost from confidence scores"""
    
    def test_confidence_affects_ranking(self):
        """Higher confidence should boost ranking score"""
        from retriever import MemoryRetriever
        from extractor import MemoryNode, MemoryType
        
        # Create test memories with different confidence
        mem_high_conf = MemoryNode(
            id="high",
            content="Python supports async/await syntax",
            type=MemoryType.SEMANTIC,
            importance_score=0.5,
        )
        mem_high_conf.confidence_score = 0.95
        
        mem_low_conf = MemoryNode(
            id="low",
            content="Python might support some concurrency features",
            type=MemoryType.SEMANTIC,
            importance_score=0.5,
        )
        mem_low_conf.confidence_score = 0.4
        
        retriever = MemoryRetriever(memory_file=None)
        retriever.memories = [mem_high_conf, mem_low_conf]
        retriever._build_indices()
        
        from retriever import QueryContext
        qc = QueryContext(query="Python concurrency", max_tokens=1000)
        
        # Calculate scores
        score_high = retriever._calculate_relevance_score(mem_high_conf, qc)
        score_low = retriever._calculate_relevance_score(mem_low_conf, qc)
        
        # High confidence should contribute to higher score
        # Given same importance (0.5) and similar content relevance,
        # confidence difference (0.95 vs 0.4) * 0.2 weight = 0.11 boost
        assert score_high > score_low, f"High confidence ({score_high:.3f}) should rank higher than low ({score_low:.3f})"


class TestEdgeCases:
    """Test edge cases and validation"""
    
    def test_empty_content(self):
        """Empty content should return default confidence"""
        score = assign_confidence("", "semantic", {})
        assert 0.0 <= score <= 1.0, "Score should be in valid range"
    
    def test_very_long_content(self):
        """Long content should get a small boost"""
        short = "Fact"
        long = "This is a very detailed and comprehensive explanation " * 10
        score_short = assign_confidence(short, "semantic", {})
        score_long = assign_confidence(long, "semantic", {})
        # Long content should have slightly higher confidence
        assert score_long >= score_short
    
    def test_score_bounds(self):
        """All scores should be clamped to [0, 1]"""
        test_cases = [
            "definitely confirmed verified absolutely certain",
            "maybe possibly might could uncertain guess",
            "neutral statement without markers",
        ]
        for content in test_cases:
            score = assign_confidence(content, "semantic", {})
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for: {content}"
    
    def test_importance_correlation(self):
        """High importance should correlate with higher confidence"""
        content = "Important verified fact"
        metadata_high = {"importance_score": 0.95}
        metadata_low = {"importance_score": 0.2}
        
        score_high_imp = assign_confidence(content, "semantic", metadata_high)
        score_low_imp = assign_confidence(content, "semantic", metadata_low)
        
        assert score_high_imp >= score_low_imp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
