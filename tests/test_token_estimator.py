"""
Unit tests for token_estimator module.

Tests token counting accuracy, memory estimation, and budget enforcement.
"""

import pytest
from datetime import datetime, timezone
from token_estimator import TokenEstimator, TokenEstimate, count_tokens


class TestTokenEstimator:
    """Test TokenEstimator class."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = TokenEstimator()
        assert estimator.model == "gpt-4"
        assert estimator.encoding_name == "cl100k_base"
        assert estimator.encoder is not None
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        estimator = TokenEstimator(model="gpt-3.5-turbo")
        assert estimator.model == "gpt-3.5-turbo"
        assert estimator.encoding_name == "cl100k_base"
    
    def test_count_tokens_simple(self):
        """Test basic token counting."""
        estimator = TokenEstimator()
        
        # Simple text
        text = "Hello, world!"
        count = estimator.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)
        assert count <= 10  # Should be small for short text
    
    def test_count_tokens_empty(self):
        """Test counting empty string."""
        estimator = TokenEstimator()
        assert estimator.count_tokens("") == 0
        assert estimator.count_tokens("   ") >= 0
    
    def test_count_tokens_long(self):
        """Test counting long text."""
        estimator = TokenEstimator()
        
        # Generate long text
        long_text = "word " * 1000
        count = estimator.count_tokens(long_text)
        assert count > 500  # Should be substantial
        assert count < 2000  # But not 1:1 with words
    
    def test_estimate_memory_basic(self):
        """Test memory estimation."""
        estimator = TokenEstimator()
        
        memory = {
            "id": "test-1",
            "content": "This is a test memory",
            "importance_score": 0.8,
            "event_time": datetime(2024, 1, 1, tzinfo=timezone.utc)
        }
        
        estimate = estimator.estimate_memory(memory)
        
        assert isinstance(estimate, TokenEstimate)
        assert estimate.token_count > 0
        assert estimate.model == "gpt-4"
        assert not estimate.truncated
        assert "test memory" in estimate.text
    
    def test_estimate_memory_with_metadata(self):
        """Test memory estimation with metadata."""
        estimator = TokenEstimator()
        
        memory = {
            "id": "test-2",
            "content": "Memory with metadata",
            "importance_score": 0.9,
            "metadata": {"key": "value"}
        }
        
        estimate = estimator.estimate_memory(memory)
        assert estimate.token_count > 0
    
    def test_truncate_to_budget_no_truncation(self):
        """Test truncation when text fits budget."""
        estimator = TokenEstimator()
        
        text = "Short text"
        budget = 100
        
        result = estimator.truncate_to_budget(text, budget)
        
        assert isinstance(result, TokenEstimate)
        assert not result.truncated
        assert result.text == text
        assert result.token_count <= budget
    
    def test_truncate_to_budget_with_truncation(self):
        """Test truncation when text exceeds budget."""
        estimator = TokenEstimator()
        
        long_text = "word " * 1000
        budget = 50
        
        result = estimator.truncate_to_budget(long_text, budget)
        
        assert result.truncated
        assert result.text.endswith("...")
        assert result.token_count == budget
        assert len(result.text) < len(long_text)
        assert result.original_length == len(long_text)
    
    def test_truncate_to_budget_zero(self):
        """Test truncation with zero budget."""
        estimator = TokenEstimator()
        
        result = estimator.truncate_to_budget("Some text", 0)
        
        assert result.text == ""
        assert result.token_count == 0
    
    def test_truncate_to_budget_empty_text(self):
        """Test truncation with empty text."""
        estimator = TokenEstimator()
        
        result = estimator.truncate_to_budget("", 100)
        
        assert result.text == ""
        assert result.token_count == 0
        assert not result.truncated
    
    def test_estimate_batch(self):
        """Test batch estimation."""
        estimator = TokenEstimator()
        
        memories = [
            {"id": "1", "content": "First memory"},
            {"id": "2", "content": "Second memory"},
            {"id": "3", "content": "Third memory"}
        ]
        
        estimates = estimator.estimate_batch(memories)
        
        assert len(estimates) == 3
        assert all(isinstance(e, TokenEstimate) for e in estimates)
        assert all(e.token_count > 0 for e in estimates)
    
    def test_format_for_context(self):
        """Test context formatting."""
        estimator = TokenEstimator()
        
        memory = {
            "id": "test",
            "content": "Test content",
            "importance_score": 0.9,
            "event_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "metadata": {"key": "value"}
        }
        
        formatted = estimator._format_for_context(memory)
        
        assert "2024-01-01" in formatted
        assert "Test content" in formatted
        assert "importance: 0.90" in formatted
    
    def test_format_for_context_minimal(self):
        """Test formatting with minimal memory."""
        estimator = TokenEstimator()
        
        memory = {
            "id": "test",
            "content": "Just content"
        }
        
        formatted = estimator._format_for_context(memory)
        
        assert "Just content" in formatted
        assert len(formatted.strip()) > 0
    
    def test_get_available_budget(self):
        """Test available budget calculation."""
        estimator = TokenEstimator()
        
        total = 1000
        prefix = "System prompt: "
        suffix = "\n\nAssistant:"
        
        available = estimator.get_available_budget(total, prefix, suffix)
        
        assert available < total
        assert available > 0
        assert isinstance(available, int)
    
    def test_get_available_budget_no_affix(self):
        """Test budget with no prefix/suffix."""
        estimator = TokenEstimator()
        
        total = 1000
        available = estimator.get_available_budget(total)
        
        assert available == total
    
    def test_get_available_budget_exceeds(self):
        """Test when prefix/suffix exceed budget."""
        estimator = TokenEstimator()
        
        total = 10
        prefix = "Very long prefix " * 50
        
        available = estimator.get_available_budget(total, prefix)
        
        assert available == 0  # Should be clamped to 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_count_tokens_function(self):
        """Test count_tokens convenience function."""
        text = "Hello, world!"
        count = count_tokens(text)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_tokens_custom_model(self):
        """Test count_tokens with custom model."""
        text = "Test text"
        count = count_tokens(text, model="gpt-3.5-turbo")
        
        assert count > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_unicode_text(self):
        """Test with Unicode characters."""
        estimator = TokenEstimator()
        
        text = "Hello 世界 🌍"
        count = estimator.count_tokens(text)
        
        assert count > 0
    
    def test_very_long_text(self):
        """Test with very long text."""
        estimator = TokenEstimator()
        
        # 100K characters
        long_text = "a" * 100000
        count = estimator.count_tokens(long_text)
        
        assert count > 10000
        assert count < 150000
    
    def test_special_characters(self):
        """Test with special characters."""
        estimator = TokenEstimator()
        
        text = "Special: @#$%^&*()[]{}|\\;':\",.<>?/~`"
        count = estimator.count_tokens(text)
        
        assert count > 0
    
    def test_whitespace_handling(self):
        """Test whitespace handling."""
        estimator = TokenEstimator()
        
        text1 = "word word word"
        text2 = "word  word  word"  # Extra spaces
        
        count1 = estimator.count_tokens(text1)
        count2 = estimator.count_tokens(text2)
        
        # Should be similar but not necessarily identical
        assert abs(count1 - count2) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
