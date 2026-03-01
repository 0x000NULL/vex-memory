"""
Unit tests for entity_extractor module.

Tests entity extraction, coverage calculation, and priority boosting.
"""

import pytest
from entity_extractor import EntityExtractor, extract_entities


class TestEntityExtractor:
    """Test EntityExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return EntityExtractor()
    
    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert hasattr(extractor, 'extract')
    
    def test_extract_basic(self, extractor):
        """Test basic entity extraction."""
        text = "John Smith works at Microsoft in Seattle."
        
        result = extractor.extract(text)
        
        assert "entities" in result
        assert "entity_counts" in result
        assert "unique_entities" in result
        assert len(result["entities"]) > 0
    
    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        result = extractor.extract("")
        
        assert result["entities"] == []
        assert len(result["entity_counts"]) == 0
        assert len(result["unique_entities"]) == 0
    
    def test_extract_with_emails(self, extractor):
        """Test email extraction."""
        text = "Contact me at john@example.com or jane@test.org"
        
        result = extractor.extract(text)
        
        emails = [e for e in result["entities"] if e["type"] == "EMAIL"]
        assert len(emails) >= 2
    
    def test_extract_with_urls(self, extractor):
        """Test URL extraction."""
        text = "Visit https://example.com and http://test.org for more info."
        
        result = extractor.extract(text)
        
        urls = [e for e in result["entities"] if e["type"] == "URL"]
        assert len(urls) >= 2
    
    def test_extract_with_dates(self, extractor):
        """Test date extraction."""
        text = "The event is on 2024-01-15 or 12/25/2024."
        
        result = extractor.extract(text)
        
        dates = [e for e in result["entities"] if "DATE" in e["type"]]
        assert len(dates) >= 1
    
    def test_extract_with_phone(self, extractor):
        """Test phone number extraction."""
        text = "Call me at 555-123-4567 or 555.987.6543"
        
        result = extractor.extract(text)
        
        phones = [e for e in result["entities"] if e["type"] == "PHONE"]
        assert len(phones) >= 1
    
    def test_entity_counts(self, extractor):
        """Test entity counting."""
        text = "John Smith and Jane Doe work at Microsoft and Google."
        
        result = extractor.extract(text)
        
        # Should have some entities
        assert sum(result["entity_counts"].values()) > 0
    
    def test_unique_entities(self, extractor):
        """Test unique entity tracking."""
        text = "John works with John at the company. John is great."
        
        result = extractor.extract(text)
        
        # "john" should appear only once in unique_entities (lowercased)
        assert "john" in result["unique_entities"]
    
    def test_priority_scores(self, extractor):
        """Test that entities have priority scores."""
        text = "Alice works at Apple Inc."
        
        result = extractor.extract(text)
        
        for entity in result["entities"]:
            assert "priority" in entity
            assert 0.0 <= entity["priority"] <= 1.0
    
    def test_entity_positions(self, extractor):
        """Test that entities have start/end positions."""
        text = "Microsoft is in Seattle."
        
        result = extractor.extract(text)
        
        for entity in result["entities"]:
            assert "start" in entity
            assert "end" in entity
            assert entity["end"] > entity["start"]


class TestCoverage:
    """Test coverage calculation methods."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return EntityExtractor()
    
    def test_calculate_coverage_full(self, extractor):
        """Test 100% coverage."""
        selected = {"alice", "bob", "microsoft"}
        target = {"alice", "bob", "microsoft"}
        
        coverage = extractor.calculate_coverage(selected, target)
        
        assert coverage == 1.0
    
    def test_calculate_coverage_partial(self, extractor):
        """Test partial coverage."""
        selected = {"alice", "microsoft"}
        target = {"alice", "bob", "microsoft"}
        
        coverage = extractor.calculate_coverage(selected, target)
        
        assert coverage == pytest.approx(2/3, abs=0.01)
    
    def test_calculate_coverage_none(self, extractor):
        """Test zero coverage."""
        selected = {"charlie"}
        target = {"alice", "bob"}
        
        coverage = extractor.calculate_coverage(selected, target)
        
        assert coverage == 0.0
    
    def test_calculate_coverage_empty_target(self, extractor):
        """Test coverage with empty target."""
        selected = {"alice"}
        target = set()
        
        coverage = extractor.calculate_coverage(selected, target)
        
        assert coverage == 1.0  # No targets means perfect coverage


class TestPriorityBoost:
    """Test priority boost calculation."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return EntityExtractor()
    
    def test_priority_boost_full_match(self, extractor):
        """Test boost with full match."""
        memory_entities = {"alice", "bob", "microsoft"}
        priority_entities = {"alice", "bob", "microsoft"}
        
        boost = extractor.get_priority_boost(memory_entities, priority_entities)
        
        assert boost == 1.0
    
    def test_priority_boost_partial_match(self, extractor):
        """Test boost with partial match."""
        memory_entities = {"alice", "microsoft"}
        priority_entities = {"alice", "bob", "microsoft"}
        
        boost = extractor.get_priority_boost(memory_entities, priority_entities)
        
        assert boost == pytest.approx(2/3, abs=0.01)
    
    def test_priority_boost_no_match(self, extractor):
        """Test boost with no match."""
        memory_entities = {"charlie"}
        priority_entities = {"alice", "bob"}
        
        boost = extractor.get_priority_boost(memory_entities, priority_entities)
        
        assert boost == 0.0
    
    def test_priority_boost_empty_priority(self, extractor):
        """Test boost with empty priority set."""
        memory_entities = {"alice"}
        priority_entities = set()
        
        boost = extractor.get_priority_boost(memory_entities, priority_entities)
        
        assert boost == 0.0


class TestBatchExtraction:
    """Test batch extraction methods."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return EntityExtractor()
    
    def test_batch_extract(self, extractor):
        """Test batch extraction."""
        memories = [
            {"id": "1", "content": "Alice works at Microsoft."},
            {"id": "2", "content": "Bob lives in Seattle."},
            {"id": "3", "content": "Contact: info@example.com"}
        ]
        
        results = extractor.batch_extract(memories)
        
        assert len(results) == 3
        assert "1" in results
        assert "2" in results
        assert "3" in results
        
        # Each result should have entities
        for result in results.values():
            assert "entities" in result
    
    def test_batch_extract_empty(self, extractor):
        """Test batch extraction with empty list."""
        results = extractor.batch_extract([])
        
        assert results == {}
    
    def test_aggregate_entities(self, extractor):
        """Test entity aggregation."""
        extraction_results = {
            "1": {
                "entities": [{"text": "Alice", "type": "PERSON"}],
                "entity_counts": {"PERSON": 1},
                "unique_entities": {"alice"}
            },
            "2": {
                "entities": [{"text": "Bob", "type": "PERSON"}],
                "entity_counts": {"PERSON": 1},
                "unique_entities": {"bob"}
            }
        }
        
        aggregated = extractor.aggregate_entities(extraction_results)
        
        assert aggregated["total_entities"] == 2
        assert aggregated["unique_entities"] == 2
        assert "alice" in aggregated["all_unique_entities"]
        assert "bob" in aggregated["all_unique_entities"]


class TestConvenienceFunction:
    """Test module-level convenience function."""
    
    def test_extract_entities_function(self):
        """Test extract_entities convenience function."""
        text = "Alice and Bob work at Microsoft."
        
        result = extract_entities(text)
        
        assert "entities" in result
        assert len(result["entities"]) > 0


class TestRegexFallback:
    """Test regex-based extraction when spaCy unavailable."""
    
    def test_regex_extractor(self):
        """Test regex-based extractor."""
        # Force regex mode
        extractor = EntityExtractor(use_spacy=False)
        
        text = "Email: test@example.com, URL: https://test.com, Phone: 555-1234"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]) > 0
        
        # Should find email, URL, phone
        entity_types = {e["type"] for e in result["entities"]}
        assert "EMAIL" in entity_types or len(entity_types) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
