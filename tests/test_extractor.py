"""Tests for the memory extractor module."""

import os
import json
import tempfile
import pytest
from datetime import datetime, date

from extractor import (
    MemoryExtractor, MemoryNode, Entity, Fact, Relationship, EmotionalMarker,
    MemoryType, EntityType, RelationType,
)


@pytest.fixture
def extractor():
    return MemoryExtractor()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

class TestSectionSplitting:
    def test_splits_by_headers(self, extractor):
        content = "# Header1\nContent1\n\n# Header2\nContent2 with enough text to be kept"
        sections = extractor._split_into_sections(content)
        assert len(sections) >= 2

    def test_splits_long_sections_by_paragraphs(self, extractor):
        content = "A " * 300 + "\n\n" + "B " * 300
        sections = extractor._split_into_sections(content)
        assert len(sections) >= 2

    def test_keeps_nonempty_sections(self, extractor):
        content = "# Header\nThis is a decent section with enough text to be meaningful\n\n# Another\nMore content here that should be kept"
        sections = extractor._split_into_sections(content)
        assert len(sections) >= 1


# ---------------------------------------------------------------------------
# Memory type classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_procedural_detection(self, extractor):
        assert extractor._classify_memory_type("How to deploy the server") == MemoryType.PROCEDURAL
        assert extractor._classify_memory_type("You need to configure the DNS") == MemoryType.PROCEDURAL

    def test_emotional_detection(self, extractor):
        assert extractor._classify_memory_type("I really prefer Python over Java") == MemoryType.EMOTIONAL

    def test_episodic_detection(self, extractor):
        assert extractor._classify_memory_type("Yesterday we discussed the plan") == MemoryType.EPISODIC

    def test_semantic_default(self, extractor):
        assert extractor._classify_memory_type("PostgreSQL supports JSONB columns") == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    def test_known_entities_found(self, extractor):
        entities = extractor._extract_entities("Alice uses Python for most projects")
        names = {e.canonical_name for e in entities}
        assert "alice" in names
        assert "python" in names

    def test_deduplication(self, extractor):
        entities = extractor._extract_entities("Python is great. I love Python.")
        canon_counts = {}
        for e in entities:
            canon_counts.setdefault(e.canonical_name, 0)
            canon_counts[e.canonical_name] += 1
        # each canonical name should appear at most once
        assert all(c == 1 for c in canon_counts.values())

    def test_canonicalization(self, extractor):
        assert extractor._canonicalize_entity_name("PostgreSQL", EntityType.TECHNOLOGY) == "postgresql"
        assert extractor._canonicalize_entity_name("JS", EntityType.TECHNOLOGY) == "javascript"


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

class TestImportance:
    def test_decision_boost(self, extractor):
        base = extractor._calculate_importance("Some normal text about stuff", MemoryType.SEMANTIC)
        boosted = extractor._calculate_importance("A critical decision was made about architecture", MemoryType.SEMANTIC)
        assert boosted > base

    def test_short_content_penalty(self, extractor):
        short = extractor._calculate_importance("hi", MemoryType.SEMANTIC)
        long = extractor._calculate_importance("This is a much longer piece of text with more detail", MemoryType.SEMANTIC)
        assert long > short

    def test_score_bounds(self, extractor):
        score = extractor._calculate_importance("x" * 10, MemoryType.SEMANTIC)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Emotional markers
# ---------------------------------------------------------------------------

class TestEmotionalMarkers:
    def test_positive_preference(self, extractor):
        markers = extractor._extract_emotional_markers("I really love using Vim for editing")
        assert any(m.polarity == "positive" for m in markers)

    def test_negative_preference(self, extractor):
        markers = extractor._extract_emotional_markers("I hate dealing with YAML configs")
        assert any(m.polarity == "negative" for m in markers)


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------

class TestFileExtraction:
    def test_extract_from_daily_log(self, extractor):
        content = "# Morning\nWe discussed the deployment plan with Alice.\n\n# Afternoon\nFixed a critical bug in the Python service that was causing memory leaks."
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, prefix="2026-01-15") as f:
            f.write(content)
            path = f.name
        try:
            # Rename to match daily log pattern
            daily_path = os.path.join(os.path.dirname(path), "2026-01-15.md")
            os.rename(path, daily_path)
            memories = extractor.extract_from_file(daily_path)
            assert len(memories) >= 1
            # Daily logs should produce episodic memories
            assert any(m.type == MemoryType.EPISODIC for m in memories)
        finally:
            if os.path.exists(daily_path):
                os.unlink(daily_path)

    def test_extract_from_long_term(self, extractor):
        content = "# Preferences\nAlice prefers Python over JavaScript for backend work.\n\n# Projects\nOpenClaw is an AI orchestration system."
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, prefix="MEMORY") as f:
            f.write(content)
            path = f.name
        try:
            memories = extractor.extract_from_file(path)
            assert len(memories) >= 1
        finally:
            os.unlink(path)

    def test_save_and_load_json(self, extractor):
        memories = [
            MemoryNode(
                id="test1", type=MemoryType.SEMANTIC,
                content="Test memory", importance_score=0.7,
                entities=[Entity(name="Python", type=EntityType.TECHNOLOGY, canonical_name="python", aliases=["py"])],
            )
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            extractor.save_to_json(memories, path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["content"] == "Test memory"
        finally:
            os.unlink(path)

    def test_extraction_summary(self, extractor):
        memories = [
            MemoryNode(id="1", type=MemoryType.SEMANTIC, content="Fact A", importance_score=0.6),
            MemoryNode(id="2", type=MemoryType.EPISODIC, content="Event B", importance_score=0.4),
        ]
        summary = extractor.get_extraction_summary(memories)
        assert summary["total_memories"] == 2
        assert "semantic" in summary["memory_types"]
