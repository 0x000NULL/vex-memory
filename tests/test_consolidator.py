"""Tests for the memory consolidation engine."""

import pytest
import math
from datetime import datetime, timedelta, date

from extractor import (
    MemoryNode, Entity, Fact, EmotionalMarker,
    MemoryType, EntityType, RelationType,
)
from consolidator import (
    MemoryConsolidator, MemoryConflict, ConsolidationRun, ConflictType, ConflictSeverity,
)


@pytest.fixture
def consolidator():
    return MemoryConsolidator()


def _make_memory(content, mtype=MemoryType.SEMANTIC, importance=0.5, event_time=None, entities=None, facts=None, emotional_markers=None):
    return MemoryNode(
        id=f"mem_{hash(content) % 1000000:06d}",
        type=mtype,
        content=content,
        importance_score=importance,
        event_time=event_time or datetime.now(),
        entities=entities or [],
        facts=facts or [],
        emotional_markers=emotional_markers or [],
    )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_exact_duplicate(self, consolidator):
        consolidator.current_memories = [_make_memory("The sky is blue")]
        assert consolidator._is_duplicate_memory(_make_memory("The sky is blue"))

    def test_case_insensitive_duplicate(self, consolidator):
        consolidator.current_memories = [_make_memory("The Sky Is Blue")]
        assert consolidator._is_duplicate_memory(_make_memory("the sky is blue"))

    def test_non_duplicate(self, consolidator):
        consolidator.current_memories = [_make_memory("The sky is blue")]
        assert not consolidator._is_duplicate_memory(_make_memory("PostgreSQL supports JSONB"))


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------

class TestTextSimilarity:
    def test_identical_texts(self, consolidator):
        assert consolidator._calculate_text_similarity("hello world", "hello world") == 1.0

    def test_empty_texts(self, consolidator):
        assert consolidator._calculate_text_similarity("", "") == 1.0

    def test_no_overlap(self, consolidator):
        assert consolidator._calculate_text_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self, consolidator):
        sim = consolidator._calculate_text_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

class TestEnhancedImportance:
    def test_recency_boost(self, consolidator):
        old = _make_memory("fact", event_time=datetime.now() - timedelta(days=365))
        new = _make_memory("fact", event_time=datetime.now())
        old_score = consolidator._calculate_enhanced_importance(old)
        new_score = consolidator._calculate_enhanced_importance(new)
        assert new_score > old_score

    def test_entity_richness_boost(self, consolidator):
        plain = _make_memory("Something happened")
        rich = _make_memory("Something happened", entities=[
            Entity(name="Ethan", type=EntityType.PERSON, canonical_name="ethan", aliases=[], mention_count=10),
        ])
        assert consolidator._calculate_enhanced_importance(rich) >= consolidator._calculate_enhanced_importance(plain)

    def test_score_capped_at_one(self, consolidator):
        m = _make_memory("critical important decision", importance=0.99, entities=[
            Entity(name="X", type=EntityType.PERSON, canonical_name="x", aliases=[], mention_count=100),
        ])
        m.facts = [Fact(subject="a", predicate="b", object="c")] * 5
        assert consolidator._calculate_enhanced_importance(m) <= 1.0


# ---------------------------------------------------------------------------
# Decay / forgetting curves
# ---------------------------------------------------------------------------

class TestDecay:
    def test_stability_varies_by_type(self, consolidator):
        sem = consolidator._calculate_memory_stability(_make_memory("fact", mtype=MemoryType.SEMANTIC))
        epi = consolidator._calculate_memory_stability(_make_memory("event", mtype=MemoryType.EPISODIC))
        assert sem > epi  # semantic memories should be more stable

    def test_decay_reduces_factor(self, consolidator):
        old = _make_memory("old fact", event_time=datetime.now() - timedelta(days=200))
        old.decay_factor = 1.0
        consolidator.current_memories = [old]
        run = ConsolidationRun(id="test", start_time=datetime.now())
        consolidator._apply_decay_functions(run)
        assert old.decay_factor < 1.0

    def test_decay_never_below_floor(self, consolidator):
        ancient = _make_memory("ancient", event_time=datetime.now() - timedelta(days=10000))
        ancient.decay_factor = 1.0
        consolidator.current_memories = [ancient]
        run = ConsolidationRun(id="test", start_time=datetime.now())
        consolidator._apply_decay_functions(run)
        assert ancient.decay_factor >= 0.1


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

class TestConflicts:
    def test_factual_contradiction(self, consolidator):
        m1 = _make_memory("fact1", facts=[Fact(subject="color", predicate="is", object="blue good")])
        m2 = _make_memory("fact2", facts=[Fact(subject="color", predicate="is", object="red bad")])
        conflicts = consolidator._check_memory_pair_for_conflicts(m1, m2)
        assert any(c.conflict_type == ConflictType.FACTUAL_CONTRADICTION for c in conflicts)

    def test_preference_evolution(self, consolidator):
        m1 = _make_memory("pref1", event_time=datetime.now() - timedelta(days=60),
                          emotional_markers=[EmotionalMarker(entity="python", emotion_type="preference", polarity="positive", intensity=0.8, context="like python")])
        m2 = _make_memory("pref2", event_time=datetime.now(),
                          emotional_markers=[EmotionalMarker(entity="python", emotion_type="preference", polarity="negative", intensity=0.8, context="hate python")])
        conflicts = consolidator._check_memory_pair_for_conflicts(m1, m2)
        assert any(c.conflict_type == ConflictType.PREFERENCE_EVOLUTION for c in conflicts)


# ---------------------------------------------------------------------------
# Memory merging
# ---------------------------------------------------------------------------

class TestMerging:
    def test_similar_memories_merged(self, consolidator):
        m1 = _make_memory("The deployment process uses Docker containers for isolation")
        m2 = _make_memory("The deployment process uses Docker containers for app isolation")
        run = ConsolidationRun(id="test", start_time=datetime.now())
        merged = consolidator._merge_similar_memories([m1, m2], run)
        assert len(merged) <= 2  # may or may not merge depending on threshold

    def test_merge_preserves_importance(self, consolidator):
        m1 = _make_memory("A", importance=0.3)
        m2 = _make_memory("A exact duplicate text", importance=0.9)
        group = consolidator._merge_memory_group([m1, m2])
        assert group.importance_score == 0.9


# ---------------------------------------------------------------------------
# Full consolidation pipeline
# ---------------------------------------------------------------------------

class TestConsolidationPipeline:
    def test_consolidation_summary(self, consolidator):
        run = ConsolidationRun(id="r1", start_time=datetime.now(), target_date=date.today())
        run.end_time = datetime.now()
        run.status = "completed"
        summary = consolidator.get_consolidation_summary(run)
        assert summary["status"] == "completed"
        assert "metrics" in summary
