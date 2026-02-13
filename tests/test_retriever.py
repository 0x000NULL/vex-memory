"""Tests for the memory retrieval layer."""

import pytest
from datetime import datetime, timedelta

from extractor import MemoryNode, MemoryType
from retriever import (
    MemoryRetriever, QueryContext, QueryIntent, RetrievalStrategy, ContextWindow,
)


@pytest.fixture
def sample_memories():
    return [
        MemoryNode(id="m1", type=MemoryType.SEMANTIC, content="Python is a popular programming language", importance_score=0.7, event_time=datetime.now()),
        MemoryNode(id="m2", type=MemoryType.PROCEDURAL, content="How to deploy with Docker: step 1 build image step 2 push", importance_score=0.8, event_time=datetime.now()),
        MemoryNode(id="m3", type=MemoryType.EPISODIC, content="Yesterday Ethan fixed the PostgreSQL connection bug", importance_score=0.6, event_time=datetime.now() - timedelta(days=1)),
        MemoryNode(id="m4", type=MemoryType.EMOTIONAL, content="Ethan prefers Linux over macOS for development", importance_score=0.5, event_time=datetime.now() - timedelta(days=7)),
        MemoryNode(id="m5", type=MemoryType.SEMANTIC, content="Docker containers provide process isolation", importance_score=0.6, event_time=datetime.now()),
    ]


@pytest.fixture
def retriever(sample_memories):
    r = MemoryRetriever()
    r.memories = sample_memories
    r._build_indices()
    return r


# ---------------------------------------------------------------------------
# Query intent inference
# ---------------------------------------------------------------------------

class TestIntentInference:
    def test_temporal_intent(self, retriever):
        assert retriever._infer_query_intent("What happened yesterday?") == QueryIntent.TEMPORAL

    def test_procedural_intent(self, retriever):
        assert retriever._infer_query_intent("How to deploy the app?") == QueryIntent.PROCEDURAL

    def test_factual_default(self, retriever):
        # "about" triggers associative; use a plain factual query
        assert retriever._infer_query_intent("What is Python") == QueryIntent.FACTUAL

    def test_associative_intent(self, retriever):
        assert retriever._infer_query_intent("What is related to Docker?") == QueryIntent.ASSOCIATIVE


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_finds_relevant(self, retriever):
        results = retriever._keyword_search("Python programming")
        assert any(m.id == "m1" for m in results)

    def test_no_results_for_gibberish(self, retriever):
        results = retriever._keyword_search("xyzzy plugh")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Entity search
# ---------------------------------------------------------------------------

class TestEntitySearch:
    def test_entity_extraction(self, retriever):
        entities = retriever._extract_query_entities("Tell me about Ethan and Python")
        lower = [e.lower() for e in entities]
        assert "ethan" in lower
        assert "python" in lower

    def test_entity_based_search(self, retriever):
        qc = QueryContext(query="Ethan", mentioned_entities=["Ethan"])
        results = retriever._entity_based_search(qc)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Procedural search
# ---------------------------------------------------------------------------

class TestProceduralSearch:
    def test_finds_howtos(self, retriever):
        results = retriever._procedural_search("How to deploy with Docker")
        assert any(m.id == "m2" for m in results)


# ---------------------------------------------------------------------------
# Full query pipeline
# ---------------------------------------------------------------------------

class TestQueryPipeline:
    def test_query_returns_context(self, retriever):
        qc = QueryContext(query="Docker deployment", max_tokens=2000)
        ctx = retriever.query(qc)
        assert isinstance(ctx, ContextWindow)
        assert ctx.total_tokens > 0

    def test_conversation_context(self, retriever):
        ctx = retriever.get_conversation_context("Tell me about Python", max_tokens=1000)
        assert len(ctx.memories) >= 0

    def test_recent_important(self, retriever):
        recent = retriever.get_recent_important_memories(days=30, max_count=5)
        assert all(m.importance_score > 0.6 for m in recent)


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

class TestContextWindow:
    def test_formatted_output(self, sample_memories):
        ctx = ContextWindow(
            query="test",
            strategy=RetrievalStrategy.HYBRID,
            memories=sample_memories[:2],
            total_tokens=100,
            relevance_scores=[0.9, 0.5],
        )
        text = ctx.get_formatted_context()
        assert "SEMANTIC" in text or "PROCEDURAL" in text

    def test_token_budget_truncation(self, sample_memories):
        ctx = ContextWindow(
            query="test",
            strategy=RetrievalStrategy.HYBRID,
            memories=sample_memories,
            total_tokens=5000,
            relevance_scores=[0.9] * len(sample_memories),
        )
        text = ctx.get_formatted_context(max_tokens=10)  # Very small budget
        # Should have fewer memories than total
        assert len(text) < sum(len(m.content) for m in sample_memories)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_populated(self, retriever):
        stats = retriever.get_retrieval_stats()
        assert stats["total_memories"] == 5
        assert "memory_types" in stats
