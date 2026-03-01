"""
Test suite for embedding cache module.

Tests cover:
1. In-memory LRU cache functionality
2. Database cache persistence
3. Cache hit/miss tracking
4. TTL expiration
5. Eviction logic
6. Cache statistics
7. Integration with API endpoints
"""

import pytest
import time
import hashlib
from unittest.mock import Mock, MagicMock, patch
import psycopg2

from embedding_cache import (
    LRUCache, 
    EmbeddingCache, 
    CacheStats,
    get_cache,
    initialize_cache
)


class TestLRUCache:
    """Test in-memory LRU cache."""
    
    def test_basic_set_get(self):
        """Test basic cache set and get operations."""
        cache = LRUCache(max_size=10, ttl_hours=1)
        
        cache.set("key1", [0.1, 0.2, 0.3])
        result = cache.get("key1")
        
        assert result == [0.1, 0.2, 0.3]
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=10, ttl_hours=1)
        
        result = cache.get("nonexistent")
        
        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3, ttl_hours=1)
        
        cache.set("key1", [1])
        cache.set("key2", [2])
        cache.set("key3", [3])
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key4, should evict key2 (least recently used)
        cache.set("key4", [4])
        
        assert cache.get("key1") == [1]
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == [3]
        assert cache.get("key4") == [4]
        assert cache.stats.evictions == 1
    
    def test_ttl_expiration(self):
        """Test TTL expiration of cached entries."""
        cache = LRUCache(max_size=10, ttl_hours=0.0001)  # Very short TTL
        
        cache.set("key1", [0.1])
        
        # Immediately should work
        assert cache.get("key1") == [0.1]
        
        # Wait for expiration
        time.sleep(0.5)
        
        # Should be expired now
        result = cache.get("key1")
        assert result is None
    
    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = LRUCache(max_size=10, ttl_hours=1)
        
        cache.set("key1", [1])
        cache.set("key2", [2])
        
        cleared = cache.clear()
        
        assert cleared == 2
        assert cache.size() == 0
        assert cache.get("key1") is None
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = LRUCache(max_size=10, ttl_hours=1)
        
        cache.set("key1", [1])
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit
        
        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == 2/3
        assert stats.memory_hits == 2


class TestEmbeddingCache:
    """Test multi-layer embedding cache."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock table exists check
        mock_cursor.fetchone.return_value = [True]
        
        return mock_conn
    
    def test_cache_key_generation(self, mock_db):
        """Test deterministic cache key generation."""
        cache = EmbeddingCache(db_conn=mock_db)
        
        text = "test query"
        key1 = cache.get_cache_key(text)
        key2 = cache.get_cache_key(text)
        
        # Same text should produce same key
        assert key1 == key2
        
        # Different text should produce different key
        key3 = cache.get_cache_key("different query")
        assert key1 != key3
        
        # Verify it's SHA-256
        expected = hashlib.sha256(text.encode('utf-8')).hexdigest()
        assert key1 == expected
    
    def test_memory_cache_hit(self, mock_db):
        """Test cache hit from memory layer."""
        cache = EmbeddingCache(db_conn=mock_db)
        
        text = "test query"
        embedding = [0.1, 0.2, 0.3]
        
        cache.set_embedding(text, embedding)
        result = cache.get_embedding(text)
        
        assert result == embedding
        assert cache.memory_cache.stats.memory_hits == 1
    
    def test_memory_cache_miss_no_db(self):
        """Test cache miss when no database connection."""
        cache = EmbeddingCache(db_conn=None)
        
        result = cache.get_embedding("nonexistent query")
        
        assert result is None
        assert cache.memory_cache.stats.misses == 1
    
    def test_db_cache_promotion(self, mock_db):
        """Test promotion from DB cache to memory cache."""
        cache = EmbeddingCache(db_conn=mock_db)
        
        # Mock DB returning a cached embedding
        mock_cursor = mock_db.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = {
            'embedding': '[0.1, 0.2, 0.3]'
        }
        
        result = cache.get_embedding("test query")
        
        # Should have fetched from DB and promoted to memory
        assert result == [0.1, 0.2, 0.3]
        assert cache.memory_cache.stats.db_hits == 1
        
        # Second access should hit memory cache
        result2 = cache.get_embedding("test query")
        assert result2 == [0.1, 0.2, 0.3]
        assert cache.memory_cache.stats.memory_hits == 1
    
    def test_write_through_caching(self, mock_db):
        """Test write-through to both cache layers."""
        cache = EmbeddingCache(db_conn=mock_db)
        mock_cursor = mock_db.cursor.return_value.__enter__.return_value
        
        text = "test query"
        embedding = [0.1, 0.2, 0.3]
        
        cache.set_embedding(text, embedding)
        
        # Should be in memory cache
        assert cache.memory_cache.get(cache.get_cache_key(text)) == embedding
        
        # Should have attempted DB write
        assert mock_cursor.execute.called
    
    def test_cache_eviction(self, mock_db):
        """Test database cache eviction of old entries."""
        cache = EmbeddingCache(db_conn=mock_db)
        mock_cursor = mock_db.cursor.return_value.__enter__.return_value
        mock_cursor.rowcount = 5
        
        evicted = cache.evict_old_entries()
        
        assert evicted == 5
        assert mock_cursor.execute.called
    
    def test_clear_all_caches(self, mock_db):
        """Test clearing all cache layers."""
        cache = EmbeddingCache(db_conn=mock_db)
        mock_cursor = mock_db.cursor.return_value.__enter__.return_value
        mock_cursor.rowcount = 10
        
        # Add some entries to memory cache
        cache.memory_cache.set("key1", [1])
        cache.memory_cache.set("key2", [2])
        
        result = cache.clear_all()
        
        assert result["memory"] == 2
        assert result["database"] == 10
        assert cache.memory_cache.size() == 0
    
    def test_cache_stats(self, mock_db):
        """Test comprehensive cache statistics."""
        cache = EmbeddingCache(db_conn=mock_db)
        mock_cursor = mock_db.cursor.return_value.__enter__.return_value
        
        # Mock DB stats query
        mock_cursor.fetchone.return_value = {
            'total_entries': 1000,
            'total_accesses': 5000,
            'avg_access_count': 5.0,
            'most_recent_access': '2026-03-01 10:00:00'
        }
        
        stats = cache.get_stats()
        
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'memory_cache_size' in stats
        assert 'db_cache' in stats
        assert stats['db_cache']['total_entries'] == 1000
    
    def test_cache_warmup(self, mock_db):
        """Test cache warmup with common queries."""
        cache = EmbeddingCache(db_conn=mock_db)
        
        # Mock embedding function
        def mock_embedding_fn(text):
            return [0.1] * 384
        
        common_texts = ["query1", "query2", "query3"]
        cached_count = cache.warmup(common_texts, mock_embedding_fn)
        
        assert cached_count == 3
        
        # Verify embeddings are cached
        for text in common_texts:
            result = cache.get_embedding(text)
            assert result == [0.1] * 384
    
    def test_warmup_skips_cached(self, mock_db):
        """Test warmup skips already cached entries."""
        cache = EmbeddingCache(db_conn=mock_db)
        
        def mock_embedding_fn(text):
            return [0.1] * 384
        
        # Pre-cache one entry
        cache.set_embedding("query1", [0.5] * 384)
        
        common_texts = ["query1", "query2"]
        cached_count = cache.warmup(common_texts, mock_embedding_fn)
        
        # Should only cache query2
        assert cached_count == 1
        
        # query1 should still have original embedding
        result = cache.get_embedding("query1")
        assert result == [0.5] * 384


class TestCacheStats:
    """Test cache statistics data class."""
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8
    
    def test_hit_rate_zero_division(self):
        """Test hit rate with no requests."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0
    
    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        stats = CacheStats(hits=50, misses=50, total_latency_ms=1000)
        assert stats.avg_latency_ms == 10.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(hits=100, misses=50, evictions=10)
        result = stats.to_dict()
        
        assert result['hits'] == 100
        assert result['misses'] == 50
        assert result['evictions'] == 10
        assert result['hit_rate'] == 100/150
        assert 'avg_latency_ms' in result


class TestCacheIntegration:
    """Integration tests for cache with API."""
    
    def test_global_cache_singleton(self):
        """Test global cache instance management."""
        with patch('embedding_cache._global_cache', None):
            cache1 = get_cache()
            cache2 = get_cache()
            
            # Should return same instance
            assert cache1 is cache2
    
    def test_initialize_cache_with_db(self, mock_db):
        """Test cache initialization with database connection."""
        cache = initialize_cache(mock_db)
        
        assert cache is not None
        assert cache.db_conn is mock_db


@pytest.mark.integration
class TestRealDatabaseCache:
    """Integration tests with real database (requires PostgreSQL)."""
    
    @pytest.fixture
    def db_conn(self):
        """Create real database connection for testing."""
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                dbname=os.getenv("DB_NAME", "vex_test"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres")
            )
            yield conn
            conn.close()
        except Exception as e:
            pytest.skip(f"Database not available: {e}")
    
    def test_db_cache_roundtrip(self, db_conn):
        """Test full roundtrip: write to DB, read back."""
        cache = EmbeddingCache(db_conn=db_conn)
        
        text = f"integration test {time.time()}"
        embedding = [0.1] * 384
        
        # Write
        cache.set_embedding(text, embedding)
        
        # Clear memory cache to force DB read
        cache.memory_cache.clear()
        
        # Read back from DB
        result = cache.get_embedding(text)
        
        assert result is not None
        assert len(result) == 384
        assert result[0] == pytest.approx(0.1, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
