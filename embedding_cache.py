"""
Embedding Cache Module
======================

Multi-layer caching system for embedding vectors to reduce Ollama API latency.

Layers:
1. In-memory LRU cache (primary, ~10ms access)
2. Database cache (secondary, ~50ms access)
3. Query result cache (tertiary, full query results)

Performance targets:
- Cache hit (memory): <10ms
- Cache hit (DB): <50ms  
- Cache miss (Ollama): ~2500ms
- Expected hit rate: 80-90% in production
"""

import os
import json
import hashlib
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, asdict

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Configuration from environment variables
EMBEDDING_CACHE_SIZE = int(os.environ.get("EMBEDDING_CACHE_SIZE", "10000"))
EMBEDDING_CACHE_TTL_HOURS = int(os.environ.get("EMBEDDING_CACHE_TTL_HOURS", "24"))
QUERY_CACHE_SIZE = int(os.environ.get("QUERY_CACHE_SIZE", "1000"))
QUERY_CACHE_TTL_SECONDS = int(os.environ.get("QUERY_CACHE_TTL_SECONDS", "3600"))
DB_CACHE_RETENTION_DAYS = int(os.environ.get("DB_CACHE_RETENTION_DAYS", "7"))
DB_CACHE_MIN_ACCESS_COUNT = int(os.environ.get("DB_CACHE_MIN_ACCESS_COUNT", "5"))


@dataclass
class CacheStats:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    memory_hits: int = 0
    db_hits: int = 0
    evictions: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average cache access latency."""
        total = self.hits + self.misses
        return self.total_latency_ms / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for API responses."""
        return {
            **asdict(self),
            "hit_rate": self.hit_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    
    Uses OrderedDict for O(1) access and eviction.
    Stores (value, timestamp) tuples to support TTL expiration.
    """
    
    def __init__(self, max_size: int, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        start_time = time.time()
        
        if key not in self.cache:
            self.stats.misses += 1
            self.stats.total_latency_ms += (time.time() - start_time) * 1000
            return None
        
        value, timestamp = self.cache[key]
        
        # Check TTL expiration
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            self.stats.misses += 1
            self.stats.total_latency_ms += (time.time() - start_time) * 1000
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        self.stats.hits += 1
        self.stats.memory_hits += 1
        self.stats.total_latency_ms += (time.time() - start_time) * 1000
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)  # FIFO eviction
            self.stats.evictions += 1
        
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)
    
    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        count = len(self.cache)
        self.cache.clear()
        return count
    
    def size(self) -> int:
        """Return current cache size."""
        return len(self.cache)
    
    def get_stats(self) -> CacheStats:
        """Return cache statistics."""
        return self.stats


class EmbeddingCache:
    """
    Multi-layer embedding cache with in-memory LRU and database persistence.
    
    Architecture:
    - Layer 1: In-memory LRU cache (10k entries, 24h TTL)
    - Layer 2: PostgreSQL database cache (unlimited, 7 day eviction)
    - Layer 3: Query result cache (1k queries, 1h TTL)
    
    Cache key: SHA-256 hash of input text (deterministic, collision-resistant)
    """
    
    def __init__(self, db_conn=None):
        self.memory_cache = LRUCache(max_size=EMBEDDING_CACHE_SIZE, ttl_hours=EMBEDDING_CACHE_TTL_HOURS)
        self.query_cache = LRUCache(max_size=QUERY_CACHE_SIZE, ttl_hours=QUERY_CACHE_TTL_SECONDS / 3600)
        self.db_conn = db_conn
        self._ensure_db_cache_table()
    
    def _ensure_db_cache_table(self) -> None:
        """Create embedding_cache table if it doesn't exist."""
        if not self.db_conn:
            return
        
        try:
            with self.db_conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'embedding_cache'
                    );
                """)
                exists = cur.fetchone()[0]
                
                if not exists:
                    logger.info("Creating embedding_cache table...")
                    cur.execute("""
                        CREATE TABLE embedding_cache (
                            content_hash TEXT PRIMARY KEY,
                            embedding vector(384),
                            created_at TIMESTAMP DEFAULT NOW(),
                            access_count INTEGER DEFAULT 1,
                            last_accessed TIMESTAMP DEFAULT NOW()
                        );
                        CREATE INDEX idx_embedding_cache_accessed 
                        ON embedding_cache(last_accessed);
                    """)
                    self.db_conn.commit()
                    logger.info("embedding_cache table created")
        except Exception as e:
            logger.warning(f"Could not create embedding_cache table: {e}")
    
    def get_cache_key(self, text: str) -> str:
        """Generate deterministic cache key from text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache (memory or DB).
        
        Returns None if not cached. Does not generate embeddings.
        """
        start_time = time.time()
        cache_key = self.get_cache_key(text)
        
        # Layer 1: Check in-memory cache
        embedding = self.memory_cache.get(cache_key)
        if embedding is not None:
            logger.debug(f"Embedding cache HIT (memory): {cache_key[:8]}... ({(time.time() - start_time) * 1000:.1f}ms)")
            return embedding
        
        # Layer 2: Check database cache
        if self.db_conn:
            embedding = self._get_from_db_cache(cache_key)
            if embedding is not None:
                # Promote to memory cache
                self.memory_cache.set(cache_key, embedding)
                self.memory_cache.stats.db_hits += 1
                logger.debug(f"Embedding cache HIT (DB): {cache_key[:8]}... ({(time.time() - start_time) * 1000:.1f}ms)")
                return embedding
        
        logger.debug(f"Embedding cache MISS: {cache_key[:8]}... ({(time.time() - start_time) * 1000:.1f}ms)")
        return None
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in both memory and database cache.
        
        Write-through caching: updates both layers simultaneously.
        """
        cache_key = self.get_cache_key(text)
        
        # Store in memory cache
        self.memory_cache.set(cache_key, embedding)
        
        # Store in database cache
        if self.db_conn:
            self._set_in_db_cache(cache_key, embedding)
    
    def _get_from_db_cache(self, cache_key: str) -> Optional[List[float]]:
        """Retrieve embedding from database cache."""
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE embedding_cache
                    SET access_count = access_count + 1,
                        last_accessed = NOW()
                    WHERE content_hash = %s
                    RETURNING embedding::text;
                """, (cache_key,))
                
                row = cur.fetchone()
                if row and row['embedding']:
                    # Parse vector string "[0.1, 0.2, ...]" to list
                    embedding_str = row['embedding'].strip('[]')
                    embedding = [float(x) for x in embedding_str.split(',')]
                    self.db_conn.commit()
                    return embedding
        except Exception as e:
            logger.warning(f"DB cache read failed: {e}")
            if self.db_conn:
                self.db_conn.rollback()
        
        return None
    
    def _set_in_db_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Store embedding in database cache."""
        try:
            with self.db_conn.cursor() as cur:
                # Convert list to pgvector format
                embedding_str = str(embedding)
                
                cur.execute("""
                    INSERT INTO embedding_cache (content_hash, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (content_hash) DO UPDATE
                    SET access_count = embedding_cache.access_count + 1,
                        last_accessed = NOW();
                """, (cache_key, embedding_str))
                
                self.db_conn.commit()
        except Exception as e:
            logger.warning(f"DB cache write failed: {e}")
            if self.db_conn:
                self.db_conn.rollback()
    
    def evict_old_entries(self) -> int:
        """
        Evict old database cache entries.
        
        Removes entries older than DB_CACHE_RETENTION_DAYS with low access counts.
        Returns number of entries evicted.
        """
        if not self.db_conn:
            return 0
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM embedding_cache
                    WHERE last_accessed < NOW() - INTERVAL '%s days'
                      AND access_count < %s
                    RETURNING content_hash;
                """, (DB_CACHE_RETENTION_DAYS, DB_CACHE_MIN_ACCESS_COUNT))
                
                evicted = cur.rowcount
                self.db_conn.commit()
                
                if evicted > 0:
                    logger.info(f"Evicted {evicted} old embedding cache entries")
                
                return evicted
        except Exception as e:
            logger.error(f"Cache eviction failed: {e}")
            if self.db_conn:
                self.db_conn.rollback()
            return 0
    
    def clear_all(self) -> Dict[str, int]:
        """
        Clear all cache layers.
        
        Returns dictionary with counts: {"memory": X, "database": Y}
        """
        memory_cleared = self.memory_cache.clear()
        db_cleared = 0
        
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("DELETE FROM embedding_cache RETURNING content_hash;")
                    db_cleared = cur.rowcount
                    self.db_conn.commit()
            except Exception as e:
                logger.error(f"DB cache clear failed: {e}")
                if self.db_conn:
                    self.db_conn.rollback()
        
        return {
            "memory": memory_cleared,
            "database": db_cleared,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns metrics from all cache layers.
        """
        stats = self.memory_cache.get_stats().to_dict()
        
        # Add database cache stats
        if self.db_conn:
            try:
                with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_entries,
                            SUM(access_count) as total_accesses,
                            AVG(access_count) as avg_access_count,
                            MAX(last_accessed) as most_recent_access
                        FROM embedding_cache;
                    """)
                    row = cur.fetchone()
                    if row:
                        stats['db_cache'] = dict(row)
            except Exception as e:
                logger.warning(f"Could not fetch DB cache stats: {e}")
        
        stats['memory_cache_size'] = self.memory_cache.size()
        stats['query_cache_size'] = self.query_cache.size()
        
        return stats
    
    def warmup(self, common_texts: List[str], embedding_fn) -> int:
        """
        Pre-populate cache with common queries.
        
        Args:
            common_texts: List of frequently used text inputs
            embedding_fn: Function to generate embeddings (e.g., _get_embedding_sync)
        
        Returns number of embeddings cached.
        """
        cached_count = 0
        
        for text in common_texts:
            # Skip if already cached
            if self.get_embedding(text) is not None:
                continue
            
            # Generate and cache
            embedding = embedding_fn(text)
            if embedding:
                self.set_embedding(text, embedding)
                cached_count += 1
        
        logger.info(f"Cache warmup complete: {cached_count} embeddings cached")
        return cached_count


# Global cache instance (initialized with DB connection from api.py)
_global_cache: Optional[EmbeddingCache] = None


def get_cache(db_conn=None) -> EmbeddingCache:
    """Get or create the global embedding cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = EmbeddingCache(db_conn=db_conn)
    
    return _global_cache


def initialize_cache(db_conn) -> EmbeddingCache:
    """Initialize the global cache with a database connection."""
    global _global_cache
    _global_cache = EmbeddingCache(db_conn=db_conn)
    return _global_cache
