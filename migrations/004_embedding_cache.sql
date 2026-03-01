-- Migration: Embedding Cache Table
-- Version: 004
-- Created: 2026-03-01
-- Description: Add caching table for embedding vectors to reduce Ollama API latency

-- Create embedding cache table
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding vector(384),  -- all-minilm embeddings are 384 dimensions
    created_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Index for eviction queries (find old, rarely accessed entries)
CREATE INDEX IF NOT EXISTS idx_embedding_cache_accessed 
ON embedding_cache(last_accessed);

-- Index for access count filtering
CREATE INDEX IF NOT EXISTS idx_embedding_cache_access_count 
ON embedding_cache(access_count);

-- Optional: Add comment for documentation
COMMENT ON TABLE embedding_cache IS 
'Persistent cache for embedding vectors. Reduces Ollama API calls by storing frequently used embeddings.';

COMMENT ON COLUMN embedding_cache.content_hash IS 
'SHA-256 hash of input text, used as cache key';

COMMENT ON COLUMN embedding_cache.embedding IS 
'Cached embedding vector (384 dimensions for all-minilm model)';

COMMENT ON COLUMN embedding_cache.access_count IS 
'Number of times this embedding has been retrieved from cache';

COMMENT ON COLUMN embedding_cache.last_accessed IS 
'Last time this embedding was accessed (for LRU eviction)';
