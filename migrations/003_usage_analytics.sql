-- Migration 003: Usage Analytics Tables
-- Part of vex-memory v2.0.0 - Adaptive Learning (Phase 3)
--
-- Creates tables for tracking query patterns, performance metrics,
-- and learned weight configurations to enable adaptive optimization.

-- =============================================================================
-- QUERY LOGS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS query_logs (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Query context
    namespace TEXT NOT NULL,  -- Which namespace was queried
    query TEXT NOT NULL,      -- Query text (may be sanitized for privacy)
    
    -- Configuration used
    weights_used JSONB NOT NULL,  -- Weight configuration as JSON
    
    -- Results
    memories_selected JSONB NOT NULL,  -- Array of selected memory IDs
    total_tokens_used INTEGER NOT NULL,
    total_tokens_budget INTEGER NOT NULL,
    memories_retrieved INTEGER NOT NULL,  -- Total candidates retrieved
    memories_dropped INTEGER NOT NULL,    -- Dropped due to budget/diversity
    
    -- Performance metrics
    computation_time_ms FLOAT NOT NULL,
    
    -- Feedback (for future explicit feedback features)
    user_feedback TEXT,  -- Optional user rating or feedback
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_query_logs_namespace ON query_logs(namespace);
CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_logs_namespace_timestamp ON query_logs(namespace, timestamp);

-- Index for cleanup queries (finding old logs)
-- Note: Cannot use NOW() in index predicate, using partial index instead
CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp_cleanup ON query_logs(timestamp);

-- =============================================================================
-- LEARNED WEIGHTS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS learned_weights (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Learned weight configuration
    weights JSONB NOT NULL,  -- Optimized weight values
    
    -- Optimization metadata
    training_queries INTEGER NOT NULL,  -- Number of queries used for training
    objective_score FLOAT NOT NULL,     -- Score achieved with these weights
    optimization_method TEXT NOT NULL,  -- 'grid_search', 'gradient_descent', etc.
    
    -- Performance metrics on validation set
    avg_diversity_score FLOAT,
    avg_token_efficiency FLOAT,
    avg_computation_time_ms FLOAT,
    
    -- Status
    is_active BOOLEAN DEFAULT true,  -- Whether this config is currently in use
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Partial unique index to ensure only one active config per namespace
CREATE UNIQUE INDEX IF NOT EXISTS idx_learned_weights_unique_active 
    ON learned_weights(namespace) 
    WHERE is_active = true;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_learned_weights_namespace ON learned_weights(namespace);
CREATE INDEX IF NOT EXISTS idx_learned_weights_active ON learned_weights(namespace, is_active);
CREATE INDEX IF NOT EXISTS idx_learned_weights_updated ON learned_weights(updated_at);

-- =============================================================================
-- OPTIMIZATION HISTORY TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS optimization_history (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Context
    namespace TEXT NOT NULL,
    optimization_method TEXT NOT NULL,
    
    -- Input configuration
    training_queries INTEGER NOT NULL,
    validation_queries INTEGER NOT NULL,
    search_space JSONB,  -- Parameter ranges searched
    
    -- Results
    best_weights JSONB NOT NULL,
    best_score FLOAT NOT NULL,
    all_scores JSONB,  -- All weight configs tried and their scores
    
    -- Performance
    computation_time_ms FLOAT NOT NULL,
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_optimization_history_namespace ON optimization_history(namespace);
CREATE INDEX IF NOT EXISTS idx_optimization_history_timestamp ON optimization_history(timestamp);

-- =============================================================================
-- COMMENTS (Documentation)
-- =============================================================================

COMMENT ON TABLE query_logs IS 
    'Logs of prioritized context queries for learning and optimization. Privacy-configurable.';

COMMENT ON TABLE learned_weights IS 
    'Optimized weight configurations learned from usage patterns per namespace.';

COMMENT ON TABLE optimization_history IS 
    'History of weight optimization runs for auditing and analysis.';

COMMENT ON COLUMN query_logs.query IS 
    'Query text. May be sanitized/hashed based on SANITIZE_QUERIES config for privacy.';

COMMENT ON COLUMN query_logs.weights_used IS 
    'Weight configuration used: {"similarity": 0.4, "importance": 0.4, "recency": 0.2}';

COMMENT ON COLUMN query_logs.memories_selected IS 
    'Array of memory UUIDs that were selected for the final context window.';

COMMENT ON COLUMN learned_weights.weights IS 
    'Optimized weights: {"similarity": 0.45, "importance": 0.35, "recency": 0.2}';

COMMENT ON COLUMN learned_weights.objective_score IS 
    'Combined score: diversity_score + token_efficiency (higher is better)';
