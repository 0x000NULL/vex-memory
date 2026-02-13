-- Vex Memory System v2.0 - PostgreSQL Schema
-- Author: Vex
-- Date: February 13, 2026
-- Dependencies: PostgreSQL 15+, pgvector extension, Apache AGE extension

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

-- pgvector for embedding storage and similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Apache AGE for graph operations
CREATE EXTENSION IF NOT EXISTS age;

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- CUSTOM TYPES
-- =============================================================================

-- Memory types based on cognitive psychology
CREATE TYPE memory_type AS ENUM (
    'episodic',    -- Specific events: "Ethan said X at time Y"
    'semantic',    -- Facts: "Ethan prefers Python over JavaScript"
    'procedural',  -- How-tos: "To deploy project X, do Y"
    'emotional'    -- Reactions: "Ethan was frustrated with deployment issues"
);

-- Relationship types for memory connections
CREATE TYPE relation_type AS ENUM (
    'temporal',     -- A happened before/after B
    'causal',       -- A caused B
    'similar',      -- A is similar to B
    'contradicts',  -- A contradicts B
    'elaborates',   -- A provides more detail on B
    'references'    -- A mentions B
);

-- Entity types for classification
CREATE TYPE entity_type AS ENUM (
    'person',
    'project', 
    'concept',
    'location',
    'organization',
    'event',
    'technology',
    'other'
);

-- =============================================================================
-- CORE MEMORY TABLES
-- =============================================================================

-- Main memory storage - each row is a discrete memory unit
CREATE TABLE memory_nodes (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type memory_type NOT NULL,
    content TEXT NOT NULL,
    
    -- Vector embedding for semantic search (nomic-embed-text-v1.5 = 768 dimensions)
    embedding vector(768),
    
    -- Temporal information
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    event_time TIMESTAMP WITH TIME ZONE, -- when the remembered event happened (may be different from created_at)
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Importance and decay scoring
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score BETWEEN 0.0 AND 1.0),
    access_count INTEGER DEFAULT 0,
    decay_factor FLOAT DEFAULT 1.0 CHECK (decay_factor BETWEEN 0.0 AND 1.0),
    
    -- Metadata and provenance
    source TEXT, -- 'conversation', 'observation', 'decision', 'consolidation'
    source_file TEXT, -- original file this memory came from (for migration tracking)
    confidence FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    
    -- Context linking
    conversation_id UUID, -- link to conversation/session if applicable
    parent_memory_id UUID REFERENCES memory_nodes(id), -- if this memory elaborates on another
    
    -- JSON metadata for flexible attributes
    metadata JSONB DEFAULT '{}',
    
    -- Audit trail
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Entities - people, places, things mentioned in memories
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    type entity_type NOT NULL,
    canonical_name TEXT, -- standardized/normalized form
    aliases TEXT[] DEFAULT '{}', -- alternative names/spellings
    
    -- Vector representation for entity matching
    embedding vector(768),
    
    -- Frequency and importance tracking
    mention_count INTEGER DEFAULT 1,
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score BETWEEN 0.0 AND 1.0),
    last_mentioned TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    first_mentioned TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Learned attributes about this entity
    attributes JSONB DEFAULT '{}', -- { "role": "developer", "prefers": ["Python", "Linux"] }
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Uniqueness constraint
    UNIQUE(canonical_name, type)
);

-- Link memories to entities they mention
CREATE TABLE memory_entity_mentions (
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    mention_context TEXT, -- the specific text that mentioned this entity
    confidence FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (memory_id, entity_id)
);

-- Relationships between memory nodes
CREATE TABLE memory_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    relation_type relation_type NOT NULL,
    
    -- Relationship strength and confidence
    weight FLOAT DEFAULT 1.0 CHECK (weight BETWEEN 0.0 AND 1.0),
    confidence FLOAT DEFAULT 0.8 CHECK (confidence BETWEEN 0.0 AND 1.0),
    
    -- Optional explanation of the relationship
    description TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT DEFAULT 'system', -- 'system', 'consolidator', 'manual'
    metadata JSONB DEFAULT '{}',
    
    -- Prevent self-references and duplicates
    CHECK (source_id != target_id),
    UNIQUE(source_id, target_id, relation_type)
);

-- =============================================================================
-- CONSOLIDATION AND PROCESSING TABLES
-- =============================================================================

-- Track consolidation runs for debugging and metrics
CREATE TABLE consolidation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'running', -- 'running', 'completed', 'failed'
    target_date DATE, -- which day's memories were processed
    
    -- Metrics
    memories_processed INTEGER DEFAULT 0,
    entities_created INTEGER DEFAULT 0,
    entities_updated INTEGER DEFAULT 0,
    relations_created INTEGER DEFAULT 0,
    conflicts_detected INTEGER DEFAULT 0,
    
    -- Processing details
    config JSONB DEFAULT '{}',
    error_message TEXT,
    logs TEXT
);

-- Store conflicts detected during processing
CREATE TABLE memory_conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_type TEXT NOT NULL, -- 'factual_contradiction', 'temporal_inconsistency', etc.
    memory1_id UUID NOT NULL REFERENCES memory_nodes(id),
    memory2_id UUID NOT NULL REFERENCES memory_nodes(id),
    
    -- Conflict details
    description TEXT,
    severity FLOAT DEFAULT 0.5 CHECK (severity BETWEEN 0.0 AND 1.0),
    resolution_status TEXT DEFAULT 'unresolved', -- 'unresolved', 'resolved', 'ignored'
    resolution_method TEXT,
    
    -- Audit
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by TEXT
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Vector similarity indexes (using IVFFlat for good performance on single node)
CREATE INDEX memory_embedding_idx ON memory_nodes 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX entity_embedding_idx ON entities 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Memory queries
CREATE INDEX memory_type_idx ON memory_nodes (type);
CREATE INDEX memory_event_time_idx ON memory_nodes (event_time);
CREATE INDEX memory_created_at_idx ON memory_nodes (created_at);
CREATE INDEX memory_importance_idx ON memory_nodes (importance_score DESC, decay_factor DESC);
CREATE INDEX memory_source_idx ON memory_nodes (source);

-- Entity queries  
CREATE INDEX entity_type_idx ON entities (type);
CREATE INDEX entity_name_idx ON entities (canonical_name);
CREATE INDEX entity_mentions_idx ON entities (mention_count DESC, importance_score DESC);
CREATE INDEX entity_last_mentioned_idx ON entities (last_mentioned);

-- Relationship queries
CREATE INDEX relation_source_type_idx ON memory_relations (source_id, relation_type);
CREATE INDEX relation_target_type_idx ON memory_relations (target_id, relation_type);
CREATE INDEX relation_weight_idx ON memory_relations (weight DESC);
CREATE INDEX relation_type_idx ON memory_relations (relation_type);

-- Entity mention queries
CREATE INDEX mention_memory_idx ON memory_entity_mentions (memory_id);
CREATE INDEX mention_entity_idx ON memory_entity_mentions (entity_id);

-- Consolidation tracking
CREATE INDEX consolidation_date_idx ON consolidation_runs (target_date);
CREATE INDEX consolidation_status_idx ON consolidation_runs (status, start_time);

-- Conflict resolution
CREATE INDEX conflicts_unresolved_idx ON memory_conflicts (resolution_status) 
WHERE resolution_status = 'unresolved';

-- =============================================================================
-- GRAPH DATABASE INTEGRATION (Apache AGE)
-- =============================================================================

-- Create AGE graph for advanced graph queries
SELECT create_graph('memory_graph');

-- Note: AGE graph nodes and edges will be created via Python code
-- The AGE graph will mirror the relational data for complex graph traversals

-- =============================================================================
-- TRIGGERS FOR MAINTENANCE
-- =============================================================================

-- Update timestamps automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_memory_nodes_updated_at 
    BEFORE UPDATE ON memory_nodes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_entities_updated_at 
    BEFORE UPDATE ON entities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Track access to memories for importance scoring
CREATE OR REPLACE FUNCTION track_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE memory_nodes 
    SET last_accessed = NOW(), access_count = access_count + 1 
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Note: This trigger would be called by application code, not on SELECT
-- to avoid performance overhead

-- =============================================================================
-- UTILITY FUNCTIONS
-- =============================================================================

-- Calculate current relevance score based on decay
CREATE OR REPLACE FUNCTION calculate_current_relevance(
    importance_score FLOAT,
    decay_factor FLOAT,
    days_since_creation INTEGER,
    access_count INTEGER
) RETURNS FLOAT AS $$
BEGIN
    -- Combine importance, decay, and access frequency
    -- Formula: importance × decay × (1 + log(1 + access_count)) × time_penalty
    DECLARE
        access_boost FLOAT := 1.0 + LN(1.0 + access_count) * 0.1;
        time_penalty FLOAT := EXP(-days_since_creation / 180.0); -- 6-month half-life
    BEGIN
        RETURN LEAST(importance_score * decay_factor * access_boost * time_penalty, 1.0);
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Find similar memories by embedding
CREATE OR REPLACE FUNCTION find_similar_memories(
    query_embedding vector(768),
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_count INTEGER DEFAULT 10
) RETURNS TABLE(
    memory_id UUID,
    content TEXT,
    similarity_score FLOAT,
    memory_type memory_type
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.content,
        1.0 - (m.embedding <=> query_embedding) AS similarity,
        m.type
    FROM memory_nodes m
    WHERE m.embedding IS NOT NULL
    AND 1.0 - (m.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY m.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Recent important memories
CREATE VIEW recent_important_memories AS
SELECT 
    m.*,
    calculate_current_relevance(
        m.importance_score, 
        m.decay_factor, 
        EXTRACT(DAYS FROM (NOW() - m.created_at))::INTEGER,
        m.access_count
    ) AS current_relevance
FROM memory_nodes m
WHERE created_at >= NOW() - INTERVAL '30 days'
ORDER BY current_relevance DESC, created_at DESC;

-- Entity mention summary
CREATE VIEW entity_mention_summary AS
SELECT 
    e.*,
    COUNT(mem.memory_id) as total_mentions,
    MAX(m.created_at) as latest_mention_date,
    ARRAY_AGG(DISTINCT m.type) as mentioned_in_types
FROM entities e
LEFT JOIN memory_entity_mentions mem ON e.id = mem.entity_id
LEFT JOIN memory_nodes m ON mem.memory_id = m.id
GROUP BY e.id
ORDER BY e.importance_score DESC, total_mentions DESC;

-- Memory relationship network
CREATE VIEW memory_network AS
SELECT 
    r.id as relation_id,
    r.relation_type,
    r.weight,
    s.id as source_id,
    s.content as source_content,
    s.type as source_type,
    t.id as target_id,
    t.content as target_content,
    t.type as target_type
FROM memory_relations r
JOIN memory_nodes s ON r.source_id = s.id
JOIN memory_nodes t ON r.target_id = t.id
WHERE r.weight > 0.3 -- Only show meaningful relationships
ORDER BY r.weight DESC;

-- =============================================================================
-- INITIAL DATA AND CONFIGURATION
-- =============================================================================

-- Insert configuration settings
CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value JSONB,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO system_config (key, value, description) VALUES 
('memory_system_version', '"2.0"', 'Current memory system version'),
('embedding_model', '"nomic-embed-text-v1.5"', 'Embedding model used for vectors'),
('embedding_dimensions', '768', 'Vector dimensions'),
('default_importance_score', '0.5', 'Default importance for new memories'),
('consolidation_frequency', '"daily"', 'How often to run consolidation'),
('max_memory_age_days', '365', 'Maximum age before aggressive decay'),
('similarity_threshold', '0.7', 'Default threshold for similarity matching');

-- =============================================================================
-- SAMPLE QUERIES FOR TESTING
-- =============================================================================

/*
-- Find memories about a specific entity
SELECT m.* FROM memory_nodes m
JOIN memory_entity_mentions mem ON m.id = mem.memory_id
JOIN entities e ON mem.entity_id = e.id
WHERE e.canonical_name = 'Ethan';

-- Find related memories via graph traversal
WITH RECURSIVE memory_walk AS (
    -- Start from memories containing specific entity
    SELECT m.id, m.content, 0 as depth
    FROM memory_nodes m
    JOIN memory_entity_mentions mem ON m.id = mem.memory_id
    JOIN entities e ON mem.entity_id = e.id
    WHERE e.canonical_name = 'Python'
    
    UNION ALL
    
    -- Follow relationships to connected memories
    SELECT m.id, m.content, mw.depth + 1
    FROM memory_walk mw
    JOIN memory_relations r ON mw.id = r.source_id
    JOIN memory_nodes m ON r.target_id = m.id
    WHERE mw.depth < 3 AND r.weight > 0.5
)
SELECT DISTINCT * FROM memory_walk ORDER BY depth, content;

-- Find contradictory memories
SELECT 
    m1.content as memory1,
    m2.content as memory2,
    mc.conflict_type,
    mc.description
FROM memory_conflicts mc
JOIN memory_nodes m1 ON mc.memory1_id = m1.id
JOIN memory_nodes m2 ON mc.memory2_id = m2.id
WHERE mc.resolution_status = 'unresolved';

-- Get context for conversation (semantic + temporal + entity-based)
WITH recent_context AS (
    SELECT * FROM memory_nodes 
    WHERE created_at >= NOW() - INTERVAL '7 days'
    ORDER BY importance_score DESC 
    LIMIT 10
),
entity_context AS (
    SELECT m.* FROM memory_nodes m
    JOIN memory_entity_mentions mem ON m.id = mem.memory_id
    JOIN entities e ON mem.entity_id = e.id
    WHERE e.canonical_name = ANY(ARRAY['Ethan', 'Python', 'OpenClaw'])
    ORDER BY m.importance_score DESC 
    LIMIT 10
)
SELECT * FROM (
    SELECT *, 'recent' as context_type FROM recent_context
    UNION ALL
    SELECT *, 'entity' as context_type FROM entity_context
) combined
ORDER BY importance_score DESC;
*/

-- =============================================================================
-- SCHEMA COMPLETE
-- =============================================================================

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO vex_memory_user;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO vex_memory_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO vex_memory_user;

COMMENT ON SCHEMA public IS 'Vex Memory System v2.0 - Next-generation AI memory architecture';