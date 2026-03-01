-- =============================================================================
-- Vex Memory v0.3.0 Migration
-- =============================================================================
-- Date: 2026-02-28
-- Description: Multi-agent namespaces + confidence scoring
-- 
-- This migration combines:
-- - add_namespaces.sql: Multi-agent memory sharing with access control
-- - add_confidence.sql: Confidence scores for fact verification
--
-- Safe to run multiple times (idempotent)
-- =============================================================================

-- Ensure uuid extension is available
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- PART 1: MULTI-AGENT NAMESPACES
-- =============================================================================

-- Create namespaces table
CREATE TABLE IF NOT EXISTS memory_namespaces (
    namespace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    owner_agent TEXT,
    access_policy JSONB DEFAULT '{"read": [], "write": []}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for owner lookups
CREATE INDEX IF NOT EXISTS idx_namespaces_owner ON memory_namespaces(owner_agent);

-- Trigger to update timestamp (only if update_updated_at_column function exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column') THEN
        DROP TRIGGER IF EXISTS update_namespaces_updated_at ON memory_namespaces;
        CREATE TRIGGER update_namespaces_updated_at 
            BEFORE UPDATE ON memory_namespaces 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END$$;

-- Add namespace_id column to memory_nodes
ALTER TABLE memory_nodes 
ADD COLUMN IF NOT EXISTS namespace_id UUID REFERENCES memory_namespaces(namespace_id);

-- Create index for namespace-filtered queries
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memory_nodes(namespace_id);

-- Insert vex-main namespace if it doesn't exist
INSERT INTO memory_namespaces (name, owner_agent, access_policy)
VALUES ('vex-main', 'vex', '{"read": ["vex"], "write": ["vex"]}'::jsonb)
ON CONFLICT (name) DO NOTHING;

-- Backfill existing memories to vex-main namespace
UPDATE memory_nodes
SET namespace_id = (SELECT namespace_id FROM memory_namespaces WHERE name = 'vex-main')
WHERE namespace_id IS NULL;

-- =============================================================================
-- Access Control Functions
-- =============================================================================

-- Check if an agent has read access to a namespace
CREATE OR REPLACE FUNCTION can_read_namespace(
    p_agent_id TEXT,
    p_namespace_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    v_owner TEXT;
    v_policy JSONB;
BEGIN
    -- Get namespace info
    SELECT owner_agent, access_policy 
    INTO v_owner, v_policy
    FROM memory_namespaces
    WHERE namespace_id = p_namespace_id;
    
    -- If namespace doesn't exist, deny
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    
    -- Owner always has access
    IF v_owner = p_agent_id THEN
        RETURN TRUE;
    END IF;
    
    -- Check read permission in access_policy
    RETURN v_policy->'read' @> to_jsonb(p_agent_id::text);
END;
$$ LANGUAGE plpgsql STABLE;

-- Check if an agent has write access to a namespace
CREATE OR REPLACE FUNCTION can_write_namespace(
    p_agent_id TEXT,
    p_namespace_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    v_owner TEXT;
    v_policy JSONB;
BEGIN
    -- Get namespace info
    SELECT owner_agent, access_policy 
    INTO v_owner, v_policy
    FROM memory_namespaces
    WHERE namespace_id = p_namespace_id;
    
    -- If namespace doesn't exist, deny
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    
    -- Owner always has access
    IF v_owner = p_agent_id THEN
        RETURN TRUE;
    END IF;
    
    -- Check write permission in access_policy
    RETURN v_policy->'write' @> to_jsonb(p_agent_id::text);
END;
$$ LANGUAGE plpgsql STABLE;

-- Get all memories accessible to an agent
CREATE OR REPLACE FUNCTION get_agent_memories(
    p_agent_id TEXT,
    p_namespace_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE (
    id UUID,
    type TEXT,
    content TEXT,
    importance_score FLOAT,
    namespace_id UUID,
    namespace_name TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.type::text,
        m.content,
        m.importance_score,
        m.namespace_id,
        n.name as namespace_name,
        m.created_at
    FROM memory_nodes m
    LEFT JOIN memory_namespaces n ON m.namespace_id = n.namespace_id
    WHERE 
        (p_namespace_id IS NULL OR m.namespace_id = p_namespace_id)
        AND can_read_namespace(p_agent_id, m.namespace_id)
    ORDER BY m.importance_score DESC, m.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- PART 2: CONFIDENCE SCORING
-- =============================================================================

-- Add confidence_score column with default 0.8 (moderate confidence)
ALTER TABLE memory_nodes 
ADD COLUMN IF NOT EXISTS confidence_score REAL DEFAULT 0.8 
CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0);

-- Create index for efficient filtering by confidence
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memory_nodes(confidence_score);

-- Backfill existing memories with confidence scores based on heuristics
UPDATE memory_nodes
SET confidence_score = CASE
    -- Episodic memories (witnessed events) = high confidence
    WHEN type::text = 'episodic' THEN 0.9
    
    -- Semantic memories from verified sources = high confidence
    WHEN type::text = 'semantic' AND source IN ('database', 'consolidation', 'verified') THEN 0.85
    
    -- Procedural memories (how-tos) = medium-high confidence
    WHEN type::text = 'procedural' THEN 0.8
    
    -- Emotional memories (subjective) = medium confidence
    WHEN type::text = 'emotional' THEN 0.7
    
    -- Content-based adjustments
    WHEN LOWER(content) ~ '(confirmed|verified|definitely|is|are|was)' THEN 0.9
    WHEN LOWER(content) ~ '(probably|likely|seems|appears)' THEN 0.7
    WHEN LOWER(content) ~ '(maybe|possibly|might|could be|uncertain)' THEN 0.5
    
    -- High importance usually means verified facts
    WHEN importance_score >= 0.8 THEN 0.85
    
    -- Default: moderate confidence
    ELSE 0.8
END
WHERE confidence_score IS NULL OR confidence_score = 0.8;

-- =============================================================================
-- Documentation
-- =============================================================================

COMMENT ON TABLE memory_namespaces IS 'Namespaces for multi-agent memory sharing with access control';
COMMENT ON COLUMN memory_namespaces.access_policy IS 'JSONB with read/write arrays of agent IDs';
COMMENT ON FUNCTION can_read_namespace IS 'Check if agent can read from namespace (returns boolean)';
COMMENT ON FUNCTION can_write_namespace IS 'Check if agent can write to namespace (returns boolean)';
COMMENT ON FUNCTION get_agent_memories IS 'Get all memories an agent can access (respects namespace permissions)';

COMMENT ON COLUMN memory_nodes.confidence_score IS 
'Confidence level in the accuracy/certainty of this memory (0.0-1.0). 
High values (0.9-1.0) indicate verified facts. 
Medium values (0.6-0.8) indicate likely but unverified information. 
Low values (0.3-0.5) indicate assumptions or uncertain inferences.';

COMMENT ON COLUMN memory_nodes.namespace_id IS 'Namespace this memory belongs to (for multi-agent access control)';

-- =============================================================================
-- Migration Complete
-- =============================================================================

-- Verify tables exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM memory_namespaces) >= 1, 
           'Expected at least 1 namespace (vex-main) after migration';
    RAISE NOTICE 'v0.3.0 migration completed successfully';
END$$;
