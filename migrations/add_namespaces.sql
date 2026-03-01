-- Migration: Add namespace support for multi-agent memory sharing
-- Date: 2026-02-28
-- Purpose: Enable sub-agents to access Vex's memory context without cold starts

-- =============================================================================
-- 1. Create namespaces table
-- =============================================================================

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

-- Trigger to update timestamp (only create if the trigger function exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column') THEN
        CREATE TRIGGER update_namespaces_updated_at 
            BEFORE UPDATE ON memory_namespaces 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- =============================================================================
-- 2. Add namespace_id column to memory_nodes
-- =============================================================================

ALTER TABLE memory_nodes 
ADD COLUMN IF NOT EXISTS namespace_id UUID REFERENCES memory_namespaces(namespace_id);

-- Create index for namespace-filtered queries
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memory_nodes(namespace_id);

-- =============================================================================
-- 3. Create default namespace and backfill existing memories
-- =============================================================================

-- Insert vex-main namespace if it doesn't exist
INSERT INTO memory_namespaces (name, owner_agent, access_policy)
VALUES ('vex-main', 'vex', '{"read": ["vex"], "write": ["vex"]}'::jsonb)
ON CONFLICT (name) DO NOTHING;

-- Backfill existing memories to vex-main namespace
UPDATE memory_nodes
SET namespace_id = (SELECT namespace_id FROM memory_namespaces WHERE name = 'vex-main')
WHERE namespace_id IS NULL;

-- =============================================================================
-- 4. Create helper functions for access control
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

-- =============================================================================
-- 5. Create view for accessible memories per agent
-- =============================================================================

-- View to get memories accessible to an agent (pass agent_id as parameter)
-- Usage: SELECT * FROM get_agent_memories('agent-123', 'namespace-uuid')
CREATE OR REPLACE FUNCTION get_agent_memories(
    p_agent_id TEXT,
    p_namespace_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE (
    id UUID,
    type memory_type,
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
        m.type,
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
-- 6. Add comments for documentation
-- =============================================================================

COMMENT ON TABLE memory_namespaces IS 'Namespaces for multi-agent memory sharing';
COMMENT ON COLUMN memory_namespaces.access_policy IS 'JSONB with read/write arrays of agent IDs';
COMMENT ON FUNCTION can_read_namespace IS 'Check if agent can read from namespace';
COMMENT ON FUNCTION can_write_namespace IS 'Check if agent can write to namespace';
COMMENT ON FUNCTION get_agent_memories IS 'Get all memories an agent can access';

-- =============================================================================
-- Migration complete
-- =============================================================================
