-- Migration: Add confidence_score to memory_nodes
-- Purpose: Distinguish verified facts from assumptions/inferences
-- Date: 2026-02-28

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
    WHEN type = 'episodic' THEN 0.9
    
    -- Semantic memories from verified sources = high confidence
    WHEN type = 'semantic' AND source IN ('database', 'consolidation', 'verified') THEN 0.85
    
    -- Procedural memories (how-tos) = medium-high confidence
    WHEN type = 'procedural' THEN 0.8
    
    -- Emotional memories (subjective) = medium confidence
    WHEN type = 'emotional' THEN 0.7
    
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

-- Add comment for documentation
COMMENT ON COLUMN memory_nodes.confidence_score IS 
'Confidence level in the accuracy/certainty of this memory (0.0-1.0). 
High values (0.9-1.0) indicate verified facts. 
Medium values (0.6-0.8) indicate likely but unverified information. 
Low values (0.3-0.5) indicate assumptions or uncertain inferences.';
