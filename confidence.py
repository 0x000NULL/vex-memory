"""
Confidence Scoring for Vex Memory
==================================

Automatic confidence score assignment based on linguistic markers,
memory type, and source metadata. Helps distinguish verified facts
from assumptions and inferences.
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def assign_confidence(content: str, memory_type: str, metadata: Optional[Dict[str, Any]] = None) -> float:
    """
    Assign confidence score based on linguistic markers and memory type.
    
    Args:
        content: The memory content text
        memory_type: Type of memory (episodic, semantic, procedural, emotional)
        metadata: Optional metadata dict that may contain source info
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    content_lower = content.lower()
    
    # Start with base score based on memory type
    if memory_type == "episodic":
        # Events are witnessed/experienced, high confidence
        base = 0.9
    elif memory_type == "semantic":
        # Facts tend to be higher confidence if from good sources
        base = 0.8
    elif memory_type == "procedural":
        # How-tos are usually well-defined
        base = 0.8
    elif memory_type == "emotional":
        # Emotional interpretations are subjective
        base = 0.7
    else:
        base = 0.7
    
    # Linguistic marker adjustments
    score_adjustments = []
    
    # High confidence markers (0.9-1.0)
    high_markers = [
        r'\b(is|are|was|were)\b',           # Definite statements
        r'\b(confirmed|verified)\b',         # Explicit verification
        r'\b(definitely|certainly)\b',       # Strong certainty
        r'\b(always|never)\b',               # Absolute statements
        r'\d{4}-\d{2}-\d{2}',               # Specific dates
        r'#\d+',                             # Issue/ticket numbers
        r'version \d+\.\d+',                # Specific versions
    ]
    
    # Medium confidence markers (0.6-0.8)
    medium_markers = [
        r'\b(probably|likely)\b',
        r'\b(seems|appears)\b',
        r'\b(usually|generally)\b',
        r'\b(typically|normally)\b',
        r'\b(should|would)\b',
        r'\b(believes|thinks)\b',
    ]
    
    # Low confidence markers (0.3-0.5)
    low_markers = [
        r'\b(maybe|possibly|perhaps)\b',
        r'\b(might|could be|may be)\b',
        r'\b(uncertain|unsure)\b',
        r'\b(guess|assume|speculation)\b',
        r'\b(if|unless|assuming)\b',
        r'\?$',                              # Question mark at end
    ]
    
    # Count marker occurrences
    high_count = sum(1 for m in high_markers if re.search(m, content_lower))
    medium_count = sum(1 for m in medium_markers if re.search(m, content_lower))
    low_count = sum(1 for m in low_markers if re.search(m, content_lower))
    
    # Apply adjustments based on marker prevalence
    if high_count > 0:
        score_adjustments.append(0.1)
    if medium_count > 0:
        score_adjustments.append(-0.1)
    if low_count > 0:
        score_adjustments.append(-0.2)
    
    # Multiple low-confidence markers = stronger penalty
    if low_count >= 2:
        score_adjustments.append(-0.1)
    
    # Source-based adjustments
    if metadata:
        source = metadata.get("source", "")
        
        # High-confidence sources
        if source in ["database", "consolidation", "verified", "api", "system"]:
            score_adjustments.append(0.05)
        
        # Lower confidence for auto-extraction
        if source == "auto-extraction":
            score_adjustments.append(-0.1)
        
        # Check for verification metadata
        if metadata.get("verified", False):
            score_adjustments.append(0.1)
        
        # Check importance score correlation
        importance = metadata.get("importance_score", 0.5)
        if importance >= 0.9:
            score_adjustments.append(0.05)
        elif importance <= 0.3:
            score_adjustments.append(-0.05)
    
    # Content quality indicators
    # Longer, more detailed memories tend to be more certain
    if len(content) > 200:
        score_adjustments.append(0.05)
    elif len(content) < 50:
        score_adjustments.append(-0.05)
    
    # Specific details (numbers, names) increase confidence
    if re.search(r'\d+(\.\d+)?', content):
        score_adjustments.append(0.02)
    
    # Proper nouns (capitalized words) suggest specific facts
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', content)
    if len(proper_nouns) >= 3:
        score_adjustments.append(0.03)
    
    # Calculate final score
    final_score = base + sum(score_adjustments)
    
    # Clamp to valid range
    final_score = max(0.0, min(1.0, final_score))
    
    logger.debug(
        f"Confidence scoring: base={base:.2f}, adjustments={score_adjustments}, "
        f"final={final_score:.2f} for content: {content[:60]}..."
    )
    
    return final_score


def bulk_assign_confidence(memories: list) -> Dict[str, float]:
    """
    Assign confidence scores to a batch of memories.
    
    Args:
        memories: List of dicts with 'id', 'content', 'type', and optional 'metadata'
        
    Returns:
        Dict mapping memory_id -> confidence_score
    """
    results = {}
    
    for mem in memories:
        mem_id = mem.get("id")
        content = mem.get("content", "")
        mem_type = mem.get("type", "semantic")
        metadata = mem.get("metadata", {})
        
        if not mem_id or not content:
            logger.warning(f"Skipping invalid memory: {mem}")
            continue
        
        score = assign_confidence(content, mem_type, metadata)
        results[mem_id] = score
    
    logger.info(f"Bulk assigned confidence scores to {len(results)} memories")
    return results


def update_confidence_in_db(memory_id: str, confidence_score: float) -> bool:
    """
    Update confidence_score for a memory in the database.
    
    Args:
        memory_id: UUID of the memory
        confidence_score: New confidence score (0.0-1.0)
        
    Returns:
        True if update succeeded, False otherwise
    """
    try:
        import db
        with db.get_cursor() as cur:
            cur.execute(
                "UPDATE memory_nodes SET confidence_score = %s WHERE id = %s",
                (confidence_score, memory_id),
            )
        return True
    except Exception as e:
        logger.error(f"Failed to update confidence for {memory_id}: {e}")
        return False


def backfill_all_confidence_scores() -> Dict[str, int]:
    """
    Backfill confidence scores for all memories that don't have them.
    
    Returns:
        Dict with 'updated' and 'failed' counts
    """
    try:
        import db
        
        # Fetch memories without confidence scores (NULL or default 0.8)
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id::text, content, type::text, importance_score, source, metadata
                    FROM memory_nodes
                    WHERE confidence_score IS NULL OR confidence_score = 0.8
                    ORDER BY created_at DESC
                """)
                rows = cur.fetchall()
        
        updated = 0
        failed = 0
        
        for row in rows:
            mem_id = row["id"]
            content = row["content"]
            mem_type = row["type"]
            metadata = row.get("metadata", {}) or {}
            
            # Add importance_score to metadata for scoring
            metadata["importance_score"] = row.get("importance_score", 0.5)
            metadata["source"] = row.get("source", "unknown")
            
            score = assign_confidence(content, mem_type, metadata)
            
            if update_confidence_in_db(mem_id, score):
                updated += 1
            else:
                failed += 1
        
        logger.info(f"Backfill complete: updated={updated}, failed={failed}")
        return {"updated": updated, "failed": failed}
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        return {"updated": 0, "failed": 0, "error": str(e)}


if __name__ == "__main__":
    # Test confidence assignment
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    test_cases = [
        ("Ethan's birthday is December 20", "semantic", {}),
        ("Ethan probably prefers dark mode", "semantic", {}),
        ("Maybe the server is on port 8080", "semantic", {}),
        ("I observed the user deploying the API at 3pm", "episodic", {}),
        ("User seemed frustrated with Docker networking", "emotional", {}),
        ("To deploy: run docker-compose up -d", "procedural", {}),
    ]
    
    print("\n=== Confidence Scoring Tests ===\n")
    for content, mem_type, metadata in test_cases:
        score = assign_confidence(content, mem_type, metadata)
        print(f"[{mem_type:12s}] {score:.2f} | {content}")
    
    print("\n")
