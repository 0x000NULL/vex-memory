"""
Cross-session learning/feedback for Vex Memory System.

Tracks which memories are used, ignored, or corrected, and adjusts
importance scores based on observed usefulness over time.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import db

logger = logging.getLogger(__name__)

VALID_FEEDBACK_TYPES = {"used", "ignored", "corrected"}


def ensure_table():
    """Create the memory_feedback table if it doesn't exist."""
    with db.get_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_feedback (
                id SERIAL PRIMARY KEY,
                memory_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_feedback_memory_id
            ON memory_feedback(memory_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_feedback_type
            ON memory_feedback(feedback_type)
        """)


def record_feedback(memory_id: str, feedback_type: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Record feedback for a memory.

    Args:
        memory_id: The memory that was returned
        feedback_type: "used", "ignored", or "corrected"
        session_id: Optional session identifier for cross-session tracking

    Returns:
        Dict with the created feedback record info
    """
    if feedback_type not in VALID_FEEDBACK_TYPES:
        raise ValueError(f"Invalid feedback_type: {feedback_type}. Must be one of {VALID_FEEDBACK_TYPES}")

    with db.get_cursor() as cur:
        cur.execute(
            """INSERT INTO memory_feedback (memory_id, feedback_type, session_id)
               VALUES (%s, %s, %s)
               RETURNING id, memory_id, feedback_type, session_id, created_at""",
            (memory_id, feedback_type, session_id),
        )
        row = cur.fetchone()
    return dict(row)


def calculate_usefulness(memory_id: str) -> float:
    """Calculate usefulness score for a memory.

    Returns ratio of used/(used+ignored), weighted by recency.
    More recent feedback counts more. Returns 0.5 (neutral) if no feedback exists.
    """
    with db.get_cursor() as cur:
        cur.execute(
            """SELECT feedback_type, created_at
               FROM memory_feedback
               WHERE memory_id = %s AND feedback_type IN ('used', 'ignored')
               ORDER BY created_at DESC""",
            (memory_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return 0.5  # neutral — no data

    now = datetime.now(timezone.utc)
    weighted_used = 0.0
    weighted_total = 0.0

    for row in rows:
        created = row["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max((now - created).total_seconds() / 86400, 0.01)
        # Exponential decay: half-life of 14 days
        weight = math.exp(-0.693 * age_days / 14.0)

        weighted_total += weight
        if row["feedback_type"] == "used":
            weighted_used += weight

    if weighted_total == 0:
        return 0.5

    return weighted_used / weighted_total


def apply_feedback_scores() -> Dict[str, Any]:
    """Adjust importance_score for all memories that have feedback.

    - Highly used memories (usefulness > 0.7) get boosted
    - Consistently ignored (usefulness < 0.3) get decayed
    - Corrected memories get a penalty

    Returns summary of adjustments made.
    """
    with db.get_cursor() as cur:
        # Get all memory_ids that have feedback
        cur.execute("SELECT DISTINCT memory_id FROM memory_feedback")
        memory_ids = [row["memory_id"] for row in cur.fetchall()]

    boosted = 0
    decayed = 0
    corrected = 0
    total = len(memory_ids)

    for mid in memory_ids:
        usefulness = calculate_usefulness(mid)

        # Check for corrections
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM memory_feedback WHERE memory_id = %s AND feedback_type = 'corrected'",
                (mid,),
            )
            correction_count = cur.fetchone()["cnt"]

        # Calculate adjustment
        adjustment = 0.0
        if correction_count > 0:
            # Corrected memories lose credibility
            adjustment = -0.1 * min(correction_count, 3)
            corrected += 1
        elif usefulness > 0.7:
            # Boost useful memories (up to +0.15)
            adjustment = 0.05 + 0.1 * (usefulness - 0.7) / 0.3
            boosted += 1
        elif usefulness < 0.3:
            # Decay ignored memories (down to -0.15)
            adjustment = -(0.05 + 0.1 * (0.3 - usefulness) / 0.3)
            decayed += 1

        if adjustment != 0.0:
            try:
                with db.get_cursor() as cur:
                    cur.execute(
                        """UPDATE memory_nodes
                           SET importance_score = GREATEST(0.01, LEAST(1.0, importance_score + %s))
                           WHERE id = %s""",
                        (adjustment, mid),
                    )
            except Exception as e:
                logger.warning(f"Failed to update importance for {mid}: {e}")

    return {
        "total_evaluated": total,
        "boosted": boosted,
        "decayed": decayed,
        "corrected": corrected,
    }


def get_learning_stats() -> Dict[str, Any]:
    """Return overall stats on memory usefulness and feedback patterns."""
    with db.get_cursor() as cur:
        # Total feedback count by type
        cur.execute(
            """SELECT feedback_type, COUNT(*) as count
               FROM memory_feedback
               GROUP BY feedback_type"""
        )
        type_counts = {row["feedback_type"]: row["count"] for row in cur.fetchall()}

        # Unique memories with feedback
        cur.execute("SELECT COUNT(DISTINCT memory_id) as cnt FROM memory_feedback")
        unique_memories = cur.fetchone()["cnt"]

        # Total feedback
        total_feedback = sum(type_counts.values())

        # Average usefulness for memories that have feedback
        cur.execute("SELECT DISTINCT memory_id FROM memory_feedback")
        memory_ids = [row["memory_id"] for row in cur.fetchall()]

    usefulness_scores = []
    for mid in memory_ids:
        usefulness_scores.append(calculate_usefulness(mid))

    avg_usefulness = sum(usefulness_scores) / len(usefulness_scores) if usefulness_scores else 0.5

    # Top 5 most useful and least useful
    scored = sorted(zip(memory_ids, usefulness_scores), key=lambda x: x[1], reverse=True)
    most_useful = [{"memory_id": mid, "usefulness": round(s, 3)} for mid, s in scored[:5]]
    least_useful = [{"memory_id": mid, "usefulness": round(s, 3)} for mid, s in scored[-5:]] if scored else []

    return {
        "total_feedback_records": total_feedback,
        "unique_memories_with_feedback": unique_memories,
        "feedback_by_type": type_counts,
        "average_usefulness": round(avg_usefulness, 3),
        "most_useful": most_useful,
        "least_useful": least_useful,
    }
