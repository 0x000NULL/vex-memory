"""
Memory decay and access tracking for the Vex Memory System.

Implements exponential decay with recency boost and access-pattern weighting.
Score formula: importance * decay_factor * recency_boost
"""

import math
import logging
from datetime import datetime, timezone

import db

logger = logging.getLogger(__name__)

# Half-life in days for exponential decay
HALF_LIFE_DAYS = 30.0
DECAY_LAMBDA = math.log(2) / HALF_LIFE_DAYS


def compute_recency_boost(last_accessed: datetime) -> float:
    """Exponential decay based on time since last access. Half-life ~30 days."""
    now = datetime.now(timezone.utc)
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    age_days = (now - last_accessed).total_seconds() / 86400.0
    return math.exp(-DECAY_LAMBDA * max(age_days, 0))


def compute_decay_factor(created_at: datetime, access_count: int, last_accessed: datetime) -> float:
    """
    Compute decay_factor from age, access count, and recency.

    decay_factor = time_decay * access_boost, clamped to [0, 1].
    - time_decay: exponential decay from creation date
    - access_boost: 1 + log(1 + access_count) * 0.1
    """
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400.0
    time_decay = math.exp(-DECAY_LAMBDA * max(age_days, 0))
    access_boost = 1.0 + math.log(1.0 + access_count) * 0.1
    return min(time_decay * access_boost, 1.0)


def compute_score(importance: float, decay_factor: float, last_accessed: datetime) -> float:
    """Final relevance score: importance * decay_factor * recency_boost."""
    recency = compute_recency_boost(last_accessed)
    return importance * decay_factor * recency


def update_access(memory_id: str) -> None:
    """Increment access_count and update last_accessed timestamp."""
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """UPDATE memory_nodes
                   SET access_count = access_count + 1,
                       last_accessed = NOW()
                   WHERE id = %s""",
                (memory_id,),
            )
    except Exception as e:
        logger.warning(f"Failed to update access for {memory_id}: {e}")


def apply_decay(memory_id: str) -> None:
    """Recalculate decay_factor for a single memory based on age and access patterns."""
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT created_at, access_count, last_accessed FROM memory_nodes WHERE id = %s",
                (memory_id,),
            )
            row = cur.fetchone()
            if not row:
                return
            new_factor = compute_decay_factor(
                row["created_at"], row["access_count"], row["last_accessed"]
            )
            cur.execute(
                "UPDATE memory_nodes SET decay_factor = %s WHERE id = %s",
                (new_factor, memory_id),
            )
    except Exception as e:
        logger.warning(f"Failed to apply decay for {memory_id}: {e}")


def bulk_decay_update() -> dict:
    """Recalculate decay_factor for all memories. Returns stats dict."""
    updated = 0
    failed = 0
    try:
        with db.get_cursor() as cur:
            cur.execute("SELECT id::text, created_at, access_count, last_accessed FROM memory_nodes")
            rows = cur.fetchall()

        for row in rows:
            try:
                new_factor = compute_decay_factor(
                    row["created_at"], row["access_count"], row["last_accessed"]
                )
                with db.get_cursor() as cur:
                    cur.execute(
                        "UPDATE memory_nodes SET decay_factor = %s WHERE id = %s",
                        (new_factor, row["id"]),
                    )
                updated += 1
            except Exception as e:
                logger.warning(f"Decay update failed for {row['id']}: {e}")
                failed += 1
    except Exception as e:
        logger.error(f"Bulk decay update failed: {e}")

    logger.info(f"Bulk decay update: {updated} updated, {failed} failed")
    return {"updated": updated, "failed": failed, "total": updated + failed}
