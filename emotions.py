"""
Emotional Context Tagging for Vex Memory System
================================================

Keyword/pattern-based sentiment analysis and emotion tagging for memories.
Stores emotion data in the metadata JSONB column.
"""

import re
import logging
from typing import Dict, Any, List, Optional

import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS = {"great", "awesome", "shipped", "complete", "success", "fixed", "working", "proud"}
NEGATIVE_KEYWORDS = {"broken", "failed", "frustrated", "blocked", "regression", "bug", "error"}
EXCITEMENT_KEYWORDS = {"amazing", "finally", "breakthrough"}
CONCERN_KEYWORDS = {"worried", "risk", "careful", "danger", "security"}

# Emotion-specific patterns (keyword sets + extra heuristics)
EMOTION_PATTERNS: Dict[str, Dict[str, Any]] = {
    "joy": {"keywords": {"great", "awesome", "wonderful", "happy", "love", "fantastic"}, "weight": 1.0},
    "pride": {"keywords": {"proud", "shipped", "launched", "achieved", "accomplished", "built"}, "weight": 1.0},
    "frustration": {"keywords": {"frustrated", "broken", "failed", "blocked", "regression", "annoying", "stuck"}, "weight": 1.0},
    "excitement": {"keywords": {"amazing", "finally", "breakthrough", "incredible", "wow", "exciting"}, "weight": 1.0},
    "concern": {"keywords": {"worried", "risk", "careful", "danger", "security", "vulnerable"}, "weight": 1.0},
    "relief": {"keywords": {"relief", "relieved", "finally", "fixed", "resolved", "phew"}, "weight": 1.0},
    "curiosity": {"keywords": {"wonder", "curious", "interesting", "how", "why", "explore", "investigate"}, "weight": 1.0},
    "satisfaction": {"keywords": {"complete", "success", "working", "done", "solid", "clean", "smooth"}, "weight": 1.0},
}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text using keyword/pattern matching.

    Returns:
        {"valence": float (-1 to 1), "arousal": float (0 to 1), "dominant_emotion": str}
    """
    lower = text.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    # Count keyword hits
    pos_hits = len(words & POSITIVE_KEYWORDS)
    neg_hits = len(words & NEGATIVE_KEYWORDS)
    exc_hits = len(words & EXCITEMENT_KEYWORDS) + lower.count("!")
    con_hits = len(words & CONCERN_KEYWORDS)

    # Valence: -1 to 1
    total = pos_hits + neg_hits + exc_hits + con_hits
    if total == 0:
        valence = 0.0
    else:
        valence = ((pos_hits + exc_hits * 0.5) - (neg_hits + con_hits * 0.5)) / total
        valence = max(-1.0, min(1.0, valence))

    # Arousal: 0 to 1 (excitement and strong emotions raise arousal)
    arousal = min(1.0, (exc_hits * 0.3 + abs(neg_hits) * 0.2 + pos_hits * 0.1 + con_hits * 0.15))

    # Score each emotion
    emotion_scores: Dict[str, float] = {}
    for emotion, config in EMOTION_PATTERNS.items():
        hits = len(words & config["keywords"])
        if emotion == "excitement":
            hits += min(3, lower.count("!"))
        emotion_scores[emotion] = hits * config["weight"]

    # Dominant emotion
    dominant = max(emotion_scores, key=emotion_scores.get)  # type: ignore
    if emotion_scores[dominant] == 0:
        dominant = "neutral"

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "dominant_emotion": dominant,
        "emotion_scores": {k: v for k, v in emotion_scores.items() if v > 0},
    }


def tag_memory_emotion(memory_id: str) -> Optional[Dict[str, Any]]:
    """Analyze a memory's content and update its metadata with emotional tags.

    Returns the emotion data on success, None on failure.
    """
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT content, metadata FROM memory_nodes WHERE id = %s",
                (memory_id,),
            )
            row = cur.fetchone()
            if not row:
                logger.warning(f"Memory {memory_id} not found")
                return None

            content = row["content"]
            metadata = row["metadata"] or {}

            emotion_data = analyze_sentiment(content)
            metadata["emotion"] = emotion_data

            import json
            cur.execute(
                "UPDATE memory_nodes SET metadata = %s::jsonb WHERE id = %s",
                (json.dumps(metadata), memory_id),
            )
            return emotion_data
    except Exception as e:
        logger.error(f"Failed to tag emotion for {memory_id}: {e}")
        return None


def bulk_tag_emotions() -> Dict[str, int]:
    """Process all untagged memories (those without emotion in metadata).

    Returns {"tagged": int, "failed": int, "skipped": int}.
    """
    tagged = 0
    failed = 0
    skipped = 0

    try:
        with db.get_cursor() as cur:
            # Find memories without emotion tags
            cur.execute(
                """SELECT id::text, content, metadata FROM memory_nodes
                   WHERE metadata IS NULL
                      OR NOT (metadata ? 'emotion')
                   ORDER BY created_at DESC"""
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to query untagged memories: {e}")
        return {"tagged": 0, "failed": 0, "skipped": 0}

    import json
    for row in rows:
        try:
            content = row["content"]
            if not content or not content.strip():
                skipped += 1
                continue

            metadata = row["metadata"] or {}
            emotion_data = analyze_sentiment(content)
            metadata["emotion"] = emotion_data

            with db.get_cursor() as cur:
                cur.execute(
                    "UPDATE memory_nodes SET metadata = %s::jsonb WHERE id = %s",
                    (json.dumps(metadata), row["id"]),
                )
            tagged += 1
        except Exception as e:
            logger.error(f"Failed to tag {row['id']}: {e}")
            failed += 1

    return {"tagged": tagged, "failed": failed, "skipped": skipped}


def get_emotional_memories(emotion: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return memories tagged with a specific emotion.

    Args:
        emotion: The dominant emotion to filter by.
        limit: Max results.

    Returns:
        List of memory dicts.
    """
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """SELECT id::text, type::text, content, importance_score, decay_factor,
                          access_count, source, event_time, created_at, metadata
                   FROM memory_nodes
                   WHERE metadata->'emotion'->>'dominant_emotion' = %s
                   ORDER BY importance_score DESC
                   LIMIT %s""",
                (emotion, limit),
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to get emotional memories for '{emotion}': {e}")
        return []
