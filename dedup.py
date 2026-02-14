"""
Duplicate detection and merging for Vex Memory System.
"""

import logging
from typing import List, Dict, Any, Optional

import httpx

import db

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://host.docker.internal:11434"
EMBED_MODEL = "all-minilm"


def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama synchronously."""
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model": EMBED_MODEL,
                "prompt": text[:8000],
            })
            if r.status_code == 200:
                return r.json().get("embedding")
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
    return None


def find_duplicates(content: str, threshold: float = 0.85) -> List[dict]:
    """Check if content is semantically similar to existing memories.

    Returns list of dicts with keys: id, content, similarity
    """
    embedding = _get_embedding(content)
    if not embedding:
        return []

    try:
        with db.get_cursor() as cur:
            cur.execute(
                """SELECT id::text, content,
                          1 - (embedding <=> %s::vector) as similarity
                   FROM memory_nodes
                   WHERE embedding IS NOT NULL
                     AND 1 - (embedding <=> %s::vector) > %s
                   ORDER BY similarity DESC
                   LIMIT 5""",
                (str(embedding), str(embedding), threshold),
            )
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"find_duplicates query failed: {e}")
        return []


def merge_memories(mem_id_1: str, mem_id_2: str) -> Optional[str]:
    """Merge two similar memories, keeping the higher importance and richer content.

    Returns the surviving memory id, or None on failure.
    The memory with shorter content is deleted; the survivor gets the
    higher importance_score and the longer content.
    """
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """SELECT id::text, content, importance_score, source, metadata,
                          type::text, event_time, created_at
                   FROM memory_nodes WHERE id IN (%s, %s)""",
                (mem_id_1, mem_id_2),
            )
            rows = cur.fetchall()
            if len(rows) < 2:
                logger.warning(f"merge_memories: could not find both {mem_id_1} and {mem_id_2}")
                return None

            m1, m2 = rows[0], rows[1]

            # Pick the richer (longer) content as survivor
            if len(m1["content"]) >= len(m2["content"]):
                survivor, victim = m1, m2
            else:
                survivor, victim = m2, m1

            best_importance = max(m1["importance_score"], m2["importance_score"])

            # Merge metadata
            merged_meta = {**(victim.get("metadata") or {}), **(survivor.get("metadata") or {})}
            merged_meta["merged_from"] = victim["id"]

            # Re-embed the surviving content
            embedding = _get_embedding(survivor["content"])

            import json as _json
            if embedding:
                cur.execute(
                    """UPDATE memory_nodes
                       SET importance_score = %s, metadata = %s::jsonb, embedding = %s::vector
                       WHERE id = %s""",
                    (best_importance, _json.dumps(merged_meta), str(embedding), survivor["id"]),
                )
            else:
                cur.execute(
                    """UPDATE memory_nodes
                       SET importance_score = %s, metadata = %s::jsonb
                       WHERE id = %s""",
                    (best_importance, _json.dumps(merged_meta), survivor["id"]),
                )

            cur.execute("DELETE FROM memory_nodes WHERE id = %s", (victim["id"],))
            logger.info(f"Merged memory {victim['id']} into {survivor['id']}")
            return survivor["id"]
    except Exception as e:
        logger.error(f"merge_memories failed: {e}")
        return None


def deduplicate_all(threshold: float = 0.9) -> int:
    """Bulk dedup: find and merge all memory pairs above threshold.

    Returns the number of merges performed.
    """
    merged_count = 0
    deleted_ids: set = set()

    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT id::text, content FROM memory_nodes WHERE embedding IS NOT NULL ORDER BY created_at"
            )
            all_memories = cur.fetchall()
    except Exception as e:
        logger.error(f"deduplicate_all: failed to fetch memories: {e}")
        return 0

    for mem in all_memories:
        if mem["id"] in deleted_ids:
            continue

        duplicates = find_duplicates(mem["content"], threshold=threshold)
        for dup in duplicates:
            if dup["id"] == mem["id"] or dup["id"] in deleted_ids:
                continue
            # Merge: the current mem vs the duplicate
            result = merge_memories(mem["id"], dup["id"])
            if result:
                # The victim is whichever was deleted
                victim_id = dup["id"] if result == mem["id"] else mem["id"]
                deleted_ids.add(victim_id)
                merged_count += 1
                if victim_id == mem["id"]:
                    break  # current mem was the victim, move on

    logger.info(f"deduplicate_all: merged {merged_count} memory pairs")
    return merged_count
