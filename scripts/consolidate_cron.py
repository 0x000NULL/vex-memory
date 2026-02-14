#!/usr/bin/env python3
"""
Periodic memory consolidation script.
Run via cron, e.g.:
    0 3 * * * cd /path/to/vex-memory && python scripts/consolidate_cron.py

Runs both similarity-based and topic-based consolidation.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consolidator import PgVectorConsolidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [consolidate_cron] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    dsn = os.environ.get(
        "DATABASE_URL",
        "postgresql://vex:vex_memory_dev@localhost:5433/vex_memory",
    )
    consolidator = PgVectorConsolidator(dsn=dsn)

    logger.info("Starting periodic consolidation run")
    start = datetime.now()

    # 1. Similarity-based consolidation
    try:
        related = consolidator.consolidate_related_memories(
            similarity_threshold=0.75, min_cluster_size=3
        )
        logger.info(
            "Related-memory consolidation: %d clusters, %d memories affected",
            related["clusters_created"],
            related["memories_affected"],
        )
    except Exception:
        logger.exception("Related-memory consolidation failed")
        related = {"clusters_created": 0, "memories_affected": 0, "summaries": []}

    # 2. Topic-based consolidation
    try:
        topics = consolidator.consolidate_by_topic()
        logger.info(
            "Topic consolidation: %d topics summarized",
            topics["topics_summarized"],
        )
    except Exception:
        logger.exception("Topic consolidation failed")
        topics = {"topics_summarized": 0, "summaries": []}

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("Consolidation complete in %.1fs", elapsed)

    return {"related": related, "topics": topics, "elapsed_seconds": elapsed}


if __name__ == "__main__":
    result = main()
    import json
    print(json.dumps(result, indent=2, default=str))
