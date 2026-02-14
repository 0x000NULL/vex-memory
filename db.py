"""
Database connection management for Vex Memory System.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://vex:vex_memory_dev@localhost:5433/vex_memory"
)


def get_connection(dsn: Optional[str] = None):
    """Get a new database connection."""
    return psycopg2.connect(dsn or DATABASE_URL, cursor_factory=RealDictCursor)


@contextmanager
def get_cursor(dsn: Optional[str] = None):
    """Context manager yielding a cursor with auto-commit on success."""
    conn = get_connection(dsn)
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_health(dsn: Optional[str] = None) -> bool:
    """Check if the database is reachable."""
    try:
        with get_cursor(dsn) as cur:
            cur.execute("SELECT 1")
            return True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")
        return False
