"""
Usage Analytics Module for Vex Memory System
============================================

Tracks query patterns, memory selection, and performance metrics to enable
adaptive learning and weight optimization.

Privacy-first design:
- Optional: Can be disabled via USAGE_LOGGING_ENABLED env var
- Configurable retention period (default: 90 days)
- No PII stored - query content can be hashed/sanitized
- Transparent: Users can view/export their analytics
"""

import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import db

logger = logging.getLogger(__name__)

# Configuration from environment
USAGE_LOGGING_ENABLED = os.environ.get("USAGE_LOGGING_ENABLED", "true").lower() == "true"
USAGE_LOG_RETENTION_DAYS = int(os.environ.get("USAGE_LOG_RETENTION_DAYS", "90"))
SANITIZE_QUERIES = os.environ.get("SANITIZE_QUERIES", "false").lower() == "true"


class AnalyticsEventType(str, Enum):
    """Types of analytics events we track."""
    QUERY = "query"
    OPTIMIZATION = "optimization"
    WEIGHT_UPDATE = "weight_update"


def is_enabled() -> bool:
    """Check if usage logging is enabled."""
    return USAGE_LOGGING_ENABLED


def sanitize_query(query: str) -> str:
    """
    Optionally sanitize query text for privacy.
    
    When SANITIZE_QUERIES is enabled, returns a hash instead of the actual query.
    This protects user privacy while still allowing query pattern analysis.
    """
    if not SANITIZE_QUERIES:
        return query
    
    import hashlib
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"<sanitized:{query_hash}>"


def log_query(
    namespace: str,
    query: str,
    weights_used: Dict[str, float],
    memories_selected: List[str],
    total_tokens_used: int,
    total_tokens_budget: int,
    memories_retrieved: int,
    memories_dropped: int,
    computation_time_ms: float,
    user_feedback: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Log a query and its results for learning and optimization.
    
    Args:
        namespace: Namespace the query was executed in
        query: Original query text (will be sanitized if configured)
        weights_used: Dict of weight name -> value used for scoring
        memories_selected: List of memory IDs that were selected
        total_tokens_used: Total tokens in final context
        total_tokens_budget: Maximum tokens allowed
        memories_retrieved: Total memories retrieved from DB
        memories_dropped: Memories dropped due to budget/diversity
        computation_time_ms: Time taken to compute prioritization
        user_feedback: Optional user feedback on result quality
        metadata: Additional metadata to store
    
    Returns:
        Query log ID (UUID as string) if logged, None if logging disabled
    """
    if not USAGE_LOGGING_ENABLED:
        logger.debug("Usage logging disabled, skipping query log")
        return None
    
    try:
        log_id = str(uuid.uuid4())
        sanitized_query = sanitize_query(query)
        
        with db.get_cursor() as cur:
            cur.execute("""
                INSERT INTO query_logs (
                    id,
                    timestamp,
                    namespace,
                    query,
                    weights_used,
                    memories_selected,
                    total_tokens_used,
                    total_tokens_budget,
                    memories_retrieved,
                    memories_dropped,
                    computation_time_ms,
                    user_feedback,
                    metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                log_id,
                datetime.utcnow(),
                namespace,
                sanitized_query,
                json.dumps(weights_used),
                json.dumps(memories_selected),
                total_tokens_used,
                total_tokens_budget,
                memories_retrieved,
                memories_dropped,
                computation_time_ms,
                user_feedback,
                json.dumps(metadata or {})
            ))
        
        logger.debug(f"Logged query {log_id} for namespace {namespace}")
        return log_id
        
    except Exception as e:
        # Don't fail the main query if logging fails
        logger.error(f"Failed to log query analytics: {e}", exc_info=True)
        return None


def get_namespace_analytics(
    namespace: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Retrieve analytics for a specific namespace.
    
    Args:
        namespace: Namespace to query
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Max results to return
    
    Returns:
        List of query log dictionaries
    """
    if not USAGE_LOGGING_ENABLED:
        return []
    
    try:
        with db.get_cursor() as cur:
            sql = "SELECT * FROM query_logs WHERE namespace = %s"
            params = [namespace]
            
            if start_date:
                sql += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                sql += " AND timestamp <= %s"
                params.append(end_date)
            
            sql += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(sql, params)
            rows = cur.fetchall()
            
            return [dict(row) for row in rows]
        
    except Exception as e:
        logger.error(f"Failed to retrieve namespace analytics: {e}")
        return []


def get_analytics_summary(namespace: str) -> Dict[str, Any]:
    """
    Get summary statistics for a namespace.
    
    Returns:
        Dict with summary metrics (query count, avg tokens, etc.)
    """
    if not USAGE_LOGGING_ENABLED:
        return {"enabled": False}
    
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(total_tokens_used) as avg_tokens_used,
                    AVG(total_tokens_budget) as avg_tokens_budget,
                    AVG(total_tokens_used::float / NULLIF(total_tokens_budget, 0)) as avg_token_efficiency,
                    AVG(memories_retrieved) as avg_memories_retrieved,
                    AVG(memories_dropped) as avg_memories_dropped,
                    AVG(computation_time_ms) as avg_computation_time_ms,
                    MIN(timestamp) as first_query,
                    MAX(timestamp) as last_query
                FROM query_logs
                WHERE namespace = %s
            """, (namespace,))
            
            row = cur.fetchone()
            
            return {
                "enabled": True,
                "namespace": namespace,
                "total_queries": row["total_queries"] or 0,
                "avg_tokens_used": float(row["avg_tokens_used"] or 0),
                "avg_tokens_budget": float(row["avg_tokens_budget"] or 0),
                "avg_token_efficiency": float(row["avg_token_efficiency"] or 0),
                "avg_memories_retrieved": float(row["avg_memories_retrieved"] or 0),
                "avg_memories_dropped": float(row["avg_memories_dropped"] or 0),
                "avg_computation_time_ms": float(row["avg_computation_time_ms"] or 0),
                "first_query": row["first_query"],
                "last_query": row["last_query"]
            }
        
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        return {"enabled": True, "error": str(e)}


def get_top_queries(namespace: str, limit: int = 10) -> List[Tuple[str, int]]:
    """
    Get most frequent queries in a namespace.
    
    Returns:
        List of (query, count) tuples
    """
    if not USAGE_LOGGING_ENABLED:
        return []
    
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                SELECT query, COUNT(*) as count
                FROM query_logs
                WHERE namespace = %s
                GROUP BY query
                ORDER BY count DESC
                LIMIT %s
            """, (namespace, limit))
            
            return [(row["query"], row["count"]) for row in cur.fetchall()]
        
    except Exception as e:
        logger.error(f"Failed to get top queries: {e}")
        return []


def get_weight_usage_stats(namespace: str) -> Dict[str, Any]:
    """
    Analyze which weight configurations are being used.
    
    Returns:
        Statistics on weight usage patterns
    """
    if not USAGE_LOGGING_ENABLED:
        return {}
    
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                SELECT weights_used, COUNT(*) as usage_count
                FROM query_logs
                WHERE namespace = %s
                GROUP BY weights_used
                ORDER BY usage_count DESC
            """, (namespace,))
            
            weight_configs = []
            for row in cur.fetchall():
                weight_configs.append({
                    "weights": json.loads(row["weights_used"]),
                    "usage_count": row["usage_count"]
                })
            
            return {
                "namespace": namespace,
                "unique_weight_configs": len(weight_configs),
                "weight_configs": weight_configs
            }
        
    except Exception as e:
        logger.error(f"Failed to get weight usage stats: {e}")
        return {}


def cleanup_old_logs(retention_days: Optional[int] = None) -> int:
    """
    Delete query logs older than retention period.
    
    Args:
        retention_days: Days to retain (defaults to USAGE_LOG_RETENTION_DAYS)
    
    Returns:
        Number of logs deleted
    """
    if not USAGE_LOGGING_ENABLED:
        return 0
    
    days = retention_days or USAGE_LOG_RETENTION_DAYS
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                DELETE FROM query_logs
                WHERE timestamp < %s
            """, (cutoff_date,))
            
            deleted_count = cur.rowcount
            logger.info(f"Cleaned up {deleted_count} query logs older than {days} days")
            return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")
        return 0


def export_namespace_data(namespace: str, format: str = "json") -> str:
    """
    Export all analytics data for a namespace (for user data portability).
    
    Args:
        namespace: Namespace to export
        format: Export format ('json' or 'csv')
    
    Returns:
        Serialized data as string
    """
    if not USAGE_LOGGING_ENABLED:
        return json.dumps({"error": "Usage logging is disabled"})
    
    try:
        logs = get_namespace_analytics(namespace, limit=100000)  # Get all
        
        if format == "json":
            return json.dumps({
                "namespace": namespace,
                "exported_at": datetime.utcnow().isoformat(),
                "total_queries": len(logs),
                "logs": logs
            }, indent=2, default=str)
        
        elif format == "csv":
            # Simple CSV export
            import csv
            import io
            
            output = io.StringIO()
            if logs:
                writer = csv.DictWriter(output, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            
            return output.getvalue()
        
        else:
            return json.dumps({"error": f"Unsupported format: {format}"})
    
    except Exception as e:
        logger.error(f"Failed to export namespace data: {e}")
        return json.dumps({"error": str(e)})


def delete_namespace_data(namespace: str) -> int:
    """
    Delete all analytics data for a namespace (GDPR compliance).
    
    Args:
        namespace: Namespace to delete
    
    Returns:
        Number of logs deleted
    """
    if not USAGE_LOGGING_ENABLED:
        return 0
    
    try:
        with db.get_cursor() as cur:
            cur.execute("DELETE FROM query_logs WHERE namespace = %s", (namespace,))
            deleted_count = cur.rowcount
            logger.info(f"Deleted {deleted_count} query logs for namespace {namespace}")
            return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to delete namespace data: {e}")
        return 0
