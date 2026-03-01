"""
Weight Optimization Engine for Vex Memory System
================================================

Analyzes historical query data to learn optimal weight configurations
per namespace. Uses grid search to maximize diversity and token efficiency.

Algorithm:
1. Fetch historical query logs for namespace
2. Split into training and validation sets
3. Grid search over weight combinations
4. Evaluate each combination on validation set
5. Select weights that maximize objective function
6. Store learned weights in database

Objective Function: diversity_score + token_efficiency
- diversity_score = average Jaccard distance between selected memories
- token_efficiency = avg(tokens_used / tokens_budget)
"""

import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from itertools import product

import db
from usage_analytics import get_namespace_analytics
from prioritizer import MemoryPrioritizer, ScoringWeights
from token_estimator import TokenEstimator

logger = logging.getLogger(__name__)

# Minimum queries required before optimization
MIN_QUERIES_FOR_OPTIMIZATION = int(os.environ.get("MIN_OPTIMIZATION_QUERIES", "50"))

# Default search space for grid search
DEFAULT_SEARCH_SPACE = {
    "similarity": [0.2, 0.3, 0.4, 0.5, 0.6],
    "importance": [0.2, 0.3, 0.4, 0.5, 0.6],
    "recency": [0.1, 0.2, 0.3, 0.4]
}


def calculate_diversity_score(memory_ids: List[str], memory_contents: Dict[str, str]) -> float:
    """
    Calculate diversity score using Jaccard distance between memories.
    
    Args:
        memory_ids: List of selected memory IDs
        memory_contents: Dict mapping memory ID to content text
    
    Returns:
        Average Jaccard distance (0-1, higher = more diverse)
    """
    if len(memory_ids) < 2:
        return 0.0
    
    def jaccard_distance(text1: str, text2: str) -> float:
        """Calculate Jaccard distance between two texts (word-level)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0.0
        return 1.0 - similarity
    
    # Calculate pairwise Jaccard distances
    distances = []
    for i, id1 in enumerate(memory_ids):
        for id2 in memory_ids[i+1:]:
            if id1 in memory_contents and id2 in memory_contents:
                distance = jaccard_distance(
                    memory_contents[id1],
                    memory_contents[id2]
                )
                distances.append(distance)
    
    return sum(distances) / len(distances) if distances else 0.0


def evaluate_weights(
    weights: Dict[str, float],
    validation_logs: List[Dict[str, Any]],
    memory_contents: Dict[str, str]
) -> float:
    """
    Evaluate a weight configuration on validation data.
    
    Args:
        weights: Weight configuration to evaluate
        validation_logs: List of query logs to evaluate on
        memory_contents: Dict mapping memory ID to content
    
    Returns:
        Combined score: diversity + token_efficiency
    """
    if not validation_logs:
        return 0.0
    
    diversity_scores = []
    token_efficiencies = []
    
    for log in validation_logs:
        # Parse log data
        memories_selected = json.loads(log["memories_selected"]) if isinstance(log["memories_selected"], str) else log["memories_selected"]
        tokens_used = log["total_tokens_used"]
        tokens_budget = log["total_tokens_budget"]
        
        # Calculate diversity
        diversity = calculate_diversity_score(memories_selected, memory_contents)
        diversity_scores.append(diversity)
        
        # Calculate token efficiency
        efficiency = tokens_used / tokens_budget if tokens_budget > 0 else 0.0
        token_efficiencies.append(efficiency)
    
    # Combined objective: maximize both diversity and efficiency
    avg_diversity = sum(diversity_scores) / len(diversity_scores)
    avg_efficiency = sum(token_efficiencies) / len(token_efficiencies)
    
    # Weight them equally (could be tuned)
    combined_score = avg_diversity + avg_efficiency
    
    logger.debug(f"Weights {weights}: diversity={avg_diversity:.3f}, efficiency={avg_efficiency:.3f}, combined={combined_score:.3f}")
    
    return combined_score


def grid_search_weights(
    namespace: str,
    search_space: Optional[Dict[str, List[float]]] = None,
    training_split: float = 0.8,
    min_queries: int = MIN_QUERIES_FOR_OPTIMIZATION
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """
    Perform grid search to find optimal weights for a namespace.
    
    Args:
        namespace: Namespace to optimize for
        search_space: Dict of parameter -> list of values to try
        training_split: Fraction of data to use for training (rest for validation)
        min_queries: Minimum queries required
    
    Returns:
        Tuple of (best_weights, best_score, metadata)
    
    Raises:
        ValueError: If insufficient data for optimization
    """
    import time
    start_time = time.time()
    
    # Fetch historical query logs
    logs = get_namespace_analytics(namespace, limit=10000)
    
    if len(logs) < min_queries:
        raise ValueError(f"Insufficient data: {len(logs)} queries (need {min_queries}+)")
    
    logger.info(f"Grid search for namespace {namespace}: {len(logs)} queries")
    
    # Split into training and validation
    split_idx = int(len(logs) * training_split)
    training_logs = logs[:split_idx]
    validation_logs = logs[split_idx:]
    
    logger.info(f"Split: {len(training_logs)} training, {len(validation_logs)} validation")
    
    # Fetch memory contents for diversity calculation
    memory_ids = set()
    for log in logs:
        memories = json.loads(log["memories_selected"]) if isinstance(log["memories_selected"], str) else log["memories_selected"]
        memory_ids.update(memories)
    
    # Fetch content from database
    memory_contents = {}
    if memory_ids:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT id::text, content FROM memory_nodes WHERE id = ANY(%s)",
                (list(memory_ids),)
            )
            for row in cur.fetchall():
                memory_contents[row["id"]] = row["content"]
    
    logger.info(f"Loaded {len(memory_contents)} memory contents for diversity calculation")
    
    # Use provided search space or default
    space = search_space or DEFAULT_SEARCH_SPACE
    
    # Generate all weight combinations
    param_names = list(space.keys())
    param_values = [space[name] for name in param_names]
    combinations = list(product(*param_values))
    
    logger.info(f"Testing {len(combinations)} weight combinations")
    
    # Evaluate each combination
    best_weights = None
    best_score = -float('inf')
    all_results = []
    
    for combo in combinations:
        # Create weight dict
        weights = {param_names[i]: combo[i] for i in range(len(param_names))}
        
        # Ensure weights sum to ~1.0 (normalize if needed)
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:  # Allow small tolerance
            weights = {k: v/total for k, v in weights.items()}
        
        # Evaluate on validation set
        score = evaluate_weights(weights, validation_logs, memory_contents)
        
        all_results.append({
            "weights": weights,
            "score": score
        })
        
        if score > best_score:
            best_score = score
            best_weights = weights
            logger.info(f"New best: {weights} -> {score:.4f}")
    
    computation_time_ms = (time.time() - start_time) * 1000
    
    # Calculate stats on validation set with best weights
    diversity_scores = []
    token_efficiencies = []
    
    for log in validation_logs:
        memories_selected = json.loads(log["memories_selected"]) if isinstance(log["memories_selected"], str) else log["memories_selected"]
        tokens_used = log["total_tokens_used"]
        tokens_budget = log["total_tokens_budget"]
        
        diversity = calculate_diversity_score(memories_selected, memory_contents)
        diversity_scores.append(diversity)
        
        efficiency = tokens_used / tokens_budget if tokens_budget > 0 else 0.0
        token_efficiencies.append(efficiency)
    
    metadata = {
        "training_queries": len(training_logs),
        "validation_queries": len(validation_logs),
        "combinations_tested": len(combinations),
        "computation_time_ms": computation_time_ms,
        "avg_diversity_score": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0,
        "avg_token_efficiency": sum(token_efficiencies) / len(token_efficiencies) if token_efficiencies else 0.0,
        "search_space": space,
        "all_results": all_results[:10]  # Top 10 results for reference
    }
    
    logger.info(f"Grid search complete: best_score={best_score:.4f}, time={computation_time_ms:.0f}ms")
    
    return best_weights, best_score, metadata


def save_learned_weights(
    namespace: str,
    weights: Dict[str, float],
    objective_score: float,
    training_queries: int,
    optimization_method: str = "grid_search",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save learned weights to database and mark as active.
    
    Args:
        namespace: Namespace these weights apply to
        weights: Optimized weight configuration
        objective_score: Score achieved with these weights
        training_queries: Number of queries used for training
        optimization_method: Method used (e.g., 'grid_search')
        metadata: Additional metadata
    
    Returns:
        Weight configuration ID (UUID as string)
    """
    weight_id = str(uuid.uuid4())
    
    with db.get_cursor() as cur:
        # Deactivate existing active weights for this namespace
        cur.execute("""
            UPDATE learned_weights
            SET is_active = false, updated_at = %s
            WHERE namespace = %s AND is_active = true
        """, (datetime.utcnow(), namespace))
        
        # Insert new weights
        cur.execute("""
            INSERT INTO learned_weights (
                id, namespace, weights, training_queries, objective_score,
                optimization_method, avg_diversity_score, avg_token_efficiency,
                metadata, is_active, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            weight_id,
            namespace,
            json.dumps(weights),
            training_queries,
            objective_score,
            optimization_method,
            metadata.get("avg_diversity_score") if metadata else None,
            metadata.get("avg_token_efficiency") if metadata else None,
            json.dumps(metadata or {}),
            True,  # is_active
            datetime.utcnow(),
            datetime.utcnow()
        ))
    
    logger.info(f"Saved learned weights {weight_id} for namespace {namespace}")
    return weight_id


def get_learned_weights(namespace: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve active learned weights for a namespace.
    
    Args:
        namespace: Namespace to query
    
    Returns:
        Dict with weights and metadata, or None if no learned weights exist
    """
    with db.get_cursor() as cur:
        cur.execute("""
            SELECT id::text, weights, objective_score, training_queries,
                   optimization_method, avg_diversity_score, avg_token_efficiency,
                   metadata, created_at, updated_at
            FROM learned_weights
            WHERE namespace = %s AND is_active = true
            ORDER BY updated_at DESC
            LIMIT 1
        """, (namespace,))
        
        row = cur.fetchone()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "namespace": namespace,
            "weights": json.loads(row["weights"]),
            "objective_score": row["objective_score"],
            "training_queries": row["training_queries"],
            "optimization_method": row["optimization_method"],
            "avg_diversity_score": row["avg_diversity_score"],
            "avg_token_efficiency": row["avg_token_efficiency"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }


def log_optimization_run(
    namespace: str,
    best_weights: Dict[str, float],
    best_score: float,
    metadata: Dict[str, Any]
) -> str:
    """
    Log an optimization run to optimization_history table.
    
    Args:
        namespace: Namespace optimized
        best_weights: Best weights found
        best_score: Best score achieved
        metadata: Optimization metadata
    
    Returns:
        History entry ID
    """
    history_id = str(uuid.uuid4())
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO optimization_history (
                id, namespace, optimization_method, training_queries,
                validation_queries, search_space, best_weights, best_score,
                computation_time_ms, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            history_id,
            namespace,
            "grid_search",
            metadata.get("training_queries", 0),
            metadata.get("validation_queries", 0),
            json.dumps(metadata.get("search_space", {})),
            json.dumps(best_weights),
            best_score,
            metadata.get("computation_time_ms", 0),
            json.dumps(metadata)
        ))
    
    logger.info(f"Logged optimization run {history_id} for namespace {namespace}")
    return history_id


def optimize_namespace(
    namespace: str,
    search_space: Optional[Dict[str, List[float]]] = None,
    min_queries: int = MIN_QUERIES_FOR_OPTIMIZATION
) -> Dict[str, Any]:
    """
    Complete optimization workflow for a namespace.
    
    Args:
        namespace: Namespace to optimize
        search_space: Optional custom search space
        min_queries: Minimum queries required
    
    Returns:
        Dict with optimization results
    
    Raises:
        ValueError: If insufficient data
    """
    logger.info(f"Starting optimization for namespace: {namespace}")
    
    # Run grid search
    best_weights, best_score, metadata = grid_search_weights(
        namespace=namespace,
        search_space=search_space,
        min_queries=min_queries
    )
    
    # Save learned weights
    weight_id = save_learned_weights(
        namespace=namespace,
        weights=best_weights,
        objective_score=best_score,
        training_queries=metadata["training_queries"],
        metadata=metadata
    )
    
    # Log optimization run
    history_id = log_optimization_run(
        namespace=namespace,
        best_weights=best_weights,
        best_score=best_score,
        metadata=metadata
    )
    
    logger.info(f"Optimization complete for namespace {namespace}: weight_id={weight_id}")
    
    return {
        "weight_id": weight_id,
        "history_id": history_id,
        "namespace": namespace,
        "best_weights": best_weights,
        "objective_score": best_score,
        "metadata": metadata
    }
