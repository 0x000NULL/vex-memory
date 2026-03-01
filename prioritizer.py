"""
Memory Prioritizer Module
=========================

Intelligent memory prioritization with multi-factor scoring and token budgets.
Implements greedy selection with diversity filtering for optimal context assembly.

Version 1.2.0 adds:
- MMR (Maximal Marginal Relevance) for better diversity
- Entity coverage tracking
- Type and namespace priority weighting

Author: vex-memory team
Version: 1.2.0
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
import math
import logging
from dataclasses import dataclass, field

from token_estimator import TokenEstimator

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for multi-factor scoring."""
    similarity: float = 0.4
    importance: float = 0.3
    recency: float = 0.2
    diversity: float = 0.05
    entity_coverage: float = 0.05  # New in v1.2.0
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = self.similarity + self.importance + self.recency + self.diversity + self.entity_coverage
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Weights sum to {total:.3f}, not 1.0. Normalizing...")
            # Normalize
            self.similarity /= total
            self.importance /= total
            self.recency /= total
            self.diversity /= total
            self.entity_coverage /= total


@dataclass
class MemoryScore:
    """Score result for a single memory."""
    memory_id: str
    score: float
    token_count: int
    factors: Dict[str, float]
    memory: Dict[str, Any] = field(repr=False)


class MemoryPrioritizer:
    """Intelligent memory prioritization with token budgets."""
    
    def __init__(
        self,
        token_estimator: Optional[TokenEstimator] = None,
        weights: Optional[ScoringWeights] = None,
        recency_half_life_days: float = 30.0
    ):
        """Initialize prioritizer.
        
        Args:
            token_estimator: Token estimator instance (creates default if None)
            weights: Scoring weights (uses defaults if None)
            recency_half_life_days: Half-life for recency decay in days
        """
        self.token_estimator = token_estimator or TokenEstimator()
        self.weights = weights or ScoringWeights()
        self.recency_half_life_days = recency_half_life_days
        self.recency_decay_rate = math.log(2) / recency_half_life_days
    
    def prioritize(
        self,
        memories: List[Dict[str, Any]],
        token_budget: int,
        diversity_threshold: float = 0.7,
        min_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select optimal memories within token budget.
        
        Args:
            memories: List of memory dictionaries
            token_budget: Maximum tokens allowed
            diversity_threshold: Jaccard similarity threshold for diversity (0-1)
            min_score: Optional minimum score threshold
            
        Returns:
            Tuple of (selected_memories, metadata)
        """
        if not memories:
            return [], {
                "total_tokens": 0,
                "budget": token_budget,
                "utilization": 0.0,
                "memories_selected": 0,
                "memories_available": 0,
                "diversity_filtered": 0
            }
        
        # Score all memories
        scored = []
        for memory in memories:
            try:
                score_obj = self._score_memory(memory)
                
                # Apply minimum score filter
                if min_score is None or score_obj.score >= min_score:
                    scored.append(score_obj)
            except Exception as e:
                logger.warning(f"Failed to score memory {memory.get('id')}: {e}")
                continue
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # Greedy selection with diversity and budget enforcement
        selected = []
        selected_contents = []
        total_tokens = 0
        diversity_filtered_count = 0
        
        for score_obj in scored:
            memory = score_obj.memory
            
            # Check token budget
            if total_tokens + score_obj.token_count > token_budget:
                # Try to truncate if this is a high-value memory
                if score_obj.score > 0.7 and not selected:
                    # First memory and high score - allow truncation
                    available = token_budget - total_tokens
                    if available > 50:  # Minimum useful tokens
                        truncated = self.token_estimator.truncate_to_budget(
                            memory["content"],
                            available - 10  # Reserve for formatting
                        )
                        memory_copy = memory.copy()
                        memory_copy["content"] = truncated.text
                        memory_copy["_truncated"] = True
                        
                        selected.append(memory_copy)
                        total_tokens += truncated.token_count
                continue
            
            # Check diversity (skip for first memory)
            if selected and self._is_too_similar(
                memory,
                selected_contents,
                diversity_threshold
            ):
                diversity_filtered_count += 1
                continue
            
            # Add to selected
            selected.append(memory)
            selected_contents.append(memory.get("content", ""))
            total_tokens += score_obj.token_count
        
        # Calculate metadata
        metadata = {
            "total_tokens": total_tokens,
            "budget": token_budget,
            "utilization": total_tokens / token_budget if token_budget > 0 else 0.0,
            "memories_selected": len(selected),
            "memories_available": len(memories),
            "diversity_filtered": diversity_filtered_count,
            "average_score": sum(s.score for s in scored[:len(selected)]) / len(selected) if selected else 0.0,
            "weights": {
                "similarity": self.weights.similarity,
                "importance": self.weights.importance,
                "recency": self.weights.recency,
                "diversity": self.weights.diversity
            }
        }
        
        # Add score information to selected memories
        for i, memory in enumerate(selected):
            # Find the original score object
            score_obj = next((s for s in scored if s.memory_id == memory["id"]), None)
            if score_obj:
                memory["_score"] = score_obj.score
                memory["_score_factors"] = score_obj.factors
        
        return selected, metadata
    
    def _score_memory(self, memory: Dict[str, Any]) -> MemoryScore:
        """Calculate composite score for a memory.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            MemoryScore object
        """
        factors = {}
        
        # Factor 1: Similarity (from vector search, should be in memory)
        factors["similarity"] = memory.get("similarity_score", memory.get("similarity", 0.5))
        
        # Factor 2: Importance
        factors["importance"] = memory.get("importance_score", 0.5)
        
        # Factor 3: Recency (exponential decay)
        factors["recency"] = self._calculate_recency(memory)
        
        # Composite score (weighted sum)
        score = (
            self.weights.similarity * factors["similarity"] +
            self.weights.importance * factors["importance"] +
            self.weights.recency * factors["recency"]
        )
        
        # Estimate tokens
        token_estimate = self.token_estimator.estimate_memory(memory)
        
        return MemoryScore(
            memory_id=memory["id"],
            score=score,
            token_count=token_estimate.token_count,
            factors=factors,
            memory=memory
        )
    
    def _calculate_recency(self, memory: Dict[str, Any]) -> float:
        """Calculate recency score with exponential decay.
        
        Args:
            memory: Memory dictionary with event_time
            
        Returns:
            Recency score (0-1)
        """
        event_time = memory.get("event_time")
        
        if not event_time:
            # No timestamp - return neutral score
            return 0.5
        
        # Parse event_time if it's a string
        if isinstance(event_time, str):
            try:
                event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return 0.5
        
        # Ensure timezone-aware
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        
        # Calculate age in days
        now = datetime.now(timezone.utc)
        age_seconds = (now - event_time).total_seconds()
        age_days = age_seconds / 86400
        
        # Exponential decay: score = e^(-λ * age)
        # where λ = ln(2) / half_life
        recency_score = math.exp(-self.recency_decay_rate * age_days)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, recency_score))
    
    def _is_too_similar(
        self,
        memory: Dict[str, Any],
        selected_contents: List[str],
        threshold: float
    ) -> bool:
        """Check if memory is too similar to already selected memories.
        
        Args:
            memory: Memory to check
            selected_contents: List of already selected memory contents
            threshold: Jaccard similarity threshold (0-1)
            
        Returns:
            True if too similar, False otherwise
        """
        content = memory.get("content", "")
        
        if not content or not selected_contents:
            return False
        
        # Check similarity against each selected memory
        for prev_content in selected_contents:
            similarity = self._jaccard_similarity(content, prev_content)
            if similarity > threshold:
                return True
        
        return False
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts.
        
        Uses word-level tokenization (simple split).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize by splitting on whitespace and lowercase
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def update_weights(self, weights: ScoringWeights):
        """Update scoring weights.
        
        Args:
            weights: New scoring weights
        """
        self.weights = weights
        logger.info(f"Updated weights: {weights}")
    
    def get_weights(self) -> ScoringWeights:
        """Get current scoring weights.
        
        Returns:
            Current ScoringWeights
        """
        return self.weights
    
    def prioritize_mmr(
        self,
        memories: List[Dict[str, Any]],
        token_budget: int,
        lambda_param: float = 0.7,
        min_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select memories using Maximal Marginal Relevance (MMR).
        
        MMR balances relevance and diversity by iteratively selecting
        memories that are relevant to the query but different from
        already-selected memories.
        
        Args:
            memories: List of memory dictionaries
            token_budget: Maximum tokens allowed
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            min_score: Optional minimum score threshold
            
        Returns:
            Tuple of (selected_memories, metadata)
        """
        if not memories:
            return [], {
                "total_tokens": 0,
                "budget": token_budget,
                "utilization": 0.0,
                "memories_selected": 0,
                "memories_available": 0,
                "method": "mmr"
            }
        
        # Score all memories
        scored = []
        for memory in memories:
            try:
                score_obj = self._score_memory(memory)
                if min_score is None or score_obj.score >= min_score:
                    scored.append(score_obj)
            except Exception as e:
                logger.warning(f"Failed to score memory {memory.get('id')}: {e}")
                continue
        
        if not scored:
            return [], {
                "total_tokens": 0,
                "budget": token_budget,
                "utilization": 0.0,
                "memories_selected": 0,
                "memories_available": len(memories),
                "method": "mmr"
            }
        
        # MMR selection
        selected = []
        remaining = scored.copy()
        total_tokens = 0
        
        while remaining and total_tokens < token_budget:
            if not selected:
                # First memory: highest relevance score
                best = max(remaining, key=lambda x: x.score)
            else:
                # Subsequent memories: MMR score
                best = None
                best_mmr_score = float('-inf')
                
                for candidate in remaining:
                    # Relevance component
                    relevance = candidate.score
                    
                    # Diversity component: max similarity to selected
                    max_sim_to_selected = max(
                        self._jaccard_similarity(
                            candidate.memory.get("content", ""),
                            sel.memory.get("content", "")
                        )
                        for sel in selected
                    )
                    
                    # MMR score: λ * relevance - (1-λ) * max_similarity
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best = candidate
            
            if best is None:
                break
            
            # Check token budget
            if total_tokens + best.token_count > token_budget:
                # Try truncation for high-value memories
                if best.score > 0.7 and not selected:
                    available = token_budget - total_tokens
                    if available > 50:
                        truncated = self.token_estimator.truncate_to_budget(
                            best.memory["content"],
                            available - 10
                        )
                        memory_copy = best.memory.copy()
                        memory_copy["content"] = truncated.text
                        memory_copy["_truncated"] = True
                        memory_copy["_score"] = best.score
                        memory_copy["_score_factors"] = best.factors
                        
                        selected.append(best)
                        total_tokens += truncated.token_count
                break
            
            # Add to selected
            selected.append(best)
            remaining.remove(best)
            total_tokens += best.token_count
        
        # Convert to memory dictionaries
        selected_memories = []
        for score_obj in selected:
            memory = score_obj.memory.copy()
            memory["_score"] = score_obj.score
            memory["_score_factors"] = score_obj.factors
            selected_memories.append(memory)
        
        # Metadata
        metadata = {
            "total_tokens": total_tokens,
            "budget": token_budget,
            "utilization": total_tokens / token_budget if token_budget > 0 else 0.0,
            "memories_selected": len(selected_memories),
            "memories_available": len(memories),
            "method": "mmr",
            "lambda": lambda_param,
            "average_score": sum(s.score for s in selected) / len(selected) if selected else 0.0,
            "weights": {
                "similarity": self.weights.similarity,
                "importance": self.weights.importance,
                "recency": self.weights.recency,
                "diversity": self.weights.diversity,
                "entity_coverage": self.weights.entity_coverage
            }
        }
        
        return selected_memories, metadata
    
    def prioritize_with_entity_coverage(
        self,
        memories: List[Dict[str, Any]],
        target_entities: set,
        token_budget: int,
        entity_boost: float = 0.2,
        diversity_threshold: float = 0.7,
        min_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prioritize memories with entity coverage tracking.
        
        Gives bonus scores to memories that cover important entities.
        
        Args:
            memories: List of memory dictionaries (with '_entities' field)
            target_entities: Set of important entities to cover
            token_budget: Maximum tokens
            entity_boost: Maximum boost for entity coverage (default: 0.2)
            diversity_threshold: Jaccard threshold for diversity
            min_score: Optional minimum score threshold
            
        Returns:
            Tuple of (selected_memories, metadata)
        """
        if not memories or not target_entities:
            # Fall back to regular prioritization
            return self.prioritize(
                memories,
                token_budget,
                diversity_threshold=diversity_threshold,
                min_score=min_score
            )
        
        # Track covered entities
        covered_entities = set()
        
        # Score memories with entity coverage boost
        scored = []
        for memory in memories:
            try:
                base_score_obj = self._score_memory(memory)
                
                # Calculate entity coverage boost
                memory_entities = set(memory.get("_entities", set()))
                uncovered = target_entities - covered_entities
                overlap = memory_entities & uncovered
                
                coverage_boost = 0.0
                if uncovered:
                    coverage_boost = entity_boost * (len(overlap) / len(uncovered))
                
                # Adjusted score
                adjusted_score = base_score_obj.score + coverage_boost
                
                if min_score is None or adjusted_score >= min_score:
                    score_obj = MemoryScore(
                        memory_id=base_score_obj.memory_id,
                        score=adjusted_score,
                        token_count=base_score_obj.token_count,
                        factors={
                            **base_score_obj.factors,
                            "entity_coverage": coverage_boost
                        },
                        memory=memory
                    )
                    scored.append(score_obj)
            except Exception as e:
                logger.warning(f"Failed to score memory {memory.get('id')}: {e}")
                continue
        
        # Sort by adjusted score
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # Greedy selection with diversity and coverage tracking
        selected = []
        selected_contents = []
        total_tokens = 0
        diversity_filtered_count = 0
        
        for score_obj in scored:
            memory = score_obj.memory
            
            # Check token budget
            if total_tokens + score_obj.token_count > token_budget:
                if score_obj.score > 0.7 and not selected:
                    available = token_budget - total_tokens
                    if available > 50:
                        truncated = self.token_estimator.truncate_to_budget(
                            memory["content"],
                            available - 10
                        )
                        memory_copy = memory.copy()
                        memory_copy["content"] = truncated.text
                        memory_copy["_truncated"] = True
                        
                        selected.append(memory_copy)
                        total_tokens += truncated.token_count
                        
                        # Update covered entities
                        covered_entities.update(memory.get("_entities", set()))
                continue
            
            # Check diversity
            if selected and self._is_too_similar(
                memory,
                selected_contents,
                diversity_threshold
            ):
                diversity_filtered_count += 1
                continue
            
            # Add to selected
            selected.append(memory)
            selected_contents.append(memory.get("content", ""))
            total_tokens += score_obj.token_count
            
            # Track covered entities
            covered_entities.update(memory.get("_entities", set()))
            
            # Add score info
            memory["_score"] = score_obj.score
            memory["_score_factors"] = score_obj.factors
        
        # Calculate coverage
        coverage = len(covered_entities & target_entities) / len(target_entities) if target_entities else 1.0
        
        # Metadata
        metadata = {
            "total_tokens": total_tokens,
            "budget": token_budget,
            "utilization": total_tokens / token_budget if token_budget > 0 else 0.0,
            "memories_selected": len(selected),
            "memories_available": len(memories),
            "diversity_filtered": diversity_filtered_count,
            "entity_coverage": coverage,
            "entities_covered": len(covered_entities & target_entities),
            "entities_target": len(target_entities),
            "average_score": sum(s.score for s in scored[:len(selected)]) / len(selected) if selected else 0.0,
            "method": "entity_coverage",
            "weights": {
                "similarity": self.weights.similarity,
                "importance": self.weights.importance,
                "recency": self.weights.recency,
                "diversity": self.weights.diversity,
                "entity_coverage": self.weights.entity_coverage
            }
        }
        
        return selected, metadata


# Convenience function
def prioritize_memories(
    memories: List[Dict[str, Any]],
    token_budget: int = 4000,
    model: str = "gpt-4",
    **kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Quick prioritization without creating instances.
    
    Args:
        memories: List of memory dictionaries
        token_budget: Maximum tokens
        model: Model for token counting
        **kwargs: Additional arguments for prioritizer
        
    Returns:
        Tuple of (selected_memories, metadata)
    """
    estimator = TokenEstimator(model=model)
    prioritizer = MemoryPrioritizer(token_estimator=estimator)
    
    return prioritizer.prioritize(memories, token_budget, **kwargs)
