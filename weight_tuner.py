"""
Weight Tuner Module
===================

Utilities for tuning and optimizing scoring weights based on test data.
Helps find optimal weight configurations for different use cases.

Author: vex-memory team
Version: 1.2.0
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import itertools
import logging
from prioritizer import (
    MemoryPrioritizer,
    ScoringWeights,
    PriorityMappings,
    TokenEstimator
)

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """Configuration for a set of weights."""
    weights: ScoringWeights
    name: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "weights": {
                "similarity": self.weights.similarity,
                "importance": self.weights.importance,
                "recency": self.weights.recency,
                "diversity": self.weights.diversity,
                "entity_coverage": self.weights.entity_coverage
            }
        }


@dataclass
class BenchmarkResult:
    """Result from a weight configuration benchmark."""
    config: WeightConfig
    score: float
    metrics: Dict[str, float]
    selected_memory_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "score": self.score,
            "metrics": self.metrics,
            "selected_count": len(self.selected_memory_ids)
        }


class WeightTuner:
    """Utility for tuning and optimizing scoring weights."""
    
    # Predefined weight configurations for common use cases
    PRESETS = {
        "balanced": WeightConfig(
            weights=ScoringWeights(
                similarity=0.35,
                importance=0.30,
                recency=0.20,
                diversity=0.10,
                entity_coverage=0.05
            ),
            name="Balanced",
            description="Balanced across all factors"
        ),
        "relevance_focused": WeightConfig(
            weights=ScoringWeights(
                similarity=0.60,
                importance=0.25,
                recency=0.10,
                diversity=0.03,
                entity_coverage=0.02
            ),
            name="Relevance Focused",
            description="Prioritizes similarity and importance over time"
        ),
        "recency_focused": WeightConfig(
            weights=ScoringWeights(
                similarity=0.25,
                importance=0.20,
                recency=0.40,
                diversity=0.10,
                entity_coverage=0.05
            ),
            name="Recency Focused",
            description="Prioritizes recent memories"
        ),
        "diversity_focused": WeightConfig(
            weights=ScoringWeights(
                similarity=0.30,
                importance=0.25,
                recency=0.15,
                diversity=0.20,
                entity_coverage=0.10
            ),
            name="Diversity Focused",
            description="Maximizes variety and coverage"
        ),
        "entity_focused": WeightConfig(
            weights=ScoringWeights(
                similarity=0.30,
                importance=0.25,
                recency=0.10,
                diversity=0.10,
                entity_coverage=0.25
            ),
            name="Entity Focused",
            description="Prioritizes entity coverage"
        ),
        "importance_focused": WeightConfig(
            weights=ScoringWeights(
                similarity=0.25,
                importance=0.50,
                recency=0.15,
                diversity=0.05,
                entity_coverage=0.05
            ),
            name="Importance Focused",
            description="Prioritizes memory importance"
        )
    }
    
    def __init__(
        self,
        token_estimator: Optional[TokenEstimator] = None,
        priority_mappings: Optional[PriorityMappings] = None
    ):
        """Initialize weight tuner.
        
        Args:
            token_estimator: Token estimator instance
            priority_mappings: Priority mappings for types/namespaces
        """
        self.token_estimator = token_estimator or TokenEstimator()
        self.priority_mappings = priority_mappings or PriorityMappings()
    
    def get_preset(self, name: str) -> Optional[WeightConfig]:
        """Get a predefined weight configuration.
        
        Args:
            name: Preset name (balanced, relevance_focused, etc.)
            
        Returns:
            WeightConfig or None if not found
        """
        return self.PRESETS.get(name.lower())
    
    def list_presets(self) -> List[Dict[str, str]]:
        """List all available presets.
        
        Returns:
            List of preset info dictionaries
        """
        return [
            {
                "name": config.name,
                "key": key,
                "description": config.description
            }
            for key, config in self.PRESETS.items()
        ]
    
    def benchmark_config(
        self,
        config: WeightConfig,
        memories: List[Dict[str, Any]],
        token_budget: int,
        ground_truth: Optional[List[str]] = None,
        evaluate_fn: Optional[Callable] = None
    ) -> BenchmarkResult:
        """Benchmark a weight configuration.
        
        Args:
            config: Weight configuration to test
            memories: List of memories to prioritize
            token_budget: Token budget for selection
            ground_truth: Optional list of ideal memory IDs for comparison
            evaluate_fn: Optional custom evaluation function
            
        Returns:
            BenchmarkResult with score and metrics
        """
        # Create prioritizer with this config
        prioritizer = MemoryPrioritizer(
            token_estimator=self.token_estimator,
            weights=config.weights,
            priority_mappings=self.priority_mappings
        )
        
        # Run prioritization
        selected, metadata = prioritizer.prioritize(
            memories=memories,
            token_budget=token_budget
        )
        
        selected_ids = [m["id"] for m in selected]
        
        # Calculate metrics
        metrics = {
            "count": len(selected),
            "utilization": metadata.get("utilization", 0.0),
            "average_score": metadata.get("average_score", 0.0),
            "diversity_filtered": metadata.get("diversity_filtered", 0)
        }
        
        # Calculate score
        if evaluate_fn:
            score = evaluate_fn(selected, metadata)
        elif ground_truth:
            # Calculate overlap with ground truth
            overlap = len(set(selected_ids) & set(ground_truth))
            score = overlap / len(ground_truth) if ground_truth else 0.0
            metrics["ground_truth_overlap"] = overlap
            metrics["ground_truth_recall"] = score
        else:
            # Use average score as default metric
            score = metadata.get("average_score", 0.0)
        
        return BenchmarkResult(
            config=config,
            score=score,
            metrics=metrics,
            selected_memory_ids=selected_ids
        )
    
    def grid_search(
        self,
        memories: List[Dict[str, Any]],
        token_budget: int,
        similarity_range: Tuple[float, float, float] = (0.2, 0.6, 0.1),
        importance_range: Tuple[float, float, float] = (0.2, 0.5, 0.1),
        recency_range: Tuple[float, float, float] = (0.1, 0.4, 0.1),
        ground_truth: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[BenchmarkResult]:
        """Perform grid search over weight configurations.
        
        Args:
            memories: List of memories to prioritize
            token_budget: Token budget for selection
            similarity_range: (min, max, step) for similarity weight
            importance_range: (min, max, step) for importance weight
            recency_range: (min, max, step) for recency weight
            ground_truth: Optional ground truth memory IDs
            top_k: Number of top results to return
            
        Returns:
            List of top K benchmark results sorted by score
        """
        # Generate weight combinations
        sim_values = self._frange(*similarity_range)
        imp_values = self._frange(*importance_range)
        rec_values = self._frange(*recency_range)
        
        results = []
        total_configs = 0
        
        for sim, imp, rec in itertools.product(sim_values, imp_values, rec_values):
            # Reserve small amounts for diversity and entity coverage
            remaining = 1.0 - sim - imp - rec
            if remaining < 0 or remaining > 0.3:
                continue  # Skip invalid combinations
            
            div = remaining * 0.6  # 60% for diversity
            ent = remaining * 0.4  # 40% for entity coverage
            
            config = WeightConfig(
                weights=ScoringWeights(
                    similarity=sim,
                    importance=imp,
                    recency=rec,
                    diversity=div,
                    entity_coverage=ent
                ),
                name=f"Grid_{total_configs}",
                description=f"sim={sim:.2f} imp={imp:.2f} rec={rec:.2f}"
            )
            
            try:
                result = self.benchmark_config(
                    config=config,
                    memories=memories,
                    token_budget=token_budget,
                    ground_truth=ground_truth
                )
                results.append(result)
                total_configs += 1
            except Exception as e:
                logger.warning(f"Failed to benchmark config {total_configs}: {e}")
                continue
        
        # Sort by score (descending) and return top K
        results.sort(key=lambda r: r.score, reverse=True)
        logger.info(f"Grid search completed: {total_configs} configs tested, top score: {results[0].score:.3f}")
        
        return results[:top_k]
    
    def compare_presets(
        self,
        memories: List[Dict[str, Any]],
        token_budget: int,
        ground_truth: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """Compare all preset configurations.
        
        Args:
            memories: List of memories to prioritize
            token_budget: Token budget for selection
            ground_truth: Optional ground truth memory IDs
            
        Returns:
            List of benchmark results for all presets, sorted by score
        """
        results = []
        
        for key, config in self.PRESETS.items():
            try:
                result = self.benchmark_config(
                    config=config,
                    memories=memories,
                    token_budget=token_budget,
                    ground_truth=ground_truth
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to benchmark preset '{key}': {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    def recommend_weights(
        self,
        use_case: str = "balanced"
    ) -> WeightConfig:
        """Get recommended weights for a use case.
        
        Args:
            use_case: Use case identifier (balanced, relevance_focused, etc.)
            
        Returns:
            Recommended WeightConfig
        """
        preset = self.get_preset(use_case)
        if preset:
            return preset
        
        # Default to balanced
        logger.warning(f"Unknown use case '{use_case}', using 'balanced'")
        return self.PRESETS["balanced"]
    
    @staticmethod
    def _frange(start: float, stop: float, step: float) -> List[float]:
        """Generate float range (inclusive).
        
        Args:
            start: Start value
            stop: Stop value
            step: Step size
            
        Returns:
            List of float values
        """
        values = []
        current = start
        while current <= stop + 1e-9:  # Small epsilon for float comparison
            values.append(round(current, 2))
            current += step
        return values


# Convenience function
def get_recommended_weights(use_case: str = "balanced") -> Dict[str, float]:
    """Get recommended weights for a use case.
    
    Args:
        use_case: Use case identifier
        
    Returns:
        Dictionary of weight values
    """
    tuner = WeightTuner()
    config = tuner.recommend_weights(use_case)
    return config.to_dict()["weights"]
