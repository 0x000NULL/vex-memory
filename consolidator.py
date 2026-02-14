"""
Vex Memory Consolidation Engine v2.0
====================================

The "sleep cycle" - processes daily notes into long-term memory with intelligent
consolidation, deduplication, conflict resolution, and forgetting curves.

This module handles the critical transformation from raw daily memories into 
organized, interconnected long-term knowledge. Like human sleep memory processing,
it strengthens important connections and allows less important memories to fade.

"""

import os
import json
import uuid
import math
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, Counter

from extractor import (
    MemoryNode, Entity, Fact, Relationship, EmotionalMarker,
    MemoryType, EntityType, RelationType, MemoryExtractor
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConflictType(Enum):
    FACTUAL_CONTRADICTION = "factual_contradiction"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    PREFERENCE_EVOLUTION = "preference_evolution"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"


class ConflictSeverity(Enum):
    LOW = "low"        # Minor inconsistencies, can coexist
    MEDIUM = "medium"  # Requires attention, may need resolution
    HIGH = "high"      # Major contradictions, must be resolved
    CRITICAL = "critical"  # System-threatening conflicts


@dataclass
class MemoryConflict:
    """Represents a detected conflict between memories."""
    id: str
    conflict_type: ConflictType
    memory1_id: str
    memory2_id: str
    description: str
    severity: ConflictSeverity
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)
    resolution_status: str = "unresolved"
    resolution_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationRun:
    """Tracks a consolidation processing session."""
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    target_date: Optional[date] = None
    status: str = "running"  # running, completed, failed
    
    # Processing metrics
    memories_processed: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    relations_created: int = 0
    conflicts_detected: int = 0
    memories_merged: int = 0
    memories_decayed: int = 0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)


class MemoryConsolidator:
    """Main consolidation engine that processes and optimizes memory storage."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the consolidator with configuration."""
        self.config = config or {}
        self.extractor = MemoryExtractor()
        
        # Consolidation parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.importance_boost_factor = self.config.get('importance_boost', 1.2)
        self.decay_rate = self.config.get('decay_rate', 0.1)
        self.min_importance_threshold = self.config.get('min_importance', 0.1)
        
        # Working memory for current consolidation
        self.current_memories: List[MemoryNode] = []
        self.entity_registry: Dict[str, Entity] = {}
        self.detected_conflicts: List[MemoryConflict] = []
        
        logger.info("Memory consolidator initialized")

    def consolidate_daily_memories(self, target_date: date, 
                                   daily_file_path: str,
                                   existing_memories: List[MemoryNode] = None) -> ConsolidationRun:
        """
        Main consolidation pipeline - processes a day's memories.
        
        Args:
            target_date: The date being processed
            daily_file_path: Path to the daily memory file
            existing_memories: Previously consolidated memories to consider
            
        Returns:
            ConsolidationRun with processing results
        """
        run = ConsolidationRun(
            id=str(uuid.uuid4()),
            start_time=datetime.now(),
            target_date=target_date,
            config=self.config.copy()
        )
        
        try:
            logger.info(f"Starting consolidation for {target_date}")
            
            # 1. Extract structured memories from raw daily logs
            raw_memories = self._load_and_extract_daily_memories(daily_file_path, target_date)
            run.memories_processed = len(raw_memories)
            run.logs.append(f"Extracted {len(raw_memories)} raw memories")
            
            if not raw_memories:
                run.status = "completed"
                run.end_time = datetime.now()
                logger.info("No memories to process")
                return run
            
            # 2. Load existing memory context
            self.current_memories = existing_memories or []
            self._build_entity_registry()
            
            # 3. Process each new memory
            consolidated_memories = []
            for memory in raw_memories:
                processed = self._process_single_memory(memory, run)
                if processed:
                    consolidated_memories.append(processed)
            
            # 4. Detect and resolve conflicts
            self._detect_memory_conflicts(consolidated_memories, run)
            
            # 5. Update entity importance and relationships
            self._update_entity_importance(consolidated_memories, run)
            
            # 6. Apply decay to older memories
            self._apply_decay_functions(run)
            
            # 7. Merge similar memories
            merged_memories = self._merge_similar_memories(consolidated_memories, run)
            
            # 8. Update final memory collection
            self.current_memories.extend(merged_memories)
            
            run.status = "completed"
            run.end_time = datetime.now()
            
            logger.info(f"Consolidation completed in {(run.end_time - run.start_time).total_seconds():.1f}s")
            
        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.end_time = datetime.now()
            logger.error(f"Consolidation failed: {e}", exc_info=True)
        
        return run

    def _load_and_extract_daily_memories(self, file_path: str, target_date: date) -> List[MemoryNode]:
        """Load and extract memories from a daily file."""
        if not os.path.exists(file_path):
            logger.warning(f"Daily file not found: {file_path}")
            return []
        
        logger.info(f"Extracting memories from {file_path}")
        memories = self.extractor.extract_from_file(file_path)
        
        # Ensure all memories have the correct date context
        for memory in memories:
            if not memory.event_time:
                memory.event_time = datetime.combine(target_date, datetime.min.time())
        
        return memories

    def _build_entity_registry(self):
        """Build a registry of known entities for deduplication."""
        self.entity_registry = {}
        
        for memory in self.current_memories:
            for entity in memory.entities:
                key = (entity.canonical_name, entity.type)
                if key not in self.entity_registry:
                    self.entity_registry[key] = entity
                else:
                    # Merge entity information
                    existing = self.entity_registry[key]
                    existing.aliases = list(set(existing.aliases + entity.aliases))
                    existing.mention_count = getattr(existing, 'mention_count', 1) + 1
                    # Update attributes
                    if entity.attributes:
                        if not existing.attributes:
                            existing.attributes = {}
                        existing.attributes.update(entity.attributes)

    def _process_single_memory(self, memory: MemoryNode, run: ConsolidationRun) -> Optional[MemoryNode]:
        """Process a single memory through the consolidation pipeline."""
        
        # 1. Check for duplicates
        if self._is_duplicate_memory(memory):
            logger.debug(f"Skipping duplicate memory: {memory.id}")
            return None
        
        # 2. Enhance entities
        self._enhance_memory_entities(memory, run)
        
        # 3. Calculate enhanced importance score
        memory.importance_score = self._calculate_enhanced_importance(memory)
        
        # 4. Build relationships with existing memories
        self._create_memory_relationships(memory, run)
        
        # 5. Filter out low-importance memories
        if memory.importance_score < self.min_importance_threshold:
            logger.debug(f"Filtering low-importance memory: {memory.id}")
            return None
        
        return memory

    def _is_duplicate_memory(self, memory: MemoryNode) -> bool:
        """Check if this memory duplicates existing content."""
        content_lower = memory.content.lower().strip()
        
        for existing in self.current_memories:
            existing_lower = existing.content.lower().strip()
            
            # Exact match
            if content_lower == existing_lower:
                return True
            
            # High similarity (simple word-based)
            similarity = self._calculate_text_similarity(content_lower, existing_lower)
            if similarity > self.similarity_threshold:
                logger.debug(f"High similarity detected: {similarity:.3f}")
                return True
        
        return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple word-based similarity (Jaccard coefficient)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _enhance_memory_entities(self, memory: MemoryNode, run: ConsolidationRun):
        """Enhance entities with information from the entity registry."""
        enhanced_entities = []
        
        for entity in memory.entities:
            key = (entity.canonical_name, entity.type)
            
            if key in self.entity_registry:
                # Update existing entity
                existing = self.entity_registry[key]
                existing.mention_count = getattr(existing, 'mention_count', 1) + 1
                existing.aliases = list(set(existing.aliases + entity.aliases))
                
                if entity.attributes:
                    if not existing.attributes:
                        existing.attributes = {}
                    existing.attributes.update(entity.attributes)
                
                enhanced_entities.append(existing)
                run.entities_updated += 1
                
            else:
                # Add new entity to registry
                entity.mention_count = 1
                self.entity_registry[key] = entity
                enhanced_entities.append(entity)
                run.entities_created += 1
        
        memory.entities = enhanced_entities

    def _calculate_enhanced_importance(self, memory: MemoryNode) -> float:
        """Calculate importance score with multiple factors."""
        base_score = memory.importance_score
        
        # Recency factor (newer is slightly more important)
        if memory.event_time:
            days_old = (datetime.now() - memory.event_time).days
            recency_factor = math.exp(-days_old / 30.0)  # 30-day half-life
        else:
            recency_factor = 1.0
        
        # Entity importance factor
        entity_factor = 1.0
        if memory.entities:
            entity_scores = []
            for entity in memory.entities:
                mention_count = getattr(entity, 'mention_count', 1)
                entity_importance = min(1.0, math.log(mention_count + 1) * 0.2)
                entity_scores.append(entity_importance)
            entity_factor = 1.0 + (sum(entity_scores) / len(entity_scores)) * 0.5
        
        # Content richness factor
        richness_factor = 1.0
        richness_factor += len(memory.facts) * 0.1
        richness_factor += len(memory.relationships) * 0.05
        richness_factor += len(memory.emotional_markers) * 0.08
        
        # Memory type factor
        type_factors = {
            MemoryType.PROCEDURAL: 1.3,  # How-tos are valuable
            MemoryType.SEMANTIC: 1.2,    # Facts are important
            MemoryType.EMOTIONAL: 1.1,   # Preferences matter
            MemoryType.EPISODIC: 1.0     # Events baseline
        }
        type_factor = type_factors.get(memory.type, 1.0)
        
        # Combine factors
        enhanced_score = (
            base_score * 
            recency_factor * 
            entity_factor * 
            richness_factor * 
            type_factor
        )
        
        return min(1.0, enhanced_score)

    def _create_memory_relationships(self, memory: MemoryNode, run: ConsolidationRun):
        """Create relationships between this memory and existing memories."""
        
        for existing in self.current_memories:
            # Check for entity overlap
            shared_entities = self._find_shared_entities(memory, existing)
            if shared_entities:
                # Create reference relationship
                relationship = Relationship(
                    source=memory.id,
                    target=existing.id,
                    relation_type=RelationType.REFERENCES,
                    weight=min(1.0, len(shared_entities) * 0.3),
                    confidence=0.7,
                    description=f"Shared entities: {', '.join(shared_entities)}"
                )
                memory.relationships.append(relationship)
                run.relations_created += 1
            
            # Check for temporal relationships
            if self._are_temporally_related(memory, existing):
                relationship = Relationship(
                    source=memory.id,
                    target=existing.id,
                    relation_type=RelationType.TEMPORAL,
                    weight=0.6,
                    confidence=0.6,
                    description="Temporally related events"
                )
                memory.relationships.append(relationship)
                run.relations_created += 1

    def _find_shared_entities(self, memory1: MemoryNode, memory2: MemoryNode) -> List[str]:
        """Find entities shared between two memories."""
        entities1 = {e.canonical_name for e in memory1.entities}
        entities2 = {e.canonical_name for e in memory2.entities}
        return list(entities1.intersection(entities2))

    def _are_temporally_related(self, memory1: MemoryNode, memory2: MemoryNode) -> bool:
        """Check if two memories are temporally related."""
        if not memory1.event_time or not memory2.event_time:
            return False
        
        time_diff = abs((memory1.event_time - memory2.event_time).total_seconds())
        
        # Consider memories within 24 hours as temporally related
        return time_diff <= 24 * 3600

    def _detect_memory_conflicts(self, new_memories: List[MemoryNode], run: ConsolidationRun):
        """Detect conflicts between new and existing memories."""
        
        for new_memory in new_memories:
            for existing_memory in self.current_memories:
                conflicts = self._check_memory_pair_for_conflicts(new_memory, existing_memory)
                for conflict in conflicts:
                    self.detected_conflicts.append(conflict)
                    run.conflicts_detected += 1
                    logger.info(f"Conflict detected: {conflict.description}")

    def _check_memory_pair_for_conflicts(self, memory1: MemoryNode, memory2: MemoryNode) -> List[MemoryConflict]:
        """Check a pair of memories for conflicts."""
        conflicts = []
        
        # Check for factual contradictions
        factual_conflicts = self._detect_factual_contradictions(memory1, memory2)
        conflicts.extend(factual_conflicts)
        
        # Check for temporal inconsistencies
        temporal_conflicts = self._detect_temporal_inconsistencies(memory1, memory2)
        conflicts.extend(temporal_conflicts)
        
        # Check for preference evolution/contradictions
        preference_conflicts = self._detect_preference_conflicts(memory1, memory2)
        conflicts.extend(preference_conflicts)
        
        return conflicts

    def _detect_factual_contradictions(self, memory1: MemoryNode, memory2: MemoryNode) -> List[MemoryConflict]:
        """Detect contradictory facts between memories."""
        conflicts = []
        
        # Compare facts with same subject but different objects
        for fact1 in memory1.facts:
            for fact2 in memory2.facts:
                if (fact1.subject.lower() == fact2.subject.lower() and 
                    fact1.predicate.lower() == fact2.predicate.lower() and
                    fact1.object.lower() != fact2.object.lower()):
                    
                    # Check if these are truly contradictory
                    if self._are_facts_contradictory(fact1, fact2):
                        conflict = MemoryConflict(
                            id=str(uuid.uuid4()),
                            conflict_type=ConflictType.FACTUAL_CONTRADICTION,
                            memory1_id=memory1.id,
                            memory2_id=memory2.id,
                            description=f"Contradictory facts: '{fact1.subject} {fact1.predicate} {fact1.object}' vs '{fact2.subject} {fact2.predicate} {fact2.object}'",
                            severity=ConflictSeverity.MEDIUM,
                            confidence=0.8
                        )
                        conflicts.append(conflict)
        
        return conflicts

    def _are_facts_contradictory(self, fact1: Fact, fact2: Fact) -> bool:
        """Determine if two facts are actually contradictory."""
        # Simple heuristic - look for obvious opposites
        opposite_pairs = [
            ('like', 'dislike'), ('love', 'hate'), ('prefer', 'avoid'),
            ('yes', 'no'), ('true', 'false'), ('good', 'bad'),
            ('works', 'broken'), ('active', 'inactive')
        ]
        
        obj1_lower = fact1.object.lower()
        obj2_lower = fact2.object.lower()
        
        for pos, neg in opposite_pairs:
            if (pos in obj1_lower and neg in obj2_lower) or (neg in obj1_lower and pos in obj2_lower):
                return True
        
        return False

    def _detect_temporal_inconsistencies(self, memory1: MemoryNode, memory2: MemoryNode) -> List[MemoryConflict]:
        """Detect temporal inconsistencies between memories."""
        conflicts = []
        
        # Check if both memories claim to be the "latest" or "most recent" about the same thing
        if (memory1.event_time and memory2.event_time and 
            self._find_shared_entities(memory1, memory2)):
            
            content1_lower = memory1.content.lower()
            content2_lower = memory2.content.lower()
            
            latest_indicators = ['latest', 'most recent', 'current', 'now', 'today']
            
            if (any(indicator in content1_lower for indicator in latest_indicators) and
                any(indicator in content2_lower for indicator in latest_indicators)):
                
                time_diff = abs((memory1.event_time - memory2.event_time).days)
                
                if time_diff > 1:  # More than 1 day apart
                    conflict = MemoryConflict(
                        id=str(uuid.uuid4()),
                        conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                        memory1_id=memory1.id,
                        memory2_id=memory2.id,
                        description=f"Both memories claim to be 'latest' but are {time_diff} days apart",
                        severity=ConflictSeverity.LOW,
                        confidence=0.6
                    )
                    conflicts.append(conflict)
        
        return conflicts

    def _detect_preference_conflicts(self, memory1: MemoryNode, memory2: MemoryNode) -> List[MemoryConflict]:
        """Detect conflicting preferences or opinions."""
        conflicts = []
        
        # Compare emotional markers for the same entities
        for marker1 in memory1.emotional_markers:
            for marker2 in memory2.emotional_markers:
                if (marker1.entity.lower() == marker2.entity.lower() and
                    marker1.emotion_type == marker2.emotion_type and
                    marker1.polarity != marker2.polarity):
                    
                    # This could be preference evolution or contradiction
                    if memory1.event_time and memory2.event_time:
                        time_diff = abs((memory1.event_time - memory2.event_time).days)
                        
                        if time_diff > 30:  # More than 30 days - likely evolution
                            conflict_type = ConflictType.PREFERENCE_EVOLUTION
                            severity = ConflictSeverity.LOW
                        else:  # Recent contradiction
                            conflict_type = ConflictType.PREFERENCE_EVOLUTION  # Still evolution, just faster
                            severity = ConflictSeverity.MEDIUM
                    else:
                        conflict_type = ConflictType.PREFERENCE_EVOLUTION
                        severity = ConflictSeverity.LOW
                    
                    conflict = MemoryConflict(
                        id=str(uuid.uuid4()),
                        conflict_type=conflict_type,
                        memory1_id=memory1.id,
                        memory2_id=memory2.id,
                        description=f"Conflicting {marker1.emotion_type} about '{marker1.entity}': {marker1.polarity} vs {marker2.polarity}",
                        severity=severity,
                        confidence=0.7
                    )
                    conflicts.append(conflict)
        
        return conflicts

    def _update_entity_importance(self, memories: List[MemoryNode], run: ConsolidationRun):
        """Update entity importance scores based on usage patterns."""
        
        # Count entity mentions and contexts
        entity_contexts = defaultdict(list)
        
        for memory in memories + self.current_memories:
            for entity in memory.entities:
                key = (entity.canonical_name, entity.type)
                entity_contexts[key].append({
                    'memory_importance': memory.importance_score,
                    'memory_type': memory.type,
                    'event_time': memory.event_time
                })
        
        # Update importance scores
        for (canonical_name, entity_type), contexts in entity_contexts.items():
            if (canonical_name, entity_type) in self.entity_registry:
                entity = self.entity_registry[(canonical_name, entity_type)]
                
                # Calculate new importance based on:
                # 1. Frequency of mentions
                # 2. Importance of memories that mention it
                # 3. Recency of mentions
                # 4. Diversity of contexts
                
                mention_count = len(contexts)
                avg_memory_importance = sum(c['memory_importance'] for c in contexts) / mention_count
                
                # Recency factor
                recent_mentions = [c for c in contexts if c['event_time'] and 
                                   (datetime.now() - c['event_time']).days <= 30]
                recency_factor = len(recent_mentions) / mention_count if mention_count > 0 else 0
                
                # Context diversity
                memory_types = set(c['memory_type'] for c in contexts)
                diversity_factor = len(memory_types) / 4  # 4 possible memory types
                
                new_importance = min(1.0, (
                    0.3 * math.log(mention_count + 1) +
                    0.4 * avg_memory_importance +
                    0.2 * recency_factor +
                    0.1 * diversity_factor
                ))
                
                entity.importance_score = new_importance

    def _apply_decay_functions(self, run: ConsolidationRun):
        """Apply forgetting curves to older memories."""
        
        current_time = datetime.now()
        decayed_count = 0
        
        for memory in self.current_memories:
            if memory.event_time:
                days_old = (current_time - memory.event_time).days
                
                # Calculate decay based on memory type and characteristics
                stability = self._calculate_memory_stability(memory)
                
                # Ebbinghaus-inspired decay: retention = e^(-t/S)
                base_retention = math.exp(-days_old / stability)
                
                # Modifiers that fight decay
                access_boost = 1.0 + (getattr(memory, 'access_count', 0) * 0.1)
                importance_boost = 1.0 + (memory.importance_score * 0.5)
                relationship_boost = 1.0 + (len(memory.relationships) * 0.05)
                
                # Apply decay
                retention_rate = min(base_retention * access_boost * importance_boost * relationship_boost, 1.0)
                
                old_decay = memory.decay_factor
                memory.decay_factor = max(0.1, retention_rate)  # Don't completely forget
                
                if memory.decay_factor < old_decay * 0.9:  # Significant decay
                    decayed_count += 1
        
        run.memories_decayed = decayed_count
        logger.info(f"Applied decay to {decayed_count} memories")

    def _calculate_memory_stability(self, memory: MemoryNode) -> float:
        """Calculate how long this type of memory typically lasts."""
        base_stability = {
            MemoryType.SEMANTIC: 365,    # Facts last ~1 year base
            MemoryType.PROCEDURAL: 180,  # Procedures last ~6 months
            MemoryType.EPISODIC: 90,     # Events last ~3 months
            MemoryType.EMOTIONAL: 60     # Emotions fade in ~2 months
        }.get(memory.type, 90)
        
        # Adjust based on importance
        return base_stability * (1 + memory.importance_score)

    def _merge_similar_memories(self, memories: List[MemoryNode], run: ConsolidationRun) -> List[MemoryNode]:
        """Merge highly similar memories to reduce redundancy."""
        merged_memories = []
        processed_indices = set()
        
        for i, memory1 in enumerate(memories):
            if i in processed_indices:
                continue
            
            # Look for similar memories
            merge_candidates = [memory1]
            processed_indices.add(i)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                similarity = self._calculate_memory_similarity(memory1, memory2)
                
                if similarity > 0.8:  # High similarity threshold for merging
                    merge_candidates.append(memory2)
                    processed_indices.add(j)
            
            if len(merge_candidates) > 1:
                # Merge the similar memories
                merged = self._merge_memory_group(merge_candidates)
                merged_memories.append(merged)
                run.memories_merged += len(merge_candidates) - 1
                logger.debug(f"Merged {len(merge_candidates)} similar memories")
            else:
                merged_memories.append(memory1)
        
        return merged_memories

    def _calculate_memory_similarity(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Calculate similarity between two memories."""
        # Content similarity
        content_sim = self._calculate_text_similarity(
            memory1.content.lower(), 
            memory2.content.lower()
        )
        
        # Entity overlap
        entities1 = {e.canonical_name for e in memory1.entities}
        entities2 = {e.canonical_name for e in memory2.entities}
        
        if entities1 or entities2:
            entity_sim = len(entities1.intersection(entities2)) / len(entities1.union(entities2))
        else:
            entity_sim = 0.0
        
        # Type similarity
        type_sim = 1.0 if memory1.type == memory2.type else 0.5
        
        # Temporal proximity
        temporal_sim = 1.0
        if memory1.event_time and memory2.event_time:
            time_diff = abs((memory1.event_time - memory2.event_time).days)
            temporal_sim = max(0.0, 1.0 - time_diff / 30.0)  # 30-day window
        
        # Weighted combination
        return (0.5 * content_sim + 0.3 * entity_sim + 0.1 * type_sim + 0.1 * temporal_sim)

    def _merge_memory_group(self, memories: List[MemoryNode]) -> MemoryNode:
        """Merge a group of similar memories into one consolidated memory."""
        # Use the most important memory as the base
        base_memory = max(memories, key=lambda m: m.importance_score)
        
        # Combine content
        combined_content = base_memory.content
        
        # If there are significant differences, append them
        for memory in memories:
            if memory.id != base_memory.id:
                content_sim = self._calculate_text_similarity(
                    base_memory.content.lower(),
                    memory.content.lower()
                )
                if content_sim < 0.9:  # Not too similar
                    combined_content += f"\n\nAdditional context: {memory.content}"
        
        # Merge entities (deduplicate)
        all_entities = []
        for memory in memories:
            all_entities.extend(memory.entities)
        
        merged_entities = self._deduplicate_entities(all_entities)
        
        # Merge facts, relationships, emotional markers
        all_facts = []
        all_relationships = []
        all_emotional_markers = []
        
        for memory in memories:
            all_facts.extend(memory.facts)
            all_relationships.extend(memory.relationships)
            all_emotional_markers.extend(memory.emotional_markers)
        
        # Calculate merged importance (take the maximum)
        merged_importance = max(m.importance_score for m in memories)
        
        # Create merged memory
        merged_memory = MemoryNode(
            id=base_memory.id,  # Keep the base ID
            type=base_memory.type,
            content=combined_content,
            event_time=base_memory.event_time,
            importance_score=merged_importance,
            source="consolidation_merge",
            source_file=base_memory.source_file,
            entities=merged_entities,
            facts=all_facts,
            relationships=all_relationships,
            emotional_markers=all_emotional_markers,
            metadata={
                'merged_from': [m.id for m in memories],
                'merge_timestamp': datetime.now().isoformat()
            }
        )
        
        return merged_memory

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on canonical names."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.canonical_name, entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

    def get_consolidation_summary(self, run: ConsolidationRun) -> Dict[str, Any]:
        """Get a summary of the consolidation run."""
        return {
            'run_id': run.id,
            'target_date': run.target_date.isoformat() if run.target_date else None,
            'duration_seconds': (run.end_time - run.start_time).total_seconds() if run.end_time else None,
            'status': run.status,
            'metrics': {
                'memories_processed': run.memories_processed,
                'entities_created': run.entities_created,
                'entities_updated': run.entities_updated,
                'relations_created': run.relations_created,
                'conflicts_detected': run.conflicts_detected,
                'memories_merged': run.memories_merged,
                'memories_decayed': run.memories_decayed
            },
            'total_memories_after': len(self.current_memories),
            'total_entities': len(self.entity_registry),
            'conflicts': [asdict(c) for c in self.detected_conflicts],
            'error_message': run.error_message
        }

    def save_consolidation_results(self, output_dir: str, run: ConsolidationRun) -> Dict[str, str]:
        """Save consolidation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        files_created = {}
        
        # Save consolidated memories
        memories_file = os.path.join(output_dir, f"consolidated_memories_{run.target_date}.json")
        memories_data = [asdict(m) for m in self.current_memories]
        
        # Handle datetime serialization
        for memory_dict in memories_data:
            if memory_dict['event_time']:
                memory_dict['event_time'] = memory_dict['event_time'].isoformat()
        
        with open(memories_file, 'w', encoding='utf-8') as f:
            json.dump(memories_data, f, indent=2, ensure_ascii=False)
        files_created['memories'] = memories_file
        
        # Save entities
        entities_file = os.path.join(output_dir, f"entities_{run.target_date}.json")
        entities_data = [asdict(e) for e in self.entity_registry.values()]
        
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        files_created['entities'] = entities_file
        
        # Save conflicts
        if self.detected_conflicts:
            conflicts_file = os.path.join(output_dir, f"conflicts_{run.target_date}.json")
            conflicts_data = []
            
            for conflict in self.detected_conflicts:
                conflict_dict = asdict(conflict)
                conflict_dict['detected_at'] = conflict_dict['detected_at'].isoformat()
                conflicts_data.append(conflict_dict)
            
            with open(conflicts_file, 'w', encoding='utf-8') as f:
                json.dump(conflicts_data, f, indent=2, ensure_ascii=False)
            files_created['conflicts'] = conflicts_file
        
        # Save run summary
        summary_file = os.path.join(output_dir, f"run_summary_{run.target_date}.json")
        summary = self.get_consolidation_summary(run)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        files_created['summary'] = summary_file
        
        return files_created


class PgVectorConsolidator:
    """Database-backed consolidation using pgvector for semantic clustering."""

    DB_DSN = os.environ.get(
        "DATABASE_URL",
        "postgresql://vex:vex_memory_dev@localhost:5433/vex_memory",
    )
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-minilm")

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or self.DB_DSN
        import psycopg2
        from psycopg2.extras import RealDictCursor
        self._psycopg2 = psycopg2
        self._RealDictCursor = RealDictCursor

    # -- helpers ---------------------------------------------------------------

    def _conn(self):
        return self._psycopg2.connect(self.dsn, cursor_factory=self._RealDictCursor)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        import httpx
        try:
            r = httpx.post(
                f"{self.OLLAMA_URL}/api/embeddings",
                json={"model": self.EMBED_MODEL, "prompt": text[:8000]},
                timeout=30.0,
            )
            if r.status_code == 200:
                return r.json().get("embedding")
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        return None

    # -- public API ------------------------------------------------------------

    def consolidate_related_memories(
        self,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
    ) -> Dict[str, Any]:
        """Find clusters of semantically similar memories and create summaries.

        Uses pgvector cosine similarity to find clusters, creates a summary
        memory for each cluster, and lowers the importance of originals.
        """
        conn = self._conn()
        try:
            cur = conn.cursor()

            # Fetch all memories that have embeddings and aren't already summaries
            cur.execute(
                """SELECT id::text, content, importance_score, type::text, embedding::text
                   FROM memory_nodes
                   WHERE embedding IS NOT NULL
                     AND (metadata->>'is_consolidation_summary')::boolean IS NOT TRUE
                   ORDER BY importance_score DESC"""
            )
            rows = cur.fetchall()
            if len(rows) < min_cluster_size:
                return {"clusters_created": 0, "memories_affected": 0, "summaries": []}

            # Build adjacency via pairwise similarity in DB
            # For each memory, find its neighbours above threshold
            id_to_row = {r["id"]: r for r in rows}
            all_ids = list(id_to_row.keys())

            # Use DB to compute pairwise similarities efficiently
            neighbours: Dict[str, Set[str]] = defaultdict(set)
            for row in rows:
                cur.execute(
                    """SELECT id::text, 1.0 - (embedding <=> %s::vector) AS sim
                       FROM memory_nodes
                       WHERE embedding IS NOT NULL
                         AND id != %s
                         AND (metadata->>'is_consolidation_summary')::boolean IS NOT TRUE
                         AND 1.0 - (embedding <=> %s::vector) >= %s""",
                    (row["embedding"], row["id"], row["embedding"], similarity_threshold),
                )
                for nb in cur.fetchall():
                    if nb["id"] in id_to_row:
                        neighbours[row["id"]].add(nb["id"])
                        neighbours[nb["id"]].add(row["id"])

            # Greedy clustering: pick node with most neighbours, form cluster
            clustered: Set[str] = set()
            clusters: List[List[str]] = []

            sorted_ids = sorted(all_ids, key=lambda i: len(neighbours.get(i, set())), reverse=True)
            for seed in sorted_ids:
                if seed in clustered:
                    continue
                nbs = neighbours.get(seed, set()) - clustered
                cluster = {seed} | nbs
                if len(cluster) < min_cluster_size:
                    continue
                clusters.append(list(cluster))
                clustered |= cluster

            # Create summary memories for each cluster
            summaries_created = []
            total_affected = 0

            for cluster_ids in clusters:
                cluster_rows = [id_to_row[i] for i in cluster_ids]
                max_importance = max(r["importance_score"] for r in cluster_rows)

                # Build summary text
                key_points = []
                for r in cluster_rows:
                    # Take first 150 chars as key point
                    point = r["content"][:150].strip()
                    if len(r["content"]) > 150:
                        point += "..."
                    key_points.append(f"- {point}")

                summary_content = (
                    f"Summary of {len(cluster_ids)} related memories:\n"
                    + "\n".join(key_points)
                )
                summary_importance = min(1.0, max_importance * 1.2)

                # Get embedding for summary
                summary_embedding = self._get_embedding(summary_content)

                summary_id = str(uuid.uuid4())
                import json as _json
                meta = _json.dumps({
                    "is_consolidation_summary": True,
                    "source_memory_ids": cluster_ids,
                    "consolidated_at": datetime.now().isoformat(),
                })

                if summary_embedding:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata, embedding)
                           VALUES (%s, 'semantic'::memory_type, %s, %s, 'consolidation', %s::jsonb, %s::vector)""",
                        (summary_id, summary_content, summary_importance, meta, str(summary_embedding)),
                    )
                else:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata)
                           VALUES (%s, 'semantic'::memory_type, %s, %s, 'consolidation', %s::jsonb)""",
                        (summary_id, summary_content, summary_importance, meta),
                    )

                # Lower importance of originals
                lowered_importance = max_importance * 0.5
                for mem_id in cluster_ids:
                    cur.execute(
                        """UPDATE memory_nodes
                           SET importance_score = LEAST(importance_score, %s),
                               metadata = metadata || %s::jsonb
                           WHERE id = %s""",
                        (
                            lowered_importance,
                            _json.dumps({"consolidated_into": summary_id}),
                            mem_id,
                        ),
                    )

                total_affected += len(cluster_ids)
                summaries_created.append({
                    "summary_id": summary_id,
                    "source_count": len(cluster_ids),
                    "importance": summary_importance,
                })

            conn.commit()
            return {
                "clusters_created": len(clusters),
                "memories_affected": total_affected,
                "summaries": summaries_created,
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"consolidate_related_memories failed: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def consolidate_by_topic(self) -> Dict[str, Any]:
        """Group memories by entity/topic and create topic summaries.

        Finds entities with multiple memory mentions and creates a single
        topic summary for each that doesn't already have one.
        """
        conn = self._conn()
        try:
            cur = conn.cursor()

            # Find entities with >= 3 mentions that don't already have a topic summary
            cur.execute(
                """SELECT e.id::text AS entity_id, e.canonical_name, e.name,
                          COUNT(mem.memory_id) AS cnt
                   FROM entities e
                   JOIN memory_entity_mentions mem ON e.id = mem.entity_id
                   GROUP BY e.id, e.canonical_name, e.name
                   HAVING COUNT(mem.memory_id) >= 3
                   ORDER BY cnt DESC"""
            )
            entities = cur.fetchall()

            summaries_created = []
            for ent in entities:
                ent_name = ent["canonical_name"] or ent["name"]

                # Check if we already have a topic summary for this entity
                import json as _json
                cur.execute(
                    """SELECT id FROM memory_nodes
                       WHERE source = 'topic_consolidation'
                         AND metadata->>'topic_entity' = %s
                       LIMIT 1""",
                    (ent_name,),
                )
                if cur.fetchone():
                    continue

                # Fetch memories for this entity
                cur.execute(
                    """SELECT m.id::text, m.content, m.importance_score, m.type::text
                       FROM memory_nodes m
                       JOIN memory_entity_mentions mem ON m.id = mem.memory_id
                       WHERE mem.entity_id = %s
                       ORDER BY m.importance_score DESC
                       LIMIT 20""",
                    (ent["entity_id"],),
                )
                memories = cur.fetchall()
                if len(memories) < 3:
                    continue

                max_importance = max(m["importance_score"] for m in memories)

                key_points = []
                for m in memories:
                    point = m["content"][:150].strip()
                    if len(m["content"]) > 150:
                        point += "..."
                    key_points.append(f"- {point}")

                summary_content = (
                    f"Topic summary for '{ent_name}' ({len(memories)} memories):\n"
                    + "\n".join(key_points)
                )
                summary_importance = min(1.0, max_importance * 1.15)
                summary_embedding = self._get_embedding(summary_content)

                summary_id = str(uuid.uuid4())
                meta = _json.dumps({
                    "is_consolidation_summary": True,
                    "topic_entity": ent_name,
                    "source_memory_count": len(memories),
                    "consolidated_at": datetime.now().isoformat(),
                })

                if summary_embedding:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata, embedding)
                           VALUES (%s, 'semantic'::memory_type, %s, %s, 'topic_consolidation', %s::jsonb, %s::vector)""",
                        (summary_id, summary_content, summary_importance, meta, str(summary_embedding)),
                    )
                else:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata)
                           VALUES (%s, 'semantic'::memory_type, %s, %s, 'topic_consolidation', %s::jsonb)""",
                        (summary_id, summary_content, summary_importance, meta),
                    )

                summaries_created.append({
                    "summary_id": summary_id,
                    "topic": ent_name,
                    "source_count": len(memories),
                    "importance": summary_importance,
                })

            conn.commit()
            return {
                "topics_summarized": len(summaries_created),
                "summaries": summaries_created,
            }
        except Exception as e:
            conn.rollback()
            logger.error(f"consolidate_by_topic failed: {e}", exc_info=True)
            raise
        finally:
            conn.close()


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidate daily memories")
    parser.add_argument("daily_file", help="Path to daily memory file")
    parser.add_argument("-d", "--date", help="Target date (YYYY-MM-DD)", required=True)
    parser.add_argument("-e", "--existing", help="Path to existing memories JSON file")
    parser.add_argument("-o", "--output", help="Output directory", default="./consolidation_output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse target date
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    except ValueError:
        print("Error: Date must be in YYYY-MM-DD format")
        exit(1)
    
    # Load existing memories if provided
    existing_memories = []
    if args.existing and os.path.exists(args.existing):
        with open(args.existing, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
            # Note: This would need proper deserialization in a full implementation
    
    # Run consolidation
    consolidator = MemoryConsolidator()
    run = consolidator.consolidate_daily_memories(target_date, args.daily_file, existing_memories)
    
    # Save results
    files_created = consolidator.save_consolidation_results(args.output, run)
    
    # Print summary
    summary = consolidator.get_consolidation_summary(run)
    print("\nConsolidation Summary:")
    print(json.dumps(summary, indent=2, default=str))