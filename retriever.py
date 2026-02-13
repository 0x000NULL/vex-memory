"""
Vex Memory Retrieval Layer v2.0
===============================

Context assembly engine that retrieves relevant memories for conversations and queries.

This module provides multiple retrieval strategies and assembles optimal context 
windows from the memory database. It serves as the primary interface between the
AI agent and the structured memory system.

Author: Vex
Date: February 13, 2026
"""

import os
import re
import json
import logging
import math
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

from extractor import MemoryNode, Entity, MemoryType, EntityType
from consolidator import MemoryConsolidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    ENTITY_BASED = "entity_based"
    ASSOCIATIVE = "associative"
    PROCEDURAL = "procedural"
    HYBRID = "hybrid"


class QueryIntent(Enum):
    FACTUAL = "factual"          # "What do I know about X?"
    TEMPORAL = "temporal"        # "What happened last week?"
    PROCEDURAL = "procedural"    # "How do I do X?"
    CONTEXTUAL = "contextual"    # Context for ongoing conversation
    ASSOCIATIVE = "associative"  # "What's related to X?"


@dataclass
class ContextWindow:
    """Represents a context window assembled for a query."""
    query: str
    strategy: RetrievalStrategy
    memories: List[MemoryNode]
    total_tokens: int
    relevance_scores: List[float]
    assembly_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_formatted_context(self, max_tokens: Optional[int] = None) -> str:
        """Format the context window as text."""
        if max_tokens and self.total_tokens > max_tokens:
            # Truncate memories to fit token budget
            memories = self._truncate_to_token_budget(max_tokens)
        else:
            memories = self.memories
        
        context_parts = []
        
        for i, memory in enumerate(memories):
            relevance = self.relevance_scores[i] if i < len(self.relevance_scores) else 0.0
            
            # Format memory with metadata
            formatted = self._format_memory_for_context(memory, relevance)
            context_parts.append(formatted)
        
        return "\n\n".join(context_parts)
    
    def _truncate_to_token_budget(self, max_tokens: int) -> List[MemoryNode]:
        """Truncate memories to fit within token budget."""
        # Simple approximation: 4 chars per token
        current_chars = 0
        selected_memories = []
        
        for i, memory in enumerate(self.memories):
            memory_chars = len(memory.content)
            if current_chars + memory_chars <= max_tokens * 4:
                selected_memories.append(memory)
                current_chars += memory_chars
            else:
                break
        
        return selected_memories
    
    def _format_memory_for_context(self, memory: MemoryNode, relevance: float) -> str:
        """Format a single memory for inclusion in context."""
        timestamp = ""
        if memory.event_time:
            timestamp = f" ({memory.event_time.strftime('%Y-%m-%d')})"
        
        memory_type = memory.type.value.upper()
        
        # Add relevance indicator for high-relevance memories
        relevance_indicator = ""
        if relevance > 0.8:
            relevance_indicator = " [HIGHLY RELEVANT]"
        elif relevance > 0.6:
            relevance_indicator = " [RELEVANT]"
        
        return f"[{memory_type}{timestamp}]{relevance_indicator}\n{memory.content}"


@dataclass
class QueryContext:
    """Context for a memory query."""
    query: str
    intent: Optional[QueryIntent] = None
    conversation_history: List[str] = field(default_factory=list)
    mentioned_entities: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None
    max_tokens: int = 4000
    strategies: List[RetrievalStrategy] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """Main retrieval engine for memory queries."""
    
    def __init__(self, memory_file: Optional[str] = None, fallback_to_files: bool = True):
        """
        Initialize the retriever.
        
        Args:
            memory_file: Path to JSON file containing memories (for testing)
            fallback_to_files: Whether to fall back to file-based search
        """
        self.memory_file = memory_file
        self.fallback_to_files = fallback_to_files
        
        # Load memories
        self.memories: List[MemoryNode] = []
        self.entity_index: Dict[str, List[MemoryNode]] = defaultdict(list)
        self.temporal_index: Dict[date, List[MemoryNode]] = defaultdict(list)
        self.type_index: Dict[MemoryType, List[MemoryNode]] = defaultdict(list)
        
        if memory_file and os.path.exists(memory_file):
            self._load_memories_from_file(memory_file)
        
        # Retrieval configuration
        self.min_relevance_threshold = 0.3
        self.max_memories_per_strategy = 15
        self.semantic_similarity_threshold = 0.6  # Placeholder for when embeddings are available
        
        logger.info(f"Memory retriever initialized with {len(self.memories)} memories")

    def _load_memories_from_file(self, file_path: str):
        """Load memories from JSON file."""
        logger.info(f"Loading memories from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Convert dictionaries back to MemoryNode objects
            # Note: This is a simplified version - full implementation would use proper deserialization
            for mem_dict in memory_data:
                memory = self._dict_to_memory_node(mem_dict)
                if memory:
                    self.memories.append(memory)
            
            # Build indices
            self._build_indices()
            
            logger.info(f"Loaded {len(self.memories)} memories")
            
        except Exception as e:
            logger.error(f"Failed to load memories from {file_path}: {e}")

    def _dict_to_memory_node(self, mem_dict: Dict) -> Optional[MemoryNode]:
        """Convert dictionary to MemoryNode (simplified version)."""
        try:
            # This is a basic conversion - full version would handle all nested objects
            memory = MemoryNode(
                id=mem_dict.get('id', ''),
                type=MemoryType(mem_dict.get('type', 'semantic')),
                content=mem_dict.get('content', ''),
                importance_score=mem_dict.get('importance_score', 0.5),
                source=mem_dict.get('source', 'unknown')
            )
            
            # Handle event_time
            if mem_dict.get('event_time'):
                try:
                    memory.event_time = datetime.fromisoformat(mem_dict['event_time'])
                except:
                    pass
            
            return memory
            
        except Exception as e:
            logger.warning(f"Failed to convert memory dict: {e}")
            return None

    def _build_indices(self):
        """Build search indices for efficient retrieval."""
        logger.debug("Building search indices...")
        
        for memory in self.memories:
            # Type index
            self.type_index[memory.type].append(memory)
            
            # Temporal index
            if memory.event_time:
                memory_date = memory.event_time.date()
                self.temporal_index[memory_date].append(memory)
            
            # Entity index (simplified - would use actual entities in full version)
            content_lower = memory.content.lower()
            words = re.findall(r'\b\w+\b', content_lower)
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self.entity_index[word].append(memory)
        
        logger.debug(f"Built indices: {len(self.type_index)} types, {len(self.temporal_index)} dates, {len(self.entity_index)} terms")

    def query(self, query_context: QueryContext) -> ContextWindow:
        """
        Main query interface - retrieve relevant memories for a query.
        
        Args:
            query_context: Context and parameters for the query
            
        Returns:
            ContextWindow with assembled memories
        """
        logger.info(f"Processing query: {query_context.query[:100]}...")
        
        # Determine query intent if not provided
        if not query_context.intent:
            query_context.intent = self._infer_query_intent(query_context.query)
        
        # Determine strategies if not provided
        if not query_context.strategies:
            query_context.strategies = self._select_strategies(query_context)
        
        # Retrieve memories using multiple strategies
        candidate_memories = []
        
        for strategy in query_context.strategies:
            strategy_memories = self._retrieve_with_strategy(strategy, query_context)
            candidate_memories.extend(strategy_memories)
        
        # If no results and fallback enabled, try file-based search
        if not candidate_memories and self.fallback_to_files:
            candidate_memories = self._fallback_file_search(query_context)
        
        # Deduplicate and score
        unique_memories = self._deduplicate_memories(candidate_memories)
        scored_memories = self._score_memories(unique_memories, query_context)
        
        # Select top memories within token budget
        selected_memories, relevance_scores = self._select_optimal_memories(
            scored_memories, query_context.max_tokens
        )
        
        # Estimate total tokens
        total_tokens = self._estimate_token_count(selected_memories)
        
        # Create context window
        context_window = ContextWindow(
            query=query_context.query,
            strategy=RetrievalStrategy.HYBRID,
            memories=selected_memories,
            total_tokens=total_tokens,
            relevance_scores=relevance_scores,
            assembly_metadata={
                'intent': query_context.intent.value if query_context.intent else None,
                'strategies_used': [s.value for s in query_context.strategies],
                'total_candidates': len(candidate_memories),
                'unique_candidates': len(unique_memories),
                'fallback_used': len(candidate_memories) == 0 and self.fallback_to_files
            }
        )
        
        logger.info(f"Assembled context with {len(selected_memories)} memories ({total_tokens} tokens)")
        
        return context_window

    def _infer_query_intent(self, query: str) -> QueryIntent:
        """Infer the intent behind a query."""
        query_lower = query.lower()
        
        # Temporal queries
        temporal_indicators = [
            'when', 'yesterday', 'today', 'tomorrow', 'last week', 'this week',
            'last month', 'recently', 'ago', 'since', 'before', 'after'
        ]
        if any(indicator in query_lower for indicator in temporal_indicators):
            return QueryIntent.TEMPORAL
        
        # Procedural queries
        procedural_indicators = [
            'how to', 'how do', 'steps to', 'process', 'procedure', 'method',
            'way to', 'guide', 'tutorial', 'instructions'
        ]
        if any(indicator in query_lower for indicator in procedural_indicators):
            return QueryIntent.PROCEDURAL
        
        # Associative queries
        associative_indicators = [
            'related to', 'connected', 'associated', 'similar', 'like',
            'about', 'regarding', 'concerning'
        ]
        if any(indicator in query_lower for indicator in associative_indicators):
            return QueryIntent.ASSOCIATIVE
        
        # Factual queries (default)
        return QueryIntent.FACTUAL

    def _select_strategies(self, query_context: QueryContext) -> List[RetrievalStrategy]:
        """Select appropriate retrieval strategies based on query context."""
        strategies = []
        
        if query_context.intent == QueryIntent.TEMPORAL:
            strategies = [RetrievalStrategy.TEMPORAL, RetrievalStrategy.KEYWORD]
        elif query_context.intent == QueryIntent.PROCEDURAL:
            strategies = [RetrievalStrategy.PROCEDURAL, RetrievalStrategy.KEYWORD]
        elif query_context.intent == QueryIntent.ASSOCIATIVE:
            strategies = [RetrievalStrategy.ASSOCIATIVE, RetrievalStrategy.ENTITY_BASED]
        else:
            # Default factual queries
            strategies = [RetrievalStrategy.KEYWORD, RetrievalStrategy.ENTITY_BASED]
        
        # Always add semantic search if available (placeholder)
        if hasattr(self, 'semantic_model'):  # Would check for embedding model
            strategies.append(RetrievalStrategy.SEMANTIC)
        
        return strategies

    def _retrieve_with_strategy(self, strategy: RetrievalStrategy, 
                                query_context: QueryContext) -> List[MemoryNode]:
        """Retrieve memories using a specific strategy."""
        
        if strategy == RetrievalStrategy.KEYWORD:
            return self._keyword_search(query_context.query)
        elif strategy == RetrievalStrategy.TEMPORAL:
            return self._temporal_search(query_context)
        elif strategy == RetrievalStrategy.ENTITY_BASED:
            return self._entity_based_search(query_context)
        elif strategy == RetrievalStrategy.PROCEDURAL:
            return self._procedural_search(query_context.query)
        elif strategy == RetrievalStrategy.ASSOCIATIVE:
            return self._associative_search(query_context)
        elif strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_search(query_context.query)  # Placeholder
        else:
            return []

    def _keyword_search(self, query: str) -> List[MemoryNode]:
        """Search memories by keyword matching."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words
        
        if not query_words:
            return []
        
        # Score memories by keyword overlap
        memory_scores = []
        
        for memory in self.memories:
            content_words = set(re.findall(r'\b\w+\b', memory.content.lower()))
            
            # Calculate overlap
            overlap = query_words.intersection(content_words)
            if overlap:
                score = len(overlap) / len(query_words)
                memory_scores.append((memory, score))
        
        # Sort by score and return top results
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in memory_scores[:self.max_memories_per_strategy]]

    def _temporal_search(self, query_context: QueryContext) -> List[MemoryNode]:
        """Search memories by temporal criteria."""
        temporal_filters = self._parse_temporal_query(query_context.query)
        
        if not temporal_filters:
            return []
        
        relevant_memories = []
        
        for date_filter in temporal_filters:
            if isinstance(date_filter, date):
                # Specific date
                relevant_memories.extend(self.temporal_index.get(date_filter, []))
            elif isinstance(date_filter, tuple):
                # Date range
                start_date, end_date = date_filter
                for memory_date, memories in self.temporal_index.items():
                    if start_date <= memory_date <= end_date:
                        relevant_memories.extend(memories)
        
        # Sort by recency and importance
        relevant_memories.sort(
            key=lambda m: (m.event_time or datetime.min, m.importance_score),
            reverse=True
        )
        
        return relevant_memories[:self.max_memories_per_strategy]

    def _parse_temporal_query(self, query: str) -> List[Union[date, Tuple[date, date]]]:
        """Parse temporal expressions from query."""
        # Simplified temporal parsing
        query_lower = query.lower()
        today = date.today()
        
        filters = []
        
        if 'yesterday' in query_lower:
            filters.append(today - timedelta(days=1))
        elif 'today' in query_lower:
            filters.append(today)
        elif 'last week' in query_lower:
            start = today - timedelta(days=today.weekday() + 7)
            end = today - timedelta(days=today.weekday() + 1)
            filters.append((start, end))
        elif 'this week' in query_lower:
            start = today - timedelta(days=today.weekday())
            end = today + timedelta(days=6 - today.weekday())
            filters.append((start, end))
        elif 'last month' in query_lower:
            # Approximate last month
            start = today.replace(day=1) - timedelta(days=1)
            start = start.replace(day=1)
            end = today.replace(day=1) - timedelta(days=1)
            filters.append((start, end))
        
        return filters

    def _entity_based_search(self, query_context: QueryContext) -> List[MemoryNode]:
        """Search memories by entity mentions."""
        # Extract potential entities from query
        entities = self._extract_query_entities(query_context.query)
        entities.extend(query_context.mentioned_entities)
        
        if not entities:
            return []
        
        relevant_memories = []
        
        for entity in entities:
            entity_lower = entity.lower()
            # Simple entity matching - would use proper entity resolution in full version
            for word in entity_lower.split():
                if word in self.entity_index:
                    relevant_memories.extend(self.entity_index[word])
        
        # Deduplicate by ID and score by entity relevance
        seen_ids = set()
        unique_memories = []
        for memory in relevant_memories:
            if memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)
        
        # Score by number of matching entities
        scored_memories = []
        for memory in unique_memories:
            content_lower = memory.content.lower()
            entity_count = sum(1 for entity in entities if entity.lower() in content_lower)
            score = entity_count / len(entities)
            scored_memories.append((memory, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in scored_memories[:self.max_memories_per_strategy]]

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from query text."""
        # Simple capitalized word extraction
        entities = []
        
        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)
        
        # Known entity patterns
        known_entities = ['python', 'javascript', 'postgresql', 'docker', 'linux', 'ethan', 'seth']
        query_lower = query.lower()
        for entity in known_entities:
            if entity in query_lower:
                entities.append(entity)
        
        return list(set(entities))

    def _procedural_search(self, query: str) -> List[MemoryNode]:
        """Search for procedural memories (how-to information)."""
        procedural_memories = self.type_index[MemoryType.PROCEDURAL]
        
        # Also look for procedural content in other memory types
        all_procedural = list(procedural_memories)
        
        for memory in self.memories:
            if memory not in procedural_memories:
                content_lower = memory.content.lower()
                if any(indicator in content_lower for indicator in 
                       ['step', 'process', 'procedure', 'how to', 'method', 'guide']):
                    all_procedural.append(memory)
        
        # Filter by keyword relevance to query
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        scored_memories = []
        for memory in all_procedural:
            content_words = set(re.findall(r'\b\w+\b', memory.content.lower()))
            overlap = query_keywords.intersection(content_words)
            if overlap:
                score = len(overlap) / len(query_keywords)
                scored_memories.append((memory, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in scored_memories[:self.max_memories_per_strategy]]

    def _associative_search(self, query_context: QueryContext) -> List[MemoryNode]:
        """Search for associatively related memories."""
        # Start with entity-based search
        base_memories = self._entity_based_search(query_context)
        
        # Find memories that share entities with base memories
        associated_memories = []
        
        for base_memory in base_memories[:5]:  # Limit to top 5 to avoid explosion
            # Find memories with similar content (simplified similarity)
            for memory in self.memories:
                if memory != base_memory:
                    similarity = self._calculate_content_similarity(base_memory.content, memory.content)
                    if similarity > 0.3:
                        associated_memories.append((memory, similarity))
        
        # Sort by similarity and return unique memories
        associated_memories.sort(key=lambda x: x[1], reverse=True)
        unique_associated = []
        seen_ids = set()
        
        for memory, similarity in associated_memories:
            if memory.id not in seen_ids:
                unique_associated.append(memory)
                seen_ids.add(memory.id)
        
        return unique_associated[:self.max_memories_per_strategy]

    def _semantic_search(self, query: str) -> List[MemoryNode]:
        """Placeholder for semantic similarity search using embeddings."""
        # This would use actual embedding similarity when the full system is deployed
        logger.debug("Semantic search not yet implemented (requires embedding model)")
        return []

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (Jaccard coefficient)."""
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _fallback_file_search(self, query_context: QueryContext) -> List[MemoryNode]:
        """Fallback to file-based search when no structured memories match."""
        logger.info("Falling back to file-based search")
        
        # This would search the original markdown files
        # For now, return empty list
        return []

    def _deduplicate_memories(self, memories: List[MemoryNode]) -> List[MemoryNode]:
        """Remove duplicate memories from list."""
        seen_ids = set()
        unique_memories = []
        
        for memory in memories:
            if memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)
        
        return unique_memories

    def _score_memories(self, memories: List[MemoryNode], query_context: QueryContext) -> List[Tuple[MemoryNode, float]]:
        """Score memories for relevance to the query."""
        scored_memories = []
        
        for memory in memories:
            relevance_score = self._calculate_relevance_score(memory, query_context)
            
            if relevance_score >= self.min_relevance_threshold:
                scored_memories.append((memory, relevance_score))
        
        # Sort by relevance score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories

    def _calculate_relevance_score(self, memory: MemoryNode, query_context: QueryContext) -> float:
        """Calculate relevance score for a memory given the query context."""
        score = 0.0
        
        # Base importance score
        score += memory.importance_score * 0.3
        
        # Content similarity to query
        content_sim = self._calculate_content_similarity(memory.content, query_context.query)
        score += content_sim * 0.4
        
        # Recency factor (more recent is slightly better)
        if memory.event_time:
            days_old = (datetime.now() - memory.event_time).days
            recency_factor = math.exp(-days_old / 60.0)  # 60-day half-life
            score += recency_factor * 0.1
        
        # Memory type relevance
        type_bonus = 0.0
        if query_context.intent == QueryIntent.PROCEDURAL and memory.type == MemoryType.PROCEDURAL:
            type_bonus = 0.2
        elif query_context.intent == QueryIntent.TEMPORAL and memory.type == MemoryType.EPISODIC:
            type_bonus = 0.1
        
        score += type_bonus
        
        # Conversation context relevance
        if query_context.conversation_history:
            recent_context = ' '.join(query_context.conversation_history[-3:])  # Last 3 messages
            context_sim = self._calculate_content_similarity(memory.content, recent_context)
            score += context_sim * 0.1
        
        return min(1.0, score)

    def _select_optimal_memories(self, scored_memories: List[Tuple[MemoryNode, float]], 
                                 max_tokens: int) -> Tuple[List[MemoryNode], List[float]]:
        """Select optimal set of memories within token budget."""
        selected_memories = []
        relevance_scores = []
        current_tokens = 0
        
        # Estimate tokens per memory (rough approximation: 4 chars per token)
        for memory, relevance in scored_memories:
            memory_tokens = len(memory.content) // 4
            
            if current_tokens + memory_tokens <= max_tokens:
                selected_memories.append(memory)
                relevance_scores.append(relevance)
                current_tokens += memory_tokens
            else:
                break
        
        return selected_memories, relevance_scores

    def _estimate_token_count(self, memories: List[MemoryNode]) -> int:
        """Estimate total token count for memories."""
        total_chars = sum(len(memory.content) for memory in memories)
        return total_chars // 4  # Rough approximation

    def get_conversation_context(self, current_message: str, 
                                conversation_history: List[str] = None,
                                max_tokens: int = 4000) -> ContextWindow:
        """
        Assemble context for an ongoing conversation.
        
        Args:
            current_message: The current message being processed
            conversation_history: Recent conversation history
            max_tokens: Maximum tokens for context
            
        Returns:
            ContextWindow optimized for conversation
        """
        # Build query context for conversation
        query_context = QueryContext(
            query=current_message,
            conversation_history=conversation_history or [],
            max_tokens=max_tokens,
            intent=QueryIntent.CONTEXTUAL
        )
        
        # Use multiple strategies for conversation context
        query_context.strategies = [
            RetrievalStrategy.KEYWORD,
            RetrievalStrategy.ENTITY_BASED,
            RetrievalStrategy.TEMPORAL  # Recent memories
        ]
        
        return self.query(query_context)

    def get_recent_important_memories(self, days: int = 7, max_count: int = 10) -> List[MemoryNode]:
        """Get recent important memories for proactive context."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_memories = []
        for memory in self.memories:
            if (memory.event_time and memory.event_time >= cutoff_date and 
                memory.importance_score > 0.6):
                recent_memories.append(memory)
        
        # Sort by combination of recency and importance
        recent_memories.sort(
            key=lambda m: (m.importance_score, m.event_time or datetime.min),
            reverse=True
        )
        
        return recent_memories[:max_count]

    def search_by_entity(self, entity_name: str, max_count: int = 10) -> List[MemoryNode]:
        """Search for memories mentioning a specific entity."""
        query_context = QueryContext(
            query=entity_name,
            mentioned_entities=[entity_name],
            max_tokens=10000,  # Large token budget for entity search
            strategies=[RetrievalStrategy.ENTITY_BASED]
        )
        
        context_window = self.query(query_context)
        return context_window.memories[:max_count]

    def get_procedural_memories(self, task: str, max_count: int = 5) -> List[MemoryNode]:
        """Get procedural memories for a specific task."""
        query_context = QueryContext(
            query=f"how to {task}",
            intent=QueryIntent.PROCEDURAL,
            strategies=[RetrievalStrategy.PROCEDURAL],
            max_tokens=8000
        )
        
        context_window = self.query(query_context)
        return context_window.memories[:max_count]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        total_memories = len(self.memories)
        
        if total_memories == 0:
            return {'error': 'No memories loaded'}
        
        # Memory type distribution
        type_distribution = {}
        for memory_type in MemoryType:
            type_distribution[memory_type.value] = len(self.type_index[memory_type])
        
        # Temporal coverage
        if self.temporal_index:
            min_date = min(self.temporal_index.keys())
            max_date = max(self.temporal_index.keys())
            date_coverage = (max_date - min_date).days
        else:
            min_date = max_date = None
            date_coverage = 0
        
        # Importance distribution
        importance_scores = [m.importance_score for m in self.memories]
        avg_importance = sum(importance_scores) / len(importance_scores)
        
        return {
            'total_memories': total_memories,
            'memory_types': type_distribution,
            'temporal_coverage': {
                'earliest_date': min_date.isoformat() if min_date else None,
                'latest_date': max_date.isoformat() if max_date else None,
                'days_covered': date_coverage,
                'dates_with_memories': len(self.temporal_index)
            },
            'importance_stats': {
                'average_importance': round(avg_importance, 3),
                'high_importance_count': sum(1 for s in importance_scores if s > 0.7),
                'low_importance_count': sum(1 for s in importance_scores if s < 0.3)
            },
            'index_sizes': {
                'entity_terms': len(self.entity_index),
                'temporal_dates': len(self.temporal_index),
                'memory_types': len(self.type_index)
            }
        }


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test memory retrieval")
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("-m", "--memories", help="Path to memories JSON file")
    parser.add_argument("-t", "--max-tokens", type=int, default=2000, help="Maximum tokens for context")
    parser.add_argument("-s", "--strategy", choices=[s.value for s in RetrievalStrategy],
                        help="Specific retrieval strategy to use")
    parser.add_argument("--stats", action="store_true", help="Show retrieval system statistics")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize retriever
    retriever = MemoryRetriever(memory_file=args.memories)
    
    if args.stats:
        stats = retriever.get_retrieval_stats()
        print("Retrieval System Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        exit(0)
    
    # Build query context
    query_context = QueryContext(
        query=args.query,
        max_tokens=args.max_tokens
    )
    
    if args.strategy:
        query_context.strategies = [RetrievalStrategy(args.strategy)]
    
    # Perform query
    context_window = retriever.query(query_context)
    
    # Print results
    print(f"\nQuery: {args.query}")
    print(f"Intent: {context_window.assembly_metadata.get('intent', 'unknown')}")
    print(f"Strategies: {', '.join(context_window.assembly_metadata.get('strategies_used', []))}")
    print(f"Memories found: {len(context_window.memories)}")
    print(f"Estimated tokens: {context_window.total_tokens}")
    print("\n" + "="*60 + "\n")
    
    formatted_context = context_window.get_formatted_context()
    print(formatted_context)
    
    if context_window.assembly_metadata.get('fallback_used'):
        print("\n[Note: Fallback search was used]")